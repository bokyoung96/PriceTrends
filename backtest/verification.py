from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.config import BacktestConfig, BenchmarkType, EntryPriceMode
from backtest.engine import RebalancePlanner
from backtest.main import EXAMPLES, ExampleRunner
from backtest.runner import Backtester
from backtest.report import _select_font


@dataclass(frozen=True)
class SectorPick:
    sector: str
    long_ticker: str
    long_return: float
    short_ticker: str
    short_return: float


class SectorBacktestResult(NamedTuple):
    equity: pd.Series
    returns: pd.Series
    stats: dict[str, float]


def run_verification(
    example_name: str = "transformer_long_short_sector_neutral",
    *,
    base_opts: dict[str, object] | None = None,
) -> dict[str, Path]:
    spec = EXAMPLES.get(example_name)
    if spec is None:
        raise KeyError(f"Unknown example: {example_name}")

    runner = ExampleRunner(base_opts=base_opts or {})
    tester = runner.run_spec(spec)
    report = tester.latest_report()
    config = report.config

    out_dir = Path(config.output_dir) / "sector_performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    if spec.output_filename:
        stem = Path(spec.output_filename).stem
    else:
        stem = Path(report._auto_filename()).stem

    sector_results = run_sector_backtests(config)
    if sector_results:
        equity_frame = pd.DataFrame({k: v.equity for k, v in sector_results.items()}).sort_index()
        returns_frame = pd.DataFrame({k: v.returns for k, v in sector_results.items()}).sort_index()
        drawdown_frame = equity_frame.divide(equity_frame.cummax()).subtract(1.0).fillna(0.0)
        summary_frame = pd.DataFrame({k: v.stats for k, v in sector_results.items()}).T

        equity_frame.to_excel(out_dir / f"sector_equity_{stem}.xlsx")
        returns_frame.to_excel(out_dir / f"sector_returns_{stem}.xlsx")
        drawdown_frame.to_excel(out_dir / f"sector_drawdown_{stem}.xlsx")
        summary_frame.to_excel(out_dir / f"sector_summary_{stem}.xlsx")
        _save_sector_overview_plot(sector_results, out_dir, stem)
        _save_sector_subplot_plot(sector_results, out_dir, stem)
        _save_sector_summary_png(summary_frame, out_dir, stem)

    df = sector_top_bottom(config)
    xlsx_path = out_dir / f"sector_top_bottom_{stem}.xlsx"
    df.to_excel(xlsx_path, index=False)
    return {"xlsx": xlsx_path}


def sector_top_bottom(config: BacktestConfig) -> pd.DataFrame:
    dataset = config.data_loader().build()
    sector_panel = _load_sector_panel(config.sector_path)
    sector_frame = _align_sector_panel(sector_panel, dataset.scores.index, dataset.scores.columns)
    planner = RebalancePlanner(dataset.scores.index, config.rebalance_frequency, entry_lag=config.entry_lag)

    rows: list[dict[str, object]] = []
    for window in planner.windows():
        sector_row = sector_frame.loc[window.entry_date]
        if config.entry_price_mode == EntryPriceMode.NEXT_OPEN and dataset.open_prices is not None:
            entry_prices = dataset.open_prices.loc[window.entry_date]
        else:
            entry_prices = dataset.prices.loc[window.entry_date]
        exit_prices = dataset.prices.loc[window.exit_date]
        returns = (exit_prices / entry_prices) - 1.0

        for pick in _sector_picks(sector_row, returns):
            rows.append(
                {
                    "signal_date": window.signal_date,
                    "entry_date": window.entry_date,
                    "exit_date": window.exit_date,
                    "sector": pick.sector,
                    "long_ticker": pick.long_ticker,
                    "long_return": pick.long_return,
                    "short_ticker": pick.short_ticker,
                    "short_return": pick.short_return,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values(["entry_date", "sector"], inplace=True)
    return df


def sector_monthly_picks(config: BacktestConfig, sector_name: str) -> pd.DataFrame:
    picks = sector_top_bottom(config)
    if picks.empty:
        return picks
    filtered = picks[picks["sector"] == sector_name].copy()
    filtered.sort_values(["entry_date"], inplace=True)
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def sector_monthly_holdings(config: BacktestConfig, sector_name: str) -> pd.DataFrame:
    dataset = config.data_loader().build()
    sector_panel = _load_sector_panel(config.sector_path)
    sector_frame = _align_sector_panel(sector_panel, dataset.scores.index, dataset.scores.columns)
    sector_scores = dataset.scores.where(sector_frame == sector_name).dropna(axis=1, how="all")
    if sector_scores.empty:
        return pd.DataFrame()

    tester = Backtester(config.with_overrides(sector_neutral=False))
    report = tester.run(scores=sector_scores, prices=dataset.prices)
    short_ids = _short_group_ids(config)

    rows: list[dict[str, object]] = []
    for group_id, group in report.groups.items():
        if group_id == "net":
            continue
        side = "short" if group_id in short_ids else "long"
        for trade in group.trades:
            for pos in trade.positions:
                entry_value = float(pos.entry_value)
                exit_value = float(pos.exit_value)
                pos_return = 0.0 if entry_value == 0 else (exit_value / entry_value) - 1.0
                rows.append(
                    {
                        "entry_date": trade.enter_date,
                        "exit_date": trade.exit_date,
                        "sector": sector_name,
                        "side": side,
                        "group_id": group_id,
                        "ticker": pos.ticker,
                        "entry_price": pos.entry_price,
                        "exit_price": pos.exit_price,
                        "entry_value": entry_value,
                        "exit_value": exit_value,
                        "position_return": pos_return,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values(["entry_date", "side", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def run_sector_backtests(config: BacktestConfig) -> dict[str, SectorBacktestResult]:
    dataset = config.data_loader().build()
    sector_panel = _load_sector_panel(config.sector_path)
    sector_frame = _align_sector_panel(sector_panel, dataset.scores.index, dataset.scores.columns)
    sector_names = sorted(pd.unique(sector_frame.stack().dropna()))
    if not sector_names:
        return {}

    results: dict[str, SectorBacktestResult] = {}
    base_config = config.with_overrides(sector_neutral=False)
    for sector in sector_names:
        sector_scores = dataset.scores.where(sector_frame == sector)
        sector_scores = sector_scores.dropna(axis=1, how="all")
        if sector_scores.empty:
            continue
        tester = Backtester(base_config)
        report = tester.run(scores=sector_scores, prices=dataset.prices)
        net = report.groups.get("net")
        if net is None or net.equity_curve.empty:
            continue
        results[str(sector)] = SectorBacktestResult(
            equity=net.equity_curve,
            returns=net.period_returns,
            stats=net.stats,
        )
    return results


def _sector_picks(sector_row: pd.Series, returns: pd.Series) -> Iterable[SectorPick]:
    valid = returns.dropna()
    for sector, tickers in sector_row.groupby(sector_row):
        tickers = tickers.index.intersection(valid.index)
        if tickers.empty:
            continue
        sector_returns = valid.loc[tickers].astype(float)
        long_ticker = str(sector_returns.idxmax())
        short_ticker = str(sector_returns.idxmin())
        yield SectorPick(
            sector=str(sector),
            long_ticker=long_ticker,
            long_return=float(sector_returns.loc[long_ticker]),
            short_ticker=short_ticker,
            short_return=float(sector_returns.loc[short_ticker]),
        )


def _load_sector_panel(path: Path | str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "Date" in frame.columns and not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.set_index("Date")
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame.sort_index(inplace=True)
    return frame


def _align_sector_panel(
    sector_panel: pd.DataFrame,
    target_index: pd.Index,
    target_columns: pd.Index,
) -> pd.DataFrame:
    frame = sector_panel.reindex(target_index).ffill()
    frame = frame.reindex(columns=target_columns)
    return frame


def _short_group_ids(config: BacktestConfig) -> set[str]:
    if config.short_quantiles:
        return {f"q{int(idx) + 1}" for idx in config.short_quantiles}
    return {"q1"}


def _save_sector_overview_plot(
    sector_results: dict[str, SectorBacktestResult],
    out_dir: Path,
    stem: str,
) -> None:
    font_name = _select_font()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name],
            "axes.unicode_minus": False,
        }
    )
    sectors = [name for name, result in sector_results.items() if not result.equity.dropna().empty]
    if not sectors:
        return

    palette = _extended_palette(len(sectors))
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.4), sharex=True, dpi=180)
    for idx, sector in enumerate(sectors):
        equity = sector_results[sector].equity.dropna()
        if equity.empty:
            continue
        cumulative = equity / equity.iloc[0] - 1.0
        drawdown = equity.divide(equity.cummax()).subtract(1.0).fillna(0.0)
        color = palette[idx]
        axes[0].plot(cumulative.index, cumulative.values, color=color, linewidth=1.6, label=sector)
        axes[1].plot(drawdown.index, drawdown.values, color=color, linewidth=1.2, alpha=0.85)

    axes[0].set_title("Sector Cumulative Return (net)", fontsize=11)
    axes[0].set_ylabel("Cumulative Return")
    axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    axes[0].legend(loc="upper left", ncol=4, frameon=False, fontsize=8.5)

    axes[1].set_title("Sector Drawdown", fontsize=11)
    axes[1].set_ylabel("Drawdown")
    axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    axes[1].axhline(0, color="#D1D1D6", linewidth=1)

    for axis in axes:
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)

    path = out_dir / f"sector_overview_{stem}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _save_sector_subplot_plot(
    sector_results: dict[str, SectorBacktestResult],
    out_dir: Path,
    stem: str,
) -> None:
    font_name = _select_font()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name],
            "axes.unicode_minus": False,
        }
    )
    sectors = [name for name, result in sector_results.items() if not result.equity.dropna().empty]
    if not sectors:
        return
    cols = 3
    rows = int((len(sectors) + cols - 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), dpi=180, sharex=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, sector in enumerate(sectors):
        axis = axes[idx]
        equity = sector_results[sector].equity.dropna()
        if equity.empty:
            axis.axis("off")
            continue
        cumulative = equity / equity.iloc[0] - 1.0
        drawdown = equity.divide(equity.cummax()).subtract(1.0).fillna(0.0)
        axis.plot(cumulative.index, cumulative.values, color="#4F5DFF", linewidth=1.4, label="Cum")
        axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        twin = axis.twinx()
        twin.plot(drawdown.index, drawdown.values, color="#FF6B81", linewidth=1.0, alpha=0.8, label="MDD")
        twin.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        axis.set_title(str(sector), fontsize=10)
        axis.grid(False)
        twin.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for spine in twin.spines.values():
            spine.set_visible(False)
        if idx % cols == 0:
            axis.set_ylabel("Cum")
        if idx % cols == cols - 1:
            twin.set_ylabel("MDD")

    for axis in axes[len(sectors):]:
        axis.axis("off")

    path = out_dir / f"sector_subplots_{stem}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _save_sector_summary_png(summary: pd.DataFrame, out_dir: Path, stem: str) -> None:
    if summary.empty:
        return
    columns = ["total_return", "cagr", "sharpe", "max_drawdown"]
    table = summary[[col for col in columns if col in summary.columns]].copy()
    table = table.sort_values("total_return", ascending=False)
    table = table.apply(lambda col: col.map(lambda val: _format_stat(col.name, val)))

    fig, ax = plt.subplots(figsize=(7.5, 0.4 + 0.3 * len(table)), dpi=180)
    ax.axis("off")
    tbl = ax.table(
        cellText=table.to_numpy(),
        rowLabels=table.index,
        colLabels=table.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.3)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#F4F5F7")
        if row == 0:
            cell.set_facecolor("#E5E5EA")
            cell.get_text().set_fontweight("semibold")
        elif col == -1:
            cell.set_facecolor("#FFFFFF")
            cell.get_text().set_fontweight("semibold")
            cell.get_text().set_color("#6C6C70")
        elif row % 2 == 0:
            cell.set_facecolor("#FFFFFF")
        else:
            cell.set_facecolor("#F8F8FA")
    fig.tight_layout()
    path = out_dir / f"sector_summary_{stem}.png"
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _format_stat(column: str, value: float) -> str:
    if isinstance(value, (int, float)):
        if column in {"total_return", "cagr", "max_drawdown"}:
            return f"{value * 100:0.2f}%"
        if column in {"sharpe", "volatility"}:
            return f"{value:0.2f}"
        return f"{value:0.4f}"
    return str(value)


def _extended_palette(size: int) -> list:
    base = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        base.extend([cmap(i) for i in range(cmap.N)])
    if size <= len(base):
        return base[:size]
    repeats = int((size + len(base) - 1) / len(base))
    return (base * repeats)[:size]


if __name__ == "__main__":
    default_opts = dict(
        rebalance_frequency="M",
        portfolio_weighting="eq",
        apply_trading_costs=False,
        buy_cost_bps=2.0,
        sell_cost_bps=2.0,
        tax_bps=15.0,
        entry_lag=0,
        entry_price_mode="close",
        benchmark_symbol=BenchmarkType.KOSPI200,
        start_date="2012-01-31",
    )
    paths = run_verification(base_opts=default_opts)
    print(f"Saved verification outputs: {paths}")
    runner = ExampleRunner(base_opts=default_opts)
    tester = runner.run_spec(EXAMPLES["transformer_long_short_sector_neutral"])
    health_df = sector_monthly_holdings(tester.latest_report().config, "건강관리")
    print(health_df.head())
