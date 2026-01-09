from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.config import BacktestConfig, BenchmarkType, EntryPriceMode, LongShortMode
from backtest.data_sources import BacktestDataset
from backtest.engine import BacktestEngine, RebalancePlanner
from backtest.main import EXAMPLES, ExampleRunner, ExampleSpec
from backtest.runner import Backtester
from backtest.report import BacktestReport, _select_font


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
    long_equity: pd.Series
    short_equity: pd.Series


def _pick_example(name: str | None) -> ExampleSpec:
    if name is None:
        if "ls_sn_global" in EXAMPLES:
            return EXAMPLES["ls_sn_global"]
        if EXAMPLES:
            first_key = next(iter(EXAMPLES))
            return EXAMPLES[first_key]
        raise KeyError("No examples available.")
    if name not in EXAMPLES:
        raise KeyError(f"Unknown example: {name}")
    return EXAMPLES[name]


def _sector_frame(config: BacktestConfig, dataset: BacktestDataset) -> pd.DataFrame:
    sector_panel = _load_sector_panel(config.sector_path)
    return _align_sector_panel(sector_panel, dataset.scores.index, dataset.scores.columns)


def _jaccard(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _mean_jaccard(date_map: dict[pd.Timestamp, set[str]]) -> float | None:
    dates = sorted(date_map)
    if len(dates) < 2:
        return None
    values: list[float] = []
    prev = date_map[dates[0]]
    for date in dates[1:]:
        current = date_map[date]
        value = _jaccard(prev, current)
        if value is not None:
            values.append(float(value))
        prev = current
    if not values:
        return None
    return float(pd.Series(values).mean())


def _sel_stability(
    report: BacktestReport,
    sector_frame: pd.DataFrame,
    short_ids: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    buckets = {"long": {}, "short": {}}
    for group_id, group in report.groups.items():
        if group_id == "net":
            continue
        side = "short" if group_id in short_ids else "long"
        for trade in group.trades:
            date = trade.enter_date
            if date not in sector_frame.index:
                continue
            sector_row = sector_frame.loc[date]
            for pos in trade.positions:
                sector = sector_row.get(pos.ticker)
                if pd.isna(sector):
                    continue
                sec = str(sector)
                buckets[side].setdefault(sec, {}).setdefault(date, set()).add(str(pos.ticker))
    long_stats: dict[str, float] = {}
    short_stats: dict[str, float] = {}
    for sector, date_map in buckets["long"].items():
        value = _mean_jaccard(date_map)
        if value is not None:
            long_stats[sector] = value
    for sector, date_map in buckets["short"].items():
        value = _mean_jaccard(date_map)
        if value is not None:
            short_stats[sector] = value
    return long_stats, short_stats


def _ls_union_j(
    report: BacktestReport,
    sector_frame: pd.DataFrame,
) -> dict[str, float]:
    buckets: dict[str, dict[pd.Timestamp, set[str]]] = {}
    for group_id, group in report.groups.items():
        if group_id == "net":
            continue
        for trade in group.trades:
            date = trade.enter_date
            if date not in sector_frame.index:
                continue
            sector_row = sector_frame.loc[date]
            for pos in trade.positions:
                sector = sector_row.get(pos.ticker)
                if pd.isna(sector):
                    continue
                sec = str(sector)
                buckets.setdefault(sec, {}).setdefault(date, set()).add(str(pos.ticker))
    results: dict[str, float] = {}
    for sector, date_map in buckets.items():
        value = _mean_jaccard(date_map)
        if value is not None:
            results[sector] = value
    return results


def _sum_equity(groups: dict, ids: Iterable[str]) -> pd.Series:
    ids = list(ids)
    if not ids:
        return pd.Series(dtype=float)
    df = pd.concat([groups[gid].equity_curve for gid in ids], axis=1).ffill()
    if df.empty:
        return pd.Series(dtype=float)
    return df.sum(axis=1).sort_index()


def _leg_equity(report: BacktestReport, short_ids: set[str]) -> tuple[pd.Series, pd.Series]:
    long_ids = [gid for gid in report.groups if gid != "net" and gid not in short_ids]
    short_ids_in = [gid for gid in report.groups if gid in short_ids]
    return _sum_equity(report.groups, long_ids), _sum_equity(report.groups, short_ids_in)


def _attrib_report(report: BacktestReport, dataset: BacktestDataset) -> BacktestReport:
    cfg = report.config
    if cfg.long_short_mode is not LongShortMode.NET or not cfg.dollar_neutral_net:
        return report
    score_paths = getattr(cfg, "scores_path", None)
    if not score_paths:
        return report
    if not all(Path(path).exists() for path in score_paths):
        return report
    engine = BacktestEngine(cfg, dataset)
    return engine.run_raw()


def _sector_attrib_frame(report: BacktestReport, sector_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[tuple[pd.Timestamp, str, float]] = []
    for group_id, group in report.groups.items():
        if group_id == "net":
            continue
        for trade in group.trades:
            entry_date = trade.enter_date
            if entry_date not in sector_frame.index:
                continue
            sector_row = sector_frame.loc[entry_date]
            for pos in trade.positions:
                sector = sector_row.get(pos.ticker)
                if pd.isna(sector):
                    continue
                pnl = float(pos.exit_value) - float(pos.entry_value)
                rows.append((trade.exit_date, str(sector), pnl))
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["exit_date", "sector", "pnl"])
    pivot = frame.pivot_table(index="exit_date", columns="sector", values="pnl", aggfunc="sum")
    return pivot.sort_index().fillna(0.0)


def _maybe_legend(axis: plt.Axes, **kwargs) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, labels, **kwargs)


def _maybe_legend_multi(axis: plt.Axes, *axes: plt.Axes, **kwargs) -> None:
    handles, labels = axis.get_legend_handles_labels()
    for other in axes:
        extra_h, extra_l = other.get_legend_handles_labels()
        handles += extra_h
        labels += extra_l
    if handles:
        axis.legend(handles, labels, **kwargs)


def _plot_style() -> None:
    font_name = _select_font()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.sans-serif": [font_name],
            "axes.unicode_minus": False,
        }
    )


def _monthly_returns(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return equity
    monthly = equity.resample("ME").last().dropna(how="all")
    if monthly.empty:
        return monthly
    if len(monthly.index) < 2:
        return pd.DataFrame()
    returns = monthly.pct_change().dropna(how="all")
    if returns.empty:
        return pd.DataFrame()
    entry_map = pd.Series(monthly.index[:-1], index=returns.index)
    long_df = returns.stack().reset_index()
    long_df.columns = ["exit_date", "sector", "monthly_return"]
    long_df["entry_date"] = long_df["exit_date"].map(entry_map)
    return long_df[["entry_date", "exit_date", "sector", "monthly_return"]]


def _monthly_return_series(equity: pd.Series) -> pd.Series:
    series = equity.dropna()
    if series.empty:
        return pd.Series(dtype=float)
    monthly = series.resample("ME").last().dropna()
    if len(monthly.index) < 2:
        return pd.Series(dtype=float)
    return monthly.pct_change().dropna()


def _monthly_returns_leg(equity: pd.DataFrame, leg: str) -> pd.DataFrame:
    frame = _monthly_returns(equity)
    if frame.empty:
        return frame
    frame["leg"] = leg
    return frame[["entry_date", "exit_date", "sector", "leg", "monthly_return"]]


def _daily_return_series(equity: pd.Series) -> pd.Series:
    series = equity.dropna()
    if series.empty:
        return pd.Series(dtype=float)
    returns = series.pct_change().dropna()
    return returns


def _cum_dd(equity: pd.Series) -> tuple[pd.Series, pd.Series]:
    eq = equity.dropna()
    if eq.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    cumulative = eq / eq.iloc[0] - 1.0
    drawdown = eq.divide(eq.cummax()).subtract(1.0).fillna(0.0)
    return cumulative, drawdown


def _sec_cfg(config: BacktestConfig) -> BacktestConfig:
    if config.sector_neutral or config.sector_unit:
        return config.with_overrides(sector_neutral=False, sector_unit=False)
    return config


def run_verification(
    example_name: str | None = None,
    *,
    base_opts: dict[str, object] | None = None,
) -> dict[str, Path]:
    spec = _pick_example(example_name)

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

    dataset = config.data_loader().build()
    sector_frame = _sector_frame(config, dataset)
    sector_results = run_sector_backtests(config, dataset=dataset, sector_frame=sector_frame)
    if sector_results:
        sectors = list(sector_results)
        equity_frame = pd.DataFrame({k: v.equity for k, v in sector_results.items()}).sort_index()
        long_equity_frame = pd.DataFrame({k: v.long_equity for k, v in sector_results.items()}).sort_index()
        short_equity_frame = pd.DataFrame({k: v.short_equity for k, v in sector_results.items()}).sort_index()

        net_cols = {k: v.returns for k, v in sector_results.items()}
        long_cols = {f"{k}_long": v.long_equity.pct_change().fillna(0.0) for k, v in sector_results.items()}
        short_cols = {f"{k}_short": v.short_equity.pct_change().fillna(0.0) for k, v in sector_results.items()}
        returns_frame = pd.DataFrame({**net_cols, **long_cols, **short_cols}).sort_index()
        ordered_cols: list[str] = []
        for sector in sectors:
            if sector in returns_frame.columns:
                ordered_cols.append(sector)
            long_name = f"{sector}_long"
            if long_name in returns_frame.columns:
                ordered_cols.append(long_name)
            short_name = f"{sector}_short"
            if short_name in returns_frame.columns:
                ordered_cols.append(short_name)
        returns_frame = returns_frame.reindex(columns=ordered_cols)
        drawdown_frame = equity_frame.divide(equity_frame.cummax()).subtract(1.0).fillna(0.0)
        monthly_frames = [
            _monthly_returns_leg(equity_frame, "net"),
            _monthly_returns_leg(long_equity_frame, "long"),
            _monthly_returns_leg(short_equity_frame, "short"),
        ]
        monthly_frames = [frame for frame in monthly_frames if not frame.empty]
        monthly_frame = pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
        if not monthly_frame.empty:
            monthly_frame.sort_values(["entry_date", "exit_date", "sector", "leg"], inplace=True)
        summary_frame = pd.DataFrame({k: v.stats for k, v in sector_results.items()}).T
        summary_frame["monthly_avg_return"] = 0.0
        summary_frame["monthly_win_rate"] = 0.0
        for sector, result in sector_results.items():
            monthly = _monthly_return_series(result.equity)
            if monthly.empty:
                continue
            summary_frame.loc[sector, "monthly_avg_return"] = float(monthly.mean())
            summary_frame.loc[sector, "monthly_win_rate"] = float((monthly > 0).mean())
        short_ids = _short_group_ids(config)
        comp_stats = _ls_union_j(report, sector_frame)
        long_stats, short_stats = _sel_stability(report, sector_frame, short_ids)
        summary_frame["comp_j"] = summary_frame.index.map(comp_stats.get)
        summary_frame["long_j"] = summary_frame.index.map(long_stats.get)
        summary_frame["short_j"] = summary_frame.index.map(short_stats.get)

        equity_frame.to_excel(out_dir / f"sector_equity_{stem}.xlsx")
        returns_frame.to_excel(out_dir / f"sector_returns_{stem}.xlsx")
        if not monthly_frame.empty:
            monthly_frame.to_excel(out_dir / f"sector_returns_m_{stem}.xlsx")
        drawdown_frame.to_excel(out_dir / f"sector_drawdown_{stem}.xlsx")
        summary_frame.to_excel(out_dir / f"sector_summary_{stem}.xlsx")
        _save_sector_overview_plot(sector_results, out_dir, stem)
        _save_sector_subplots_all_plot(sector_results, out_dir, stem)
        _save_sector_subplots_ls_plot(sector_results, out_dir, stem)
        _save_sector_subplots_hist_plot(sector_results, out_dir, stem)
        _save_sector_subplots_hist_monthly_plot(sector_results, out_dir, stem)
        _save_sector_subplot_plot(sector_results, out_dir, stem)
        _save_sector_summary_png(summary_frame, out_dir, stem)

    attrib_report = _attrib_report(report, dataset)
    attrib_frame = _sector_attrib_frame(attrib_report, sector_frame)
    if not attrib_frame.empty and config.initial_capital:
        attrib_cum = attrib_frame.div(float(config.initial_capital)).cumsum().sort_index()
        _save_sector_attribution_plot(attrib_cum, out_dir, stem)

    df = sector_top_bottom(config, dataset=dataset, sector_frame=sector_frame)
    xlsx_path = out_dir / f"sector_top_bottom_{stem}.xlsx"
    df.to_excel(xlsx_path, index=False)
    return {"xlsx": xlsx_path}


def sector_top_bottom(
    config: BacktestConfig,
    *,
    dataset: BacktestDataset | None = None,
    sector_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if dataset is None:
        dataset = config.data_loader().build()
    if sector_frame is None:
        sector_frame = _sector_frame(config, dataset)
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
    sector_frame = _sector_frame(config, dataset)
    sector_scores = dataset.scores.where(sector_frame == sector_name).dropna(axis=1, how="all")
    if sector_scores.empty:
        return pd.DataFrame()

    sector_prices = dataset.prices.reindex(columns=sector_scores.columns)
    sector_weights = dataset.weights.reindex(columns=sector_scores.columns) if dataset.weights is not None else None
    sector_open = dataset.open_prices.reindex(columns=sector_scores.columns) if dataset.open_prices is not None else None
    sector_dataset = BacktestDataset(
        scores=sector_scores,
        prices=sector_prices,
        bench=dataset.bench,
        weights=sector_weights,
        open_prices=sector_open,
    )

    base_config = _sec_cfg(config)
    base_engine = BacktestEngine(base_config, sector_dataset)
    report = base_engine.run()
    short_ids = _short_group_ids(base_config)
    dn_map: dict[tuple[pd.Timestamp, pd.Timestamp, str, str, str], tuple[float, float, float, float, float, float]] = {}
    dn_net_map: dict[tuple[pd.Timestamp, pd.Timestamp], tuple[float, float, float, float]] = {}
    if base_config.long_short_mode is LongShortMode.NET and base_config.dollar_neutral_net:
        dn_engine = BacktestEngine(base_config, sector_dataset)
        dn_report = dn_engine.run_raw()
        dn_map = _build_position_map(dn_report, short_ids)
        dn_net_map = _build_net_window_map(dn_report, short_ids)

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
                key = (trade.enter_date, trade.exit_date, side, group_id, str(pos.ticker))
                dn_entry_value = None
                dn_exit_value = None
                dn_pos_return = None
                dn_trade_in = None
                dn_trade_out = None
                dn_trade_return = None
                dn_net_gross = None
                dn_net_equity_entry = None
                dn_net_equity_exit = None
                dn_net_return = None
                dn_values = dn_map.get(key)
                if dn_values is not None:
                    (
                        dn_entry_value,
                        dn_exit_value,
                        dn_pos_return,
                        dn_trade_in,
                        dn_trade_out,
                        dn_trade_return,
                    ) = dn_values
                dn_net_values = dn_net_map.get((trade.enter_date, trade.exit_date))
                if dn_net_values is not None:
                    (
                        dn_net_gross,
                        dn_net_equity_entry,
                        dn_net_equity_exit,
                        dn_net_return,
                    ) = dn_net_values
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
                        "dn_entry_value": dn_entry_value,
                        "dn_exit_value": dn_exit_value,
                        "dn_position_return": dn_pos_return,
                        "dn_trade_capital_in": dn_trade_in,
                        "dn_trade_capital_out": dn_trade_out,
                        "dn_trade_return": dn_trade_return,
                        "dn_net_gross": dn_net_gross,
                        "dn_net_equity_entry": dn_net_equity_entry,
                        "dn_net_equity_exit": dn_net_equity_exit,
                        "dn_net_return": dn_net_return,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.sort_values(["entry_date", "side", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def run_sector_backtests(
    config: BacktestConfig,
    *,
    dataset: BacktestDataset | None = None,
    sector_frame: pd.DataFrame | None = None,
) -> dict[str, SectorBacktestResult]:
    if dataset is None:
        dataset = config.data_loader().build()
    if sector_frame is None:
        sector_frame = _sector_frame(config, dataset)
    sector_names = sorted(pd.unique(sector_frame.stack().dropna()))
    if not sector_names:
        return {}

    results: dict[str, SectorBacktestResult] = {}
    base_config = _sec_cfg(config)
    short_ids = _short_group_ids(base_config)
    for sector in sector_names:
        sector_scores = dataset.scores.where(sector_frame == sector)
        sector_scores = sector_scores.dropna(axis=1, how="all")
        if sector_scores.empty:
            continue
        # Build a sector-only dataset to avoid ffill on masked scores.
        sector_prices = dataset.prices.reindex(columns=sector_scores.columns)
        sector_weights = dataset.weights.reindex(columns=sector_scores.columns) if dataset.weights is not None else None
        sector_open = dataset.open_prices.reindex(columns=sector_scores.columns) if dataset.open_prices is not None else None
        sector_dataset = BacktestDataset(
            scores=sector_scores,
            prices=sector_prices,
            bench=dataset.bench,
            weights=sector_weights,
            open_prices=sector_open,
        )
        report = BacktestEngine(base_config, sector_dataset).run()
        net = report.groups.get("net")
        if net is None or net.equity_curve.empty:
            continue
        long_eq, short_eq = _leg_equity(report, short_ids)
        results[str(sector)] = SectorBacktestResult(
            equity=net.equity_curve,
            returns=net.period_returns,
            stats=net.stats,
            long_equity=long_eq,
            short_equity=short_eq,
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


def _build_position_map(
    report: BacktestReport,
    short_ids: set[str],
) -> dict[tuple[pd.Timestamp, pd.Timestamp, str, str, str], tuple[float, float, float, float, float, float]]:
    mapping: dict[tuple[pd.Timestamp, pd.Timestamp, str, str, str], tuple[float, float, float, float, float, float]] = {}
    for group_id, group in report.groups.items():
        if group_id == "net":
            continue
        side = "short" if group_id in short_ids else "long"
        for trade in group.trades:
            trade_in = float(trade.capital_in)
            trade_out = float(trade.capital_out)
            trade_return = float(trade.period_return)
            for pos in trade.positions:
                entry_value = float(pos.entry_value)
                exit_value = float(pos.exit_value)
                pos_return = 0.0 if entry_value == 0 else (exit_value / entry_value) - 1.0
                key = (trade.enter_date, trade.exit_date, side, group_id, str(pos.ticker))
                mapping[key] = (
                    entry_value,
                    exit_value,
                    pos_return,
                    trade_in,
                    trade_out,
                    trade_return,
                )
    return mapping


def _build_net_window_map(
    report: BacktestReport,
    short_ids: set[str],
) -> dict[tuple[pd.Timestamp, pd.Timestamp], tuple[float, float, float, float]]:
    net = report.groups.get("net")
    if net is None or net.equity_curve.empty:
        return {}
    long_ids = [gid for gid in report.groups.keys() if gid != "net" and gid not in short_ids]
    short_ids_in_report = [gid for gid in report.groups.keys() if gid in short_ids]
    if not long_ids or not short_ids_in_report:
        return {}
    eq_map = {
        gid: report.groups[gid].equity_curve.sort_index()
        for gid in long_ids + short_ids_in_report
    }
    net_equity = net.equity_curve.sort_index()

    windows: set[tuple[pd.Timestamp, pd.Timestamp]] = set()
    for gid in long_ids + short_ids_in_report:
        for trade in report.groups[gid].trades:
            windows.add((trade.enter_date, trade.exit_date))

    mapping: dict[tuple[pd.Timestamp, pd.Timestamp], tuple[float, float, float, float]] = {}
    for enter_date, exit_date in sorted(windows):
        gross = 0.0
        for series in eq_map.values():
            gross += _equity_at(series, enter_date)
        entry_equity = _equity_at(net_equity, enter_date)
        exit_equity = _equity_at(net_equity, exit_date)
        net_return = 0.0 if entry_equity == 0 else (exit_equity / entry_equity) - 1.0
        mapping[(enter_date, exit_date)] = (gross, entry_equity, exit_equity, float(net_return))
    return mapping


def _equity_at(equity: pd.Series, date: pd.Timestamp) -> float:
    if equity.empty:
        return 0.0
    if date in equity.index:
        return float(equity.loc[date])
    prior = equity.loc[:date]
    if prior.empty:
        return float(equity.iloc[0])
    return float(prior.iloc[-1])


def _save_sector_overview_plot(
    sector_results: dict[str, SectorBacktestResult],
    out_dir: Path,
    stem: str,
) -> None:
    _plot_style()
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
    _maybe_legend(axes[0], loc="upper left", ncol=4, frameon=False, fontsize=8.5)

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


def _save_sector_subplots_all_plot(
    sector_results: dict[str, SectorBacktestResult],
    out_dir: Path,
    stem: str,
) -> None:
    _plot_style()
    sectors = [name for name, result in sector_results.items() if not result.equity.dropna().empty]
    if not sectors:
        return

    cols = 3
    rows = int((len(sectors) + cols - 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), dpi=180, sharex=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, sector in enumerate(sectors):
        axis = axes[idx]
        result = sector_results[sector]
        series_map = {
            "net": result.equity,
            "long": result.long_equity,
            "short": result.short_equity,
        }
        colors = {"net": "#4F5DFF", "long": "#22C55E", "short": "#FF6B81"}
        for label, equity in series_map.items():
            if equity is None or equity.dropna().empty:
                continue
            cumulative = equity / equity.iloc[0] - 1.0
            axis.plot(cumulative.index, cumulative.values, color=colors[label], linewidth=1.2, label=label)
        axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        axis.set_title(str(sector), fontsize=10)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        _maybe_legend(axis, loc="upper left", ncol=3, frameon=False, fontsize=7.5)

    for axis in axes[len(sectors):]:
        axis.axis("off")

    path = out_dir / f"sector_subplots_all_{stem}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _save_sector_subplots_ls_plot(
    sector_results: dict[str, SectorBacktestResult],
    out_dir: Path,
    stem: str,
) -> None:
    _plot_style()
    sectors = [name for name, result in sector_results.items() if not result.equity.dropna().empty]
    if not sectors:
        return

    cols = 3
    rows = int((len(sectors) + cols - 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), dpi=180, sharex=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, sector in enumerate(sectors):
        axis = axes[idx]
        result = sector_results[sector]
        long_cum, long_dd = _cum_dd(result.long_equity)
        short_cum, short_dd = _cum_dd(result.short_equity)
        has_long = not long_cum.empty
        has_short = not short_cum.empty
        if not has_long and not has_short:
            axis.axis("off")
            continue

        if has_long:
            axis.plot(
                long_cum.index,
                long_cum.values,
                color="#15803D",
                linewidth=1.4,
                marker="o",
                markevery=6,
                markersize=2.5,
                label="L",
            )
        if has_short:
            axis.plot(
                short_cum.index,
                short_cum.values,
                color="#E11D48",
                linewidth=1.4,
                marker="o",
                markevery=6,
                markersize=2.5,
                label="S",
            )
        axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        axis.set_title(str(sector), fontsize=10)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)

        twin = axis.twinx()
        if has_long:
            twin.plot(
                long_dd.index,
                long_dd.values,
                color="#86EFAC",
                linewidth=0.9,
                linestyle="--",
                alpha=0.7,
                label="L-dd",
            )
        if has_short:
            twin.plot(
                short_dd.index,
                short_dd.values,
                color="#FDA4AF",
                linewidth=0.9,
                linestyle="--",
                alpha=0.7,
                label="S-dd",
            )
        twin.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        twin.grid(False)
        for spine in twin.spines.values():
            spine.set_visible(False)
        _maybe_legend_multi(axis, twin, loc="upper left", ncol=2, frameon=False, fontsize=7.0)

    for axis in axes[len(sectors):]:
        axis.axis("off")

    path = out_dir / f"sector_subplots_ls_{stem}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _save_sector_subplots_hist_plot(
    sector_results: dict[str, SectorBacktestResult],
    out_dir: Path,
    stem: str,
) -> None:
    _plot_style()
    sectors = [name for name, result in sector_results.items() if not result.equity.dropna().empty]
    if not sectors:
        return

    cols = 3
    rows = int((len(sectors) + cols - 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), dpi=180, sharex=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    colors = {"net": "#4F5DFF", "long": "#22C55E", "short": "#FF6B81"}

    for idx, sector in enumerate(sectors):
        axis = axes[idx]
        result = sector_results[sector]
        series_map = {
            "net": result.equity,
            "long": result.long_equity,
            "short": result.short_equity,
        }
        returns_map = {
            label: _daily_return_series(series) for label, series in series_map.items()
        }
        values = [
            (returns * 100.0).to_numpy()
            for returns in returns_map.values()
            if returns is not None and not returns.empty
        ]
        if not values:
            axis.axis("off")
            continue
        combined = np.concatenate(values)
        vmin = float(np.nanmin(combined))
        vmax = float(np.nanmax(combined))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            axis.axis("off")
            continue
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        pad = (vmax - vmin) * 0.05
        bins = np.linspace(vmin - pad, vmax + pad, 50)

        for label in ("long", "short"):
            data = returns_map[label]
            if data is None or data.empty:
                continue
            axis.hist(
                data * 100.0,
                bins=bins,
                color=colors[label],
                alpha=0.35,
                edgecolor="#FFFFFF",
                linewidth=0.5,
                label=label,
            )
        net_data = returns_map["net"]
        if net_data is not None and not net_data.empty:
            axis.hist(
                net_data * 100.0,
                bins=bins,
                color=colors["net"],
                alpha=0.85,
                edgecolor=colors["net"],
                linewidth=1.0,
                label="net",
            )

        axis.axvline(0, color="#8E8E93", linewidth=0.8)
        axis.set_title(str(sector), fontsize=10)
        axis.set_xlabel("Daily Return (%)", fontsize=8.5)
        axis.set_ylabel("Count", fontsize=8.5)
        axis.tick_params(axis="both", labelsize=8)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        _maybe_legend(axis, loc="upper right", ncol=1, frameon=False, fontsize=7.5)

    for axis in axes[len(sectors):]:
        axis.axis("off")

    path = out_dir / f"sector_subplots_hist_{stem}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _save_sector_subplots_hist_monthly_plot(
    sector_results: dict[str, SectorBacktestResult],
    out_dir: Path,
    stem: str,
) -> None:
    _plot_style()
    sectors = [name for name, result in sector_results.items() if not result.equity.dropna().empty]
    if not sectors:
        return

    cols = 3
    rows = int((len(sectors) + cols - 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), dpi=180, sharex=False)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    colors = {"net": "#4F5DFF", "long": "#22C55E", "short": "#FF6B81"}

    for idx, sector in enumerate(sectors):
        axis = axes[idx]
        result = sector_results[sector]
        series_map = {
            "net": result.equity,
            "long": result.long_equity,
            "short": result.short_equity,
        }
        returns_map = {
            label: _monthly_return_series(series) for label, series in series_map.items()
        }
        values = [
            (returns * 100.0).to_numpy()
            for returns in returns_map.values()
            if returns is not None and not returns.empty
        ]
        if not values:
            axis.axis("off")
            continue
        combined = np.concatenate(values)
        vmin = float(np.nanmin(combined))
        vmax = float(np.nanmax(combined))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            axis.axis("off")
            continue
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        pad = (vmax - vmin) * 0.05
        bins = np.linspace(vmin - pad, vmax + pad, 50)

        for label in ("long", "short"):
            data = returns_map[label]
            if data is None or data.empty:
                continue
            axis.hist(
                data * 100.0,
                bins=bins,
                color=colors[label],
                alpha=0.35,
                edgecolor="#FFFFFF",
                linewidth=0.5,
                label=label,
            )
        net_data = returns_map["net"]
        if net_data is not None and not net_data.empty:
            axis.hist(
                net_data * 100.0,
                bins=bins,
                color=colors["net"],
                alpha=0.85,
                edgecolor=colors["net"],
                linewidth=1.0,
                label="net",
            )

        axis.axvline(0, color="#8E8E93", linewidth=0.8)
        axis.set_title(str(sector), fontsize=10)
        axis.set_xlabel("Monthly Return (%)", fontsize=8.5)
        axis.set_ylabel("Count", fontsize=8.5)
        axis.tick_params(axis="both", labelsize=8)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        _maybe_legend(axis, loc="upper right", ncol=1, frameon=False, fontsize=7.5)

    for axis in axes[len(sectors):]:
        axis.axis("off")

    path = out_dir / f"sector_subplots_hist_m_{stem}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _save_sector_subplot_plot(
    sector_results: dict[str, SectorBacktestResult],
    out_dir: Path,
    stem: str,
) -> None:
    _plot_style()
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
    _plot_style()
    columns = [
        "total_return",
        "cagr",
        "sharpe",
        "max_drawdown",
        "monthly_avg_return",
        "monthly_win_rate",
        "comp_j",
        "long_j",
        "short_j",
    ]
    table = summary[[col for col in columns if col in summary.columns]].copy()
    table = table.sort_values("total_return", ascending=False)
    table = table.apply(lambda col: col.map(lambda val: _format_stat(col.name, val)))
    label_map = {
        "total_return": "TotRet",
        "cagr": "CAGR",
        "sharpe": "Sharpe",
        "max_drawdown": "MDD",
        "monthly_avg_return": "M.Avg",
        "monthly_win_rate": "M.Win",
        "comp_j": "CompJ",
        "long_j": "LJ",
        "short_j": "SJ",
    }
    table = table.rename(columns=label_map)

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


def _save_sector_attribution_plot(
    cumulative: pd.DataFrame,
    out_dir: Path,
    stem: str,
) -> None:
    if cumulative.empty:
        return
    _plot_style()
    sectors = list(cumulative.columns)
    palette = _extended_palette(len(sectors))
    fig, ax = plt.subplots(figsize=(12.0, 5.8), dpi=180)
    pos = cumulative.clip(lower=0.0)
    neg = cumulative.clip(upper=0.0)
    pos_list = [pos[sector].to_numpy() for sector in sectors]
    neg_list = [neg[sector].to_numpy() for sector in sectors]
    ax.stackplot(cumulative.index, pos_list, labels=sectors, colors=palette, alpha=0.9)
    if neg.values.any():
        ax.stackplot(cumulative.index, neg_list, colors=palette, alpha=0.45)
    ax.axhline(0.0, color="#D1D1D6", linewidth=1)
    ax.set_title("Sector Attribution (cumulative, signed)", fontsize=11)
    ax.set_ylabel("Cumulative Contribution")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    _maybe_legend(ax, loc="upper left", ncol=4, frameon=False, fontsize=8.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    path = out_dir / f"sector_attribution_{stem}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def _format_stat(column: str, value: float) -> str:
    if isinstance(value, (int, float)):
        if column in {
            "total_return",
            "cagr",
            "max_drawdown",
            "monthly_avg_return",
            "monthly_win_rate",
            "comp_j",
            "long_j",
            "short_j",
        }:
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
        end_date="2025-10-31",
    )
    name = "ls_sn_unit_tb3"
    paths = run_verification(example_name=name, base_opts=default_opts)
    print(f"Saved verification outputs: {paths}")
    runner = ExampleRunner(base_opts=default_opts)
    tester = runner.run_spec(_pick_example(name))
    df = sector_monthly_holdings(tester.latest_report().config, "Industrials")
    print(df.head())
