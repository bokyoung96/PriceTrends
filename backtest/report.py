from __future__ import annotations

import calendar
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects, ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.config import BacktestConfig
from backtest.portfolio import TradeRecord


@dataclass(frozen=True)
class PortfolioReport:
    """Holds time-series, trade records, and summary stats for one portfolio group."""

    group_id: str
    equity_curve: pd.Series
    period_returns: pd.Series
    trades: List[TradeRecord]
    stats: Dict[str, float]


@dataclass
class BacktestReport:
    """Aggregates portfolio-level reports and exposes convenience helpers."""

    config: BacktestConfig
    groups: Dict[str, PortfolioReport]
    bench_equity: Optional[pd.Series] = None
    labels: Optional[Dict[str, str]] = None

    def equity_frame(self) -> pd.DataFrame:
        series = {self._group_label(gid): rpt.equity_curve for gid, rpt in self.groups.items()}
        return pd.DataFrame(series).sort_index()

    def return_frame(self) -> pd.DataFrame:
        series = {self._group_label(gid): rpt.period_returns for gid, rpt in self.groups.items()}
        return pd.DataFrame(series).sort_index()

    def daily_return_frame(self) -> pd.DataFrame:
        equity = self.equity_frame()
        return equity.pct_change().dropna(how="all")

    def daily_pnl_frame(self) -> pd.DataFrame:
        equity = self.equity_frame()
        return equity.diff().fillna(0.0)

    def daily_equity_frame(self) -> pd.DataFrame:
        return self.equity_frame()

    def summary_table(self) -> pd.DataFrame:
        table = pd.DataFrame(
            {self._group_label(gid): rpt.stats for gid, rpt in self.groups.items()}
        ).T
        bench_stats = self._benchmark_summary()
        if bench_stats is not None:
            table.loc["benchmark"] = bench_stats
        table.index.name = "portfolio"
        return table

    def render_summary(self) -> str:
        table = self.summary_table()
        return table.to_string(float_format=lambda x: f"{x:0.4f}")

    def save(self, output_dir: Path | None = None, filename: str | None = None) -> Path:
        out_dir = Path(output_dir or self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        figure = self._build_report_figure()
        output_name = filename or self._auto_filename()
        output_path = out_dir / output_name
        figure.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close(figure)
        return output_path

    def _build_report_figure(self) -> Figure:
        equity = self.equity_frame()
        returns_frame = self.return_frame()
        summary = self.summary_table()
        bench = self._benchmark_series(equity.index)

        plt.rcParams.update(
            {
                "font.family": "Helvetica Neue",
                "axes.titleweight": "semibold",
                "axes.labelcolor": "#1C1C1E",
                "axes.facecolor": "#FFFFFF",
                "text.color": "#1C1C1E",
            }
        )

        fig = plt.figure(figsize=(14.5, 10.5), facecolor="#F4F5F7", dpi=220)
        self._render_title_block(fig, equity.index.min(), equity.index.max())
        self._draw_stat_cards(fig, summary)

        grid = fig.add_gridspec(
            3,
            2,
            width_ratios=[1.35, 1.0],
            height_ratios=[2.6, 1.6, 1.5],
            hspace=0.40,
            wspace=0.34,
            left=0.06,
            right=0.94,
            bottom=0.05,
            top=0.74,
        )
        ax_equity = fig.add_subplot(grid[0, 0])
        ax_drawdown = fig.add_subplot(grid[1, 0], sharex=ax_equity)
        right_spec = grid[:2, 1].subgridspec(3, 1, height_ratios=[1.1, 1.0, 1.0], hspace=0.3)
        ax_hist = fig.add_subplot(right_spec[0, 0])
        ax_heatmap = fig.add_subplot(right_spec[1, 0])
        ax_excess = fig.add_subplot(right_spec[2, 0])
        ax_table = fig.add_subplot(grid[2, :])

        palette = self._color_palette(len(equity.columns))
        color_map = dict(zip(equity.columns, palette))
        self._style_axis(ax_equity)
        for column, color in color_map.items():
            values = equity[column]
            ax_equity.plot(equity.index, values, label=column, color=color, linewidth=2.4)
            ax_equity.fill_between(equity.index, values, color=color, alpha=0.12)
        if bench is not None:
            ax_equity.plot(bench.index, bench.values, color="#8E8E93", linestyle="--", linewidth=1.8, label="Benchmark")
        ax_equity.set_title("PriceTrends Quintile Equity", fontsize=11, fontweight="semibold")
        ax_equity.set_ylabel("Equity (KRW)")
        ax_equity.legend(loc="upper left", ncol=3, frameon=False)

        self._plot_drawdown(ax_drawdown, equity, color_map, bench)

        group_labels = [idx for idx in summary.index if idx != "benchmark"]
        if group_labels:
            best_return_label = max(group_labels, key=lambda label: summary.loc[label, "pnl"])
        else:
            best_return_label = equity.columns[-1] if not equity.empty else "n/a"
        best_color = color_map.get(best_return_label, "#4F5DFF")
        best_returns = returns_frame.get(best_return_label)
        best_equity = equity.get(best_return_label)
        self._plot_return_hist(ax_hist, best_returns, best_return_label, best_color)
        self._plot_monthly_heatmap(ax_heatmap, best_equity, best_return_label)
        self._plot_excess_heatmap(ax_excess, best_equity, bench, best_return_label)

        self._render_summary_table(ax_table, summary)

        fig.subplots_adjust(top=0.92)
        return fig

    def _group_label(self, group_id: str) -> str:
        if self.labels and group_id in self.labels:
            return self.labels[group_id]
        return group_id

    def _format_summary_table(self, summary: pd.DataFrame) -> pd.DataFrame:
        formatted = summary.copy()
        for column in formatted.columns:
            formatted[column] = formatted[column].apply(lambda val: self._format_stat(column, val))
        return formatted

    def _format_stat(self, column: str, value: float) -> str:
        currency_fields = {"final_equity", "pnl"}
        percent_fields = {"total_return", "cagr", "volatility", "avg_period_return", "max_drawdown", "win_rate"}
        if column in currency_fields:
            return f"{value:,.0f}"
        if column in percent_fields:
            suffix = ""
            if column == "avg_period_return":
                suffix = f" ({self.config.rebalance_frequency.upper()})"
            return f"{value * 100:0.2f}%{suffix}"
        return f"{value:0.4f}"

    def _style_axis(self, axis: Axes, *, rounded: bool = True) -> None:
        axis.set_facecolor("#FFFFFF")
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        if rounded:
            self._add_panel_background(axis)

    def _color_palette(self, size: int) -> list[str]:
        base = [
            "#4F5DFF",
            "#2FC2FF",
            "#52E0C4",
            "#FEBB52",
            "#FF6B81",
            "#7A7AF7",
        ]
        if size <= len(base):
            return base[:size]
        repeats = int(np.ceil(size / len(base)))
        return (base * repeats)[:size]

    def _plot_drawdown(
        self,
        axis: Axes,
        equity: pd.DataFrame,
        color_map: Dict[str, str],
        bench: Optional[pd.Series],
    ) -> None:
        drawdowns = equity.divide(equity.cummax()).subtract(1.0).fillna(0.0)
        self._style_axis(axis)
        for column, color in color_map.items():
            axis.plot(drawdowns.index, drawdowns[column], label=column, color=color, linewidth=1.8)
        if bench is not None:
            bench_drawdown = bench.divide(bench.cummax()).subtract(1.0)
            axis.plot(bench_drawdown.index, bench_drawdown, color="#8E8E93", linestyle="--", linewidth=1.6, label="Benchmark")
        axis.axhline(0, color="#D1D1D6", linewidth=1)
        axis.set_title("Running Drawdown", fontsize=11)
        axis.set_ylabel("Drawdown (%)", labelpad=10)
        axis.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        axis.set_ylim(drawdowns.min().min() * 1.1, 0.02)
        axis.legend(loc="lower left", ncol=1, frameon=False)

    def _plot_return_hist(self, axis: Axes, series: Optional[pd.Series], label: str, color: str) -> None:
        self._style_axis(axis)
        if series is None or series.dropna().empty:
            axis.text(0.5, 0.5, "No returns", ha="center", va="center")
            return
        stacked = series.dropna() * 100
        bins = int(np.clip(len(stacked) / 10, 15, 40))
        bar_color = color if color.lower() != "#ff6b81" else "#4F5DFF"
        axis.hist(
            stacked,
            bins=bins,
            color=bar_color,
            alpha=0.85,
            edgecolor="#E5E5EA",
        )
        axis.set_title(f"Return Distribution – {label}", fontsize=11)
        axis.set_xlabel("Period Return (%)")
        axis.set_ylabel("Frequency")
        avg = stacked.mean()
        axis.axvline(avg, color="#FF6B81", linestyle="--", linewidth=1.2)
        axis.axvline(0, color="#8E8E93", linestyle=":", linewidth=1)
        # No legend/text for the average line to keep the panel minimal.

    def _plot_monthly_heatmap(self, axis: Axes, equity: Optional[pd.Series], label: str) -> None:
        matrix = self._monthly_return_matrix(equity)
        if matrix is None:
            axis.text(0.5, 0.5, "No data", ha="center", va="center")
            axis.set_xticks([])
            axis.set_yticks([])
            return
        self._render_heatmap(
            axis,
            matrix,
            title=f"Last 12M Monthly Returns – {label}",
            cmap=plt.cm.RdYlGn,
        )

    def _plot_excess_heatmap(
        self,
        axis: Axes,
        quant_equity: Optional[pd.Series],
        bench_equity: Optional[pd.Series],
        label: str,
    ) -> None:
        if quant_equity is None or bench_equity is None:
            axis.text(0.5, 0.5, "No benchmark data", ha="center", va="center")
            axis.set_xticks([])
            axis.set_yticks([])
            return
        quant_matrix = self._monthly_return_matrix(quant_equity)
        bench_matrix = self._monthly_return_matrix(bench_equity)
        if quant_matrix is None or bench_matrix is None:
            axis.text(0.5, 0.5, "No benchmark data", ha="center", va="center")
            axis.set_xticks([])
            axis.set_yticks([])
            return
        bench_matrix = bench_matrix.reindex_like(quant_matrix)
        diff = quant_matrix.subtract(bench_matrix, fill_value=0.0)
        self._render_heatmap(
            axis,
            diff,
            title=f"Excess vs Benchmark – {label}",
            cmap=plt.cm.PuOr,
        )

    def _render_heatmap(self, axis: Axes, matrix: pd.DataFrame, title: str, cmap) -> None:
        self._add_panel_background(axis)
        axis.set_facecolor("#FFFFFF")
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        data = matrix.to_numpy()
        vmax = np.nanmax(np.abs(data))
        vmax = 0.0001 if np.isnan(vmax) or vmax == 0 else vmax
        im = axis.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Month", fontsize=9)
        axis.set_ylabel("Year", fontsize=9)
        axis.tick_params(axis="x", labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
        months = matrix.columns.tolist()
        years = matrix.index.tolist()
        axis.set_xticks(range(len(months)))
        axis.set_xticklabels([calendar.month_abbr[m] for m in months])
        axis.set_yticks(range(len(years)))
        axis.set_yticklabels([str(y) for y in years])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                axis.text(
                    j,
                    i,
                    f"{val * 100:0.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#1C1C1E",
                )

    def _add_panel_background(self, axis: Axes) -> None:
        fig = axis.figure
        bbox = axis.get_position()
        pad_x = 0.01
        pad_y = 0.01
        panel = FancyBboxPatch(
            (bbox.x0 - pad_x, bbox.y0 - pad_y),
            bbox.width + 2 * pad_x,
            bbox.height + 2 * pad_y,
            boxstyle="round,pad=0.02",
            transform=fig.transFigure,
            facecolor="#FFFFFF",
            linewidth=0,
            zorder=0.5,
            alpha=0.95,
        )
        fig.patches.append(panel)
        axis.set_zorder(1)

    def _render_summary_table(self, axis: Axes, summary: pd.DataFrame) -> None:
        axis.axis("off")
        summary_fmt = self._format_summary_table(summary)
        table = axis.table(
            cellText=summary_fmt.to_numpy(),
            rowLabels=summary_fmt.index,
            colLabels=summary_fmt.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1.1, 1.3)
        for (row, col), cell in table.get_celld().items():
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
        axis.set_title("Summary Statistics", pad=6, fontsize=12, fontweight="semibold")

    def _render_title_block(self, fig: Figure, start: pd.Timestamp, end: pd.Timestamp) -> None:
        freq = self.config.rebalance_frequency.upper()
        group_label = f"{len(self.groups)} portfolios"
        if self.config.active_quantiles:
            active = [f"q{idx + 1}" for idx in self.config.active_quantiles]
            group_label = f"{len(active)} buckets ({', '.join(active)})"
        stem = self.config.scores_path[0].stem
        tokens = [tok for tok in stem.split("_") if tok.lower().startswith("i") or tok.lower().startswith("r")]
        stem_hint = "_".join(tokens) if tokens else stem
        title = f"PriceTrends Backtest – {freq} – {group_label} ({stem_hint})"
        fig.text(0.05, 0.97, title, fontsize=17, fontweight="bold")
        fig.text(0.05, 0.93, f"{start:%Y-%m-%d} to {end:%Y-%m-%d}", fontsize=11, color="#6C6C70")
        fig.text(0.05, 0.90, f"Weighting: {self._weighting_label()}", fontsize=11, color="#6C6C70")

    def _draw_stat_cards(self, fig: Figure, summary: pd.DataFrame) -> None:
        best_pnl_label = summary["pnl"].idxmax()
        best_sharpe_label = summary["sharpe"].idxmax()
        best_win_label = summary["win_rate"].idxmax()
        cards = [
            ("Top PnL", best_pnl_label, f"{summary.loc[best_pnl_label, 'pnl']/1e8:0.1f} × ₩100M"),
            ("Top Sharpe", best_sharpe_label, f"{summary.loc[best_sharpe_label, 'sharpe']:0.2f}"),
            ("Top Win Rate", best_win_label, f"{summary.loc[best_win_label, 'win_rate']*100:0.1f}%"),
        ]

        width = 0.27
        height = 0.09
        spacing = 0.02
        for idx, (title, label, value) in enumerate(cards):
            left = 0.05 + idx * (width + spacing)
            bottom = 0.82
            card = FancyBboxPatch(
                (left, bottom),
                width,
                height,
                transform=fig.transFigure,
                boxstyle="round,pad=0.02",
                linewidth=0,
                facecolor="#FFFFFF",
            )
            card.set_path_effects(
                [
                    patheffects.SimpleLineShadow(offset=(0, -1), alpha=0.2),
                    patheffects.Normal(),
                ]
            )
            fig.add_artist(card)
            fig.text(left + 0.02, bottom + 0.055, title, fontsize=10, color="#6C6C70")
            fig.text(left + 0.02, bottom + 0.02, value, fontsize=13, fontweight="bold")
            fig.text(left + width - 0.02, bottom + 0.055, label.upper(), fontsize=9, ha="right", color="#A1A1A6")

    def _benchmark_summary(self) -> Optional[Dict[str, float]]:
        if self.bench_equity is None or self.bench_equity.empty:
            return None
        equity = self.bench_equity.sort_index()
        if equity.empty:
            return None
        returns = equity.pct_change().dropna()
        if returns.empty:
            return None
        return self._stat_block(equity, returns)

    def _benchmark_period_returns(self) -> Optional[pd.Series]:
        if self.bench_equity is None or self.bench_equity.empty:
            return None
        returns = self.bench_equity.sort_index().pct_change().dropna()
        return returns if not returns.empty else None

    def _benchmark_series(self, target_index: pd.Index) -> Optional[pd.Series]:
        if self.bench_equity is None or self.bench_equity.empty:
            return None
        bench = self.bench_equity.reindex(target_index).ffill().dropna()
        return bench if not bench.empty else None

    def _stat_block(self, equity: pd.Series, returns: pd.Series) -> Dict[str, float]:
        equity = equity.dropna()
        if equity.empty:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "volatility": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "final_equity": 0.0,
                "pnl": 0.0,
                "avg_period_return": 0.0,
                "win_rate": 0.0,
            }

        start = float(equity.iloc[0])
        end = float(equity.iloc[-1])
        total_return = 0.0 if start == 0 else (end / start) - 1.0

        periods_per_year = self._periods_per_year(equity.index if isinstance(equity, pd.Series) else None)
        years = len(returns) / periods_per_year if periods_per_year > 0 else 0.0
        cagr = (end / start) ** (1 / years) - 1.0 if years > 0 and start > 0 else 0.0

        vol = returns.std(ddof=0) * (periods_per_year**0.5) if not returns.empty else 0.0
        avg_return = returns.mean() * periods_per_year if not returns.empty else 0.0
        sharpe = 0.0 if vol == 0 else avg_return / vol
        max_dd = self._max_drawdown_series(equity)
        avg_period_return = float(returns.mean()) if not returns.empty else 0.0
        win_rate = float((returns > 0).mean()) if not returns.empty else 0.0
        pnl = end - start

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "volatility": float(vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "final_equity": float(end),
            "pnl": float(pnl),
            "avg_period_return": float(avg_period_return),
            "win_rate": float(win_rate),
        }

    def _periods_per_year(self, index: pd.Index | None = None) -> float:
        default = self._default_periods_per_year()
        if index is None or len(index) < 2 or not isinstance(index, pd.DatetimeIndex):
            return default
        diffs = pd.Series(index).diff().dropna()
        if diffs.empty:
            return default
        avg_days = diffs.dt.total_seconds().mean() / 86_400
        if avg_days <= 0:
            return default
        if avg_days <= 3:
            return 252.0
        return max(default, 365.25 / avg_days)

    def _default_periods_per_year(self) -> float:
        freq = self.config.rebalance_frequency.upper()
        mapping = {
            "D": 252,
            "B": 252,
            "W": 52,
            "M": 12,
            "MS": 12,
            "BM": 12,
            "BMS": 12,
            "Q": 4,
            "QS": 4,
        }
        return float(mapping.get(freq, 12))

    def _max_drawdown_series(self, equity: pd.Series) -> float:
        if equity.empty:
            return 0.0
        running_max = equity.cummax()
        drawdown = equity / running_max - 1.0
        return float(drawdown.min())

    def _monthly_return_matrix(self, equity: Optional[pd.Series], months: int = 12) -> Optional[pd.DataFrame]:
        if equity is None:
            return None
        series = equity.dropna()
        if series.empty:
            return None
        monthly_nav = series.resample("ME").last().dropna()
        monthly_returns = monthly_nav.pct_change().dropna()
        recent = monthly_returns.tail(months)
        if recent.empty:
            return None
        df = pd.DataFrame({"value": recent.values}, index=recent.index)
        df["year"] = df.index.year
        df["month"] = df.index.month
        pivot = df.pivot(index="year", columns="month", values="value").sort_index()
        month_order = sorted(pivot.columns)
        pivot = pivot.reindex(columns=month_order)
        return pivot

    def _render_heatmap(self, axis: Axes, matrix: pd.DataFrame, title: str, cmap) -> None:
        self._add_panel_background(axis)
        axis.set_facecolor("#FFFFFF")
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        data = matrix.to_numpy()
        vmax = np.nanmax(np.abs(data))
        vmax = 0.0001 if np.isnan(vmax) or vmax == 0 else vmax
        im = axis.imshow(data, cmap=cmap, vmin=-vmax, vmax=vmax)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Month", fontsize=9)
        axis.set_ylabel("Year", fontsize=9)
        axis.tick_params(axis="x", labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
        months = matrix.columns.tolist()
        years = matrix.index.tolist()
        axis.set_xticks(range(len(months)))
        axis.set_xticklabels([calendar.month_abbr[m] for m in months])
        axis.set_yticks(range(len(years)))
        axis.set_yticklabels([str(y) for y in years])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                axis.text(
                    j,
                    i,
                    f"{val * 100:0.1f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#1C1C1E",
                )

    def _add_panel_background(self, axis: Axes) -> None:
        fig = axis.figure
        bbox = axis.get_position()
        pad_x = 0.01
        pad_y = 0.01
        panel = FancyBboxPatch(
            (bbox.x0 - pad_x, bbox.y0 - pad_y),
            bbox.width + 2 * pad_x,
            bbox.height + 2 * pad_y,
            boxstyle="round,pad=0.02",
            transform=fig.transFigure,
            facecolor="#FFFFFF",
            linewidth=0,
            zorder=0.5,
            alpha=0.95,
        )
        fig.patches.append(panel)
        axis.set_zorder(1)

    def _auto_filename(self) -> str:
        stem = self.config.scores_path[0].stem
        tokens = [
            token
            for token in stem.split("_")
            if token.lower().startswith(("test", "origin", "i", "r", "fusion"))
        ]
        suffix = "_".join(tokens) if tokens else stem
        freq = self.config.rebalance_frequency.upper()
        universe = "ALL"
        if self.config.constituent_universe is not None:
            universe = self.config.constituent_universe.name
        weight_part = self._weight_code()
        return f"backtest_{freq}_{universe}_{suffix}_{weight_part}.png"

    def _weight_code(self) -> str:
        mode = getattr(self.config, "portfolio_weighting", None)
        if mode is None:
            return "eq"
        if hasattr(mode, "value"):
            return str(mode.value).lower()
        return str(mode).lower()

    def _weighting_label(self) -> str:
        mode = getattr(self.config, "portfolio_weighting", None)
        value = getattr(mode, "value", str(mode)) if mode is not None else "eq"
        mapping = {
            "eq": "Equal Weight",
            "mc": "Market Cap Weight",
        }
        key = str(value).lower()
        label = mapping.get(key, str(mode) if mode is not None else "Equal Weight")
        return f"{label} ({key})"

