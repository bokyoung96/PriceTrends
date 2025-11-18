# PriceTrends 백테스트 가이드

`backtest/` 모듈은 단일 점수 파일이든 다중 비교 작업이든 동일한 파이프라인으로 실행되도록 재구성되었습니다. 구성(`BacktestConfig`) → 데이터 적재(`BacktestDataLoader`) → 포트폴리오 그룹 계산(`QuantileGroupingStrategy` 등) → 엔진 실행(`BacktestEngine`) → 리포트(`BacktestReport`) 순으로 흐르며, `Backtester`는 전체 과정을 하나의 API로 노출합니다.

---

## 1. 핵심 요약

| 구성 요소 | 설명 |
| --- | --- |
| `BacktestConfig` | I/O 경로, 초기 자본, 리밸런스 주기, 거래 비용, 최소 종목 수를 정의하고, 데이터 로더/그룹 전략/비용 모델을 제공합니다. |
| `BacktestDataLoader` | 점수·가격·종목 구성 데이터를 적재하여 인덱스/컬럼을 정규화하고 `BacktestDataset`으로 정렬합니다. |
| `QuantileGroupingStrategy` | 단일/다중 점수 모두에 대해 분위수를 계산하고 그룹별 티커 묶음을 제공합니다. 임의 그룹 전략으로 교체 가능하며, `ExplicitGroupingStrategy`로 커스텀 로직을 정의할 수 있습니다. |
| `PortfolioTrack` | 한 포트폴리오 그룹에 배정된 자본의 트레이드·에쿼티·수익률 시퀀스를 추적합니다. |
| `BacktestEngine` | 리밸런스 타임라인을 만들고 각 그룹을 실행하여 `BacktestReport`를 생성합니다. |
| `Backtester` | 싱글/멀티 스코어 실행, 결과 저장, 샘플 데이터 프레임 추출 등을 담당하는 퍼사드입니다. |

신호일(`signal_date`)의 점수로 분위수를 결정하고 `entry_lag` 만큼 뒤의 가격으로 진입, 다음 리밸런스 종료일에 청산합니다. 할당 가능한 종목이 부족하면 해당 창은 건너뛰고 사유를 `TradeRecord.note`에 남깁니다.

---

## 2. 빠른 시작

```python
from backtest.runner import Backtester

runner = Backtester()
report = runner.run(
    rebalance_frequency="M",
    active_quantiles=(4,),
    apply_trading_costs=True,
    buy_cost_bps=2.0,
    sell_cost_bps=2.0,
    tax_bps=15.0,
)

print(report.summary_table())
```

복수 점수 파일 비교는 `score_paths`에 **단일 경로 또는 경로 리스트**를 넘기면 됩니다. 모든 실행은 동일한 `run()` 메서드를 사용하며, `group_selector` 옵션으로 비교 대상 포트폴리오 그룹을 지정합니다.

```python
from pathlib import Path
from backtest.runner import Backtester

runner = Backtester()
report = runner.run(
    score_paths=[
        Path("scores/price_trends_score_test_i20_r20.parquet"),
        Path("scores/price_trends_score_origin_i20_r20.parquet"),
    ],
    group_selector=("q1", "q5"),
)
```

실행 후에는 데이터·성과를 자유롭게 확인할 수 있습니다.

```python
scores = runner.score_df         # 단일 실행은 DataFrame, 멀티는 dict[label, DataFrame]
prices = runner.price_df
hit_rate = runner.hit_rate_df    # 분위수별 승률
returns = runner.period_return_df
pnl = runner.daily_pnl_df
report.save()                    # 리포트 이미지 저장
```

VS Code Interactive Window에서 import 문제가 없도록 모든 모듈은 루트(`ROOT`) 경로를 확보합니다.

---

## 3. 주요 파라미터

- **score/price 경로**: `BacktestConfig.scores_path`, `close_path`. 단일 Path/문자열이든 리스트/튜플이든 그대로 넘기면 되며, 상대 경로는 자동으로 프로젝트 루트를 기준으로 확장됩니다.
- **분위수/최소 종목 수**: `quantiles`, `min_assets`, `allow_partial_buckets`. 특정 분위수만 분석하려면 `active_quantiles`와 `group_selector`를 조합하거나, 커스텀 `portfolio_grouping`을 전달하세요.
- **벤치마크 심볼**: `benchmark_symbol`에 벤치마크 열 이름을 지정하면, 해당 열을 `BacktestDataset.bench`로 사용합니다(예: `"IKS200"`). `None`이면 벤치마크를 비활성화합니다.
- **리밸런스 주기**: pandas 오프셋 문자열 (`"M"`, `"MS"`, `"Q"`, …)을 그대로 사용합니다.
- **거래 비용**: `apply_trading_costs=True`일 때 `buy_cost_bps`, `sell_cost_bps`, `tax_bps`가 적용되며, 진입/청산 시 각각 차감됩니다.
- **포트폴리오 가중치**: `portfolio_weighting`을 `"eq"`(동일 가중) 또는 `"mc"`(시총 가중)로 설정합니다. 시총 가중을 사용하면 `weight_data_path`(기본 `DATA/METRIC_MKTCAP.parquet`)에 있는 시가총액 데이터를 참조해 리밸런스마다 비중을 계산합니다.
- **유니버스 필터**: `constituent_universe`로 사전 정의된 지수 구성(예: `MarketUniverse.KOSPI200`)을 적용하거나, `constituent_path`로 커스텀 마스크를 줄 수 있습니다.
- **출력 경로**: `output_dir` 아래에 `backtest_{freq}_{universe}_{suffix}.png` 형식으로 저장됩니다.

---

## 4. 직접 구성 요소 사용

```python
from backtest.config import BacktestConfig
from backtest.engine import BacktestEngine

config = BacktestConfig(initial_capital=50_000_000, rebalance_frequency="MS")
dataset = config.data_loader().build()
engine = BacktestEngine(config, dataset)
report = engine.run()
print(report.summary_table())
```

모듈들을 직접 조합하면 커스텀 데이터 전처리나 실험적인 할당 로직을 손쉽게 적용할 수 있습니다.

---

## 5. 거래 비용/에쿼티/레저 계산

- `PortfolioTrack`은 리밸런스 시점에 동일 비중으로 자본을 나누고, 각 티커별 `price × quantity`를 즉시 기록합니다. 이 정보는 `TradeRecord.positions`에 저장되어 모든 백테스트 유형에서 동일하게 조회할 수 있습니다.
- 진입 시 `ExecutionCostModel.net_entry_capital`로 매수 비용을 차감하고, 청산 시 매도+세금 비용을 적용합니다. 청산 비용은 해당 윈도우의 마지막 일자에 반영됩니다.
- 일별 에쿼티는 저장된 수량을 일별 가격에 곱해 합산함으로써 계산되며, 리밸런스 주기와 무관하게 항상 일 단위로 성과가 재구성됩니다.
- 모든 요약 통계(CAGR, volatility, Sharpe, win rate 등)는 이 일별 에쿼티/수익률을 기반으로 연율화되며, `avg_period_return`은 설정한 리밸런스 주기(M, Q 등)에 맞춰 해당 기간 수익률 평균을 보여줍니다.
- 중간에 거래 중단(정지 종목 등)이 발생하면 메모(`TradeRecord.note`)와 함께 해당 종목을 제외하여 추적합니다.

---

## 6. 테스트/확장 힌트

- `Backtester.run()`은 동일한 코드 경로로 단일/배치 실행을 수행하므로, 테스트 코드도 동일한 경로만 검증하면 됩니다.
- `BacktestSuite.combine_jobs()`는 서로 다른 점수 파일 간 비교를 위한 재사용 가능한 헬퍼입니다.
- 모든 클래스를 `dataclass`/컴포지션 기반으로 쪼개 SOLID 원칙을 지켜 두었으므로, 개별 모듈 교체나 확장이 용이합니다.
