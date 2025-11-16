# PriceTrends 백테스트 가이드

PriceTrends가 산출한 점수(신호) 테이블을 읽어, 원하는 방식으로 버킷(퀀타일) 포트폴리오를 구성하고 리밸런스하는 모듈이다. `scores/*.parquet`와 `DATA/close.parquet`만 준비되어 있다면 데이터 로딩, 버킷 배분, 거래 비용 반영, 리포트 생성까지 한 번에 수행한다.

---

## 1. 핵심 설계

| 구성요소 | 설명 |
| --- | --- |
| `BacktestConfig` | I/O 경로, 초기 자본, 버킷 수, 리밸런스 주기, 거래 비용, 최소 종목 수 등을 보관하는 불변 설정 |
| `BacktestDatasetBuilder` | 점수/종가 파켓을 읽고, 영업일 인덱스/티커를 맞춘 뒤 `BacktestDataset`으로 반환 |
| `BucketAllocator` | 점수 단면을 정렬해 버킷별 티커 리스트를 생성 |
| `BacktestEngine` | 리밸런스 스케줄을 만들고, 각 버킷 포트폴리오(`BucketPortfolio`)를 업데이트 |
| `Backtester` | 위 과정을 묶어 API 하나(`run`)로 제공하는 퍼사드 |
| `SimulationReport` | 각 버킷의 자산곡선, 수익률, 요약 통계를 보관하며 PNG 리포트도 생성 |

엔진은 매 리밸런스 시점마다 `entry_lag`일 후의 가격으로 진입하고, 다음 리밸런스 지점에서 청산한다. 거래 비용이 켜져 있다면 진입 시 매수 비용을 먼저 차감하고, 청산 시 매도+세금 비용을 빼서 다음 라운드 자본을 결정한다. 데이터가 부족하거나 가격이 비정상적으로 튀면 해당 종목을 자동 제외하고 이유를 `TradeRecord.note`에 남긴다.

---

## 2. 기본 동작 요약

- **점수 경로**: `scores/price_trends_score_test_i20_r20.parquet`
- **가격 경로**: `DATA/close.parquet`
- **버킷 수**: 5개(0~4). `active_quantiles`를 지정하면 일부만 운용 가능.
- **초기 자본**: 버킷당 100,000,000 KRW
- **리밸런스 주기**: 월말(`"M"`). 주말이더라도 실제 존재하는 마지막 영업일만 사용.
- **최소 종목 수**: 30개. 부족 시 버킷을 건너뛰고 사유를 기록.
- **거래 비용**: 기본 비활성. `apply_trading_costs=True`와 `buy_cost_bps`, `sell_cost_bps`, `tax_bps`로 설정.
- **리포트 저장**: `bt/backtest_{주기}_{iXX_rXX}.png`. 점수 파일 이름에서 `i20_r20` 같은 토큰을 추출해 제목과 파일명에 포함.

모든 파라미터는 `BacktestConfig` 생성 시 혹은 `Backtester().run(...)`에 키워드 인자로 전달해 덮어쓸 수 있다.

---

## 3. 가장 빠른 사용 예시

```python
from backtest.runner import Backtester

runner = Backtester()
report = runner.run(
    rebalance_frequency="M",
    active_quantiles=(4,),       # 최상위 버킷만 운용
    apply_trading_costs=True,
    buy_cost_bps=2.0,
    sell_cost_bps=2.0,
    tax_bps=15.0,
)

print(report.summary_table())
report.save()  # bt/backtest_M_i20_r20.png
```

`Backtester`는 내부에서 `BacktestDatasetBuilder → BucketAllocator → BacktestEngine`을 자동으로 호출한다. 실행이 끝나면 `SimulationReport`가 반환되며, `summary_table()`, `equity_frame()`, `return_frame()` 같은 메서드를 그대로 사용할 수 있다.

### VS Code Interactive Window에서 활용하기

```python
from backtest.runner import Backtester

bt = Backtester()
report = bt.run()

scores = bt.score_df         # 점수 DataFrame (dates × tickers)
prices = bt.price_df         # 가격 DataFrame
hit_rate = bt.hit_rate_df    # 버킷별 승률
daily = bt.daily_return_df   # 리밸런스 사이 구간을 일별로 보간한 수익률
```

---

## 4. 구성 요소별 세부 제어

```python
from backtest.config import BacktestConfig
from backtest.data_sources import BacktestDatasetBuilder
from backtest.engine import BacktestEngine
from backtest.quantiles import BucketAllocator

config = BacktestConfig(
    initial_capital=50_000_000,
    rebalance_frequency="MS",
    entry_lag=0,  # 룩어헤드가 필요하면 0으로 지정
)

dataset = BacktestDatasetBuilder(config.scores_path, config.close_path).build()
assigner = BucketAllocator(config.quantiles, config.min_assets, config.allow_partial_buckets)

report = BacktestEngine(config, dataset, assigner).run()
print(report.summary_table())
report.save("bt/backtest_MS_custom.png")
```

이처럼 각 단계를 직접 호출하면 데이터 전처리나 버킷 로직을 원하는 대로 수정할 수 있다.

---

## 5. 거래 비용 및 자본 흐름

- `BucketPortfolio`는 직전 라운드 자본을 100% 투자 대상으로 본다.
- 진입 시 `ExecutionCostModel.net_entry_capital(...)`로 매수 비용(bps)을 차감한 금액만 실제로 배분한다.
- 청산 시 평균 수익률(청산/진입 가격비)을 적용한 뒤, 매도+세금 비용을 다시 차감한다.
- 최종 `capital_out`이 다음 라운드의 시작 자본이 되며, `equity_series`와 `period_returns`에 저장된다.

즉, 거래 비용을 크게 설정할수록 실질적으로 다시 투자할 수 있는 자본이 빠르게 줄어든다.

---

## 6. 테스트/유지보수

현재는 실데이터 기반 검증을 위해 테스트 스위트가 비워져 있다(`backtest/tests/`). 필요 시 `pytest backtest/tests -q`로 여유롭게 추가하면 된다. 모듈 전반은 패키지 루트를 직접 `sys.path`에 넣어 두었기 때문에, 노트북이나 Interactive Window에서도 별도 조작 없이 import가 가능하다.
