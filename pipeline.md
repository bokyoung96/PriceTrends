# PriceTrends Pipeline Guide

차트 기반 주가 예측 실험을 처음부터 끝까지 한눈에 볼 수 있도록 핵심 파일과 실행 순서를 정리했습니다. 복잡한 용어는 최소화했고, 실제 코드 흐름을 그대로 따라가며 필요한 명령만 남겼습니다.

---

## 1. 전체 흐름 요약

```
DATA.xlsx(or parquet) → loader.py → DATA/*.parquet
                   ↓
image.py → Images/<interval>/images_*.npy + charts_*.feather
                   ↓
training.py → models/korea_cnn_.../checkpoint*.pth.tar
evaluate.py → results/test_results_*.parquet
score.py     → results/avg_prob.parquet (pivot table)
```

필요에 따라 `read.py`로 차트를 눈으로 확인하고, `utils` 하위 스크립트로 후처리를 더할 수 있습니다.

---

## 2. 설정 (params.py & config.json)

- `params.py`  
  `CNNParams`가 `config.json`을 읽어 모드(TEST/PRODUCTION)와 윈도우(5·20·60일)별 하이퍼파라미터를 돌려줍니다.  
  ```python
  params = CNNParams()
  config = params.get_config(mode="PRODUCTION", window_size=20)
  ```
- `config.json`  
  - `mode_configs`: 배치 크기, 러닝레이트, 드롭아웃, 합성곱 채널 배열, 앙상블 크기.  
  - `window_configs`: 입력 윈도우(`ws`), 예측 구간(`pw`), 레이어별 커널 사이즈.  
  - `test_years`: 평가에 사용할 연도 목록.

---

## 3. 데이터 준비 (loader.py)

1. **엑셀 → Parquet (선택)**  
   `DataConverter(excel_path, output_dir).data_convert()`가 `DATA.xlsx`를 읽어 `open/high/low/close/volume.parquet`와 제외 종목 JSON을 만듭니다.
2. **파케 로더**  
   ```python
   loader = DataLoader("DATA")
   close_df = loader.load("close")  # 전체 틱커 × 날짜 프레임
   loader.available()               # 사용 가능한 키 확인
   ```

---

## 4. 차트 생성 (prediction/image.py)

1. **구성 요소**
   - `ChartConfig`: 간격, 이미지 높이, MA/거래량 포함 여부, 저장 경로.
   - `MarketData`: OHLCV 데이터 정합성 검증.
   - `ChartGenerator`: 이동평균 캐싱 → 정규화 → 캔들, MA, 거래량을 흑백 이미지로 렌더링 → 다음 구간 상승 여부(0/1) 레이블.
   - `ChartBatchProcessor`: 티커별 슬라이딩 윈도우, 진행률 표시, 예외 로깅, 결과 저장.
   - `GenerateImages`: 위 구성요소를 묶은 파사드.
2. **메모리 안전 저장**
   - `numpy.lib.format.open_memmap`으로 임시 `.tmp` 파일을 열고 이미지가 만들어질 때마다 바로 기록합니다.
   - 마지막에 실제 개수만큼 잘라 `images_<interval>d.npy`로 저장하고 메타데이터를 `charts_<interval>d_metadata.feather`로 기록합니다.
3. **실행 예시**
   ```python
   if __name__ == "__main__":
       ma_windows_map = {5: (5, 20, 60), 20: (5, 20, 60), 60: (5, 20, 60)}
       run_batch(frequencies=[5, 20, 60], ma_windows_map=ma_windows_map)
   ```
   출력 디렉터리: `prediction/Images/<interval>/`.

---

## 5. 학습 (core/training.py)

1. **데이터셋**  
   `KoreanEquityDataset(intervals, years)`가 위에서 만든 `.npy`와 `.feather`를 메모리맵으로 읽어 `(image_tensor, label)`을 돌려줍니다.
2. **모델**  
   `CNNModel`은 층 수, 커널, 패딩 등을 설정 가능한 2D CNN 분류기입니다. `Trainer`가 배치 정규화·드롭아웃·MaxPool 조합을 자동 구성합니다.
3. **훈련 루틴**
   ```python
   params = CNNParams()
   config = params.get_config("PRODUCTION", ws=20)
   trainer = Trainer(ws=20, pw=config["pw"], config=config)
   loaders = trainer.get_dataloaders(train_years=config["train_years"])
   trainer.train_empirical_ensem_model(loaders)
   ```
   체크포인트는 `core/models/korea_cnn_{ws}d{pw}p_{mode}/checkpoint*.pth.tar` 형태로 저장됩니다.

---

## 6. 평가 (prediction/evaluate.py)

1. **Evaluator 구성**
   - `ModelEvaluator`가 테스트 데이터(`KoreanEquityDataset`)를 만들고 앙상블 체크포인트를 순회합니다.
   - Softmax 확률을 평균 내어 최종 예측을 얻고, 라벨/확률/티커/날짜를 DataFrame으로 반환합니다.
2. **실행**
   ```python
   params = CNNParams()
   config = params.get_config("PRODUCTION", 20)
   config["test_years"] = params.get_test_years()
   evaluator = ModelEvaluator(input_days=20, return_days=20, config=config)
   results = evaluator.predict()
   ```
   결과는 `prediction/results/test_results_I20_R20.parquet`와 같이 저장되며, `AccuracyResult`가 요약 리포트를 출력합니다.

---

## 7. 점수 집계 (utils/score.py)

`ResultLoader`가 각 모델의 `prob_up` 컬럼을 읽어 공통 날짜/종목으로 맞춘 뒤 평균 확률 테이블을 만듭니다.
```python
from utils.score import ResultLoader
loader = ResultLoader()
avg_prob = loader.avg_prob   # index=ending_date, columns=tickers
```
원한다면 이 값을 전략 엔진이나 실거래 시스템으로 넘길 수 있습니다.

---

## 8. 부가 유틸

- `prediction/read.py`: 특정 티커·윈도우의 차트 이미지를 직접 확인.
- `prediction/score.py`: 간단한 요약 통계 또는 시각화.
- `results_d/`, `models/`: 실험 산출물이 모이는 기본 폴더. 용량이 커지면 주기적으로 정리하세요.

---

## 9. Transformer로 확장하고 싶다면?

- **차트 기반 ViT**: 위에서 생성한 이미지를 그대로 패치 시퀀스로 쪼개 Transformer Encoder에 넣으면 됩니다. 데이터 파이프라인은 재사용 가능합니다.
- **원시 시계열 Transformer**: 이미지 생성 단계를 생략하고 OHLCV 텐서를 직접 다루는 새로운 파이프라인이 필요합니다. `loader.py` 출력 → 시퀀스 전처리 → Transformer 학습 순서로 재구성하면 됩니다.

필요한 경우 이 문서를 업데이트하여 새로운 실험 절차를 계속 기록하세요.
