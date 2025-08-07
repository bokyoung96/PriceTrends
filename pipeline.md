# W_price_trends 전체 데이터 파이프라인 구조

## 🔄 전체 워크플로우

```
params.py & config.json → loader.py → image.py → training.py → evaluate.py → score.py
         ↓                   ↓         ↓           ↓            ↓           ↓
      설정 관리            parquet   이미지       CNN 모델      예측결과    pivot 테이블
```

---

## ⚙️ params.py & config.json - 설정 및 파라미터 관리

### `CNNParams` 클래스 - 설정 관리

```python
class CNNParams:
    def __init__(self, config_path: str = None)
    def get_config(self, mode: str, window_size: int) -> Dict[str, Any]  # 🎯 핵심
    def get_test_years() -> List[int]
    @property modes -> List[str]
    @property window_sizes -> List[int]
```

### `config.json` - 모델 설정 파일

**모드별 설정:**
```json
{
  "mode_configs": {
    "TEST": {
      "ensem_size": 1,        # 앙상블 크기
      "batch_size": 64,       # 배치 크기
      "max_epoch": 10,        # 최대 에포크
      "lr": 1e-4,             # 학습률
      "drop_prob": 0.3,       # 드롭아웃 확률
      "conv_channels": [32, 64, 128]  # 컨볼루션 채널
    },
    "PRODUCTION": {
      "ensem_size": 5,        # 더 큰 앙상블
      "batch_size": 256,      # 더 큰 배치
      "max_epoch": 50,        # 더 많은 에포크
      "lr": 1e-5,             # 더 작은 학습률
      "drop_prob": 0.5,       # 더 높은 드롭아웃
      "conv_channels": [32, 64, 128, 256]  # 더 깊은 네트워크
    }
  }
}
```

**윈도우별 설정:**
```json
{
  "window_configs": {
    "5": {
      "pw": 5,
      "filter_sizes": {
        "TEST": [[3, 2], [3, 2], [3, 2]],
        "PRODUCTION": [[3, 2], [3, 2], [3, 2], [3, 2]]
      }
    },
    "20": {
      "pw": 20,
      "filter_sizes": {
        "TEST": [[5, 3], [3, 3], [3, 3]],
        "PRODUCTION": [[5, 3], [3, 3], [3, 3], [3, 3]]
      }
    },
    "60": {
      "pw": 60,
      "filter_sizes": {
        "TEST": [[5, 3], [5, 3], [3, 3]],
        "PRODUCTION": [[5, 3], [5, 3], [3, 3], [3, 3]]
      }
    }
  }
}
```

**테스트 년도:**
```json
{
  "test_years": [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
}
```

---

## 📁 loader.py - 원본 데이터 변환 및 로드

### `DataConverter` 클래스 - 엑셀 → Parquet 변환

```python
class DataConverter:
    def __init__(self, excel_path: str, output_dir: str)
    def data_convert() -> None  # 🎯 핵심 메서드
```

**핵심 기능:**
- `DATA.xlsx` 엑셀 파일을 읽어서 OHLCV 데이터 추출
- 각 데이터 타입별로 별도 parquet 파일 생성
- `#INVALID OPTION` 포함된 종목들을 제외하고 처리

**저장 파일:**
```
DATA/
├── open.parquet     # 시가 데이터
├── high.parquet     # 고가 데이터  
├── low.parquet      # 저가 데이터
├── close.parquet    # 종가 데이터
├── volume.parquet   # 거래량 데이터
└── excluded_tickers.json  # 제외된 종목 목록
```

### `DataLoader` 클래스 - Parquet 데이터 로드

```python
class DataLoader:
    def __init__(self, data_dir: str)
    def load(self, name: str) -> pd.DataFrame
    def available() -> List[str]
    def get_excluded_tickers() -> List[str]
```

**사용법:**
```python
loader = DataLoader("DATA")
close_df = loader.load("close")  # 또는 loader.close
```

---

## 🖼️ image.py - 가격 데이터 → 차트 이미지 변환

### `ChartGenerator` 클래스 - 이미지 생성 핵심

```python
class ChartGenerator:
    def __init__(self, market_data: MarketData, config: ChartConfig)
    def generate_chart_image() -> Tuple[Image.Image, int]  # 🎯 핵심
```

**핵심 기능:**
- OHLCV 데이터를 32x15 크기 차트 이미지로 변환
- 이동평균선(MA) 및 거래량 차트 포함
- 상승/하락 라벨 자동 생성

**이미지 구조:**
```
32x15 픽셀 차트 (5일 기준)
├── 상단 25픽셀: 가격 차트 (OHLC + MA)
├── 1픽셀 갭
└── 하단 6픽셀: 거래량 차트

64x60 픽셀 차트 (20일 기준)
96x180 픽셀 차트 (60일 기준)
```

### `ChartBatchProcessor` 클래스 - 배치 처리

```python
class ChartBatchProcessor:
    def generate_batch_dataset() -> None
    def load_batch_dataset() -> Tuple[np.ndarray, pd.DataFrame]
```

**저장 결과:**
```
Images/
├── 5/
│   ├── images_5d.npy           # 5일 차트 이미지 배열
│   └── charts_5d_metadata.feather  # 메타데이터
├── 20/
│   ├── images_20d.npy          # 20일 차트 이미지 배열
│   └── charts_20d_metadata.feather
└── 60/
    ├── images_60d.npy          # 60일 차트 이미지 배열
    └── charts_60d_metadata.feather
```

---

## 🤖 training.py - CNN 모델 훈련

### `KoreanEquityDataset` 클래스 - 데이터셋

```python
class KoreanEquityDataset(Dataset):
    def __init__(self, intervals: int, years: List[int])
    def __getitem__(self, idx) -> Dict  # 이미지 + 라벨 반환
```

**반환 데이터:**
```python
{
    'image': torch.Tensor,     # 32x15 차트 이미지
    'label': torch.Tensor,     # 0(하락) or 1(상승)
    'StockID': str,           # 종목 코드
    'ending_date': str,       # 기준일
}
```

### `CNNModel` 클래스 - 모델 구조

```python
class CNNModel(nn.Module):
    def __init__(self, layer_number, input_size, conv_layer_chanls, ...)
    def forward(self, x) -> torch.Tensor  # 2개 클래스 분류 출력
```

**모델 구조:**
- 입력: 1x32x15 차트 이미지 (5일 기준)
- Conv2D 레이어들 + BatchNorm + LeakyReLU
- MaxPool2D + Dropout
- 출력: 2개 클래스 (하락/상승)

### `Trainer` 클래스 - 훈련 관리

```python
class Trainer:
    def train_empirical_ensem_model()  # 앙상블 모델 훈련
    def train_single_model()           # 단일 모델 훈련
```

**저장 결과:**
```
models/
├── korea_cnn_5d5p_TEST/
│   ├── checkpoint0.pth.tar
│   ├── checkpoint1.pth.tar
│   └── ...
├── korea_cnn_20d20p_TEST/
└── korea_cnn_60d60p_TEST/
```

---

## 📊 evaluate.py - 모델 평가 및 예측

### `ModelEvaluator` 클래스 - 모델 평가

```python
class ModelEvaluator:
    def __init__(self, input_days: int, return_days: int, config: Dict)
    def predict() -> pd.DataFrame  # 🎯 핵심 메서드
```

**핵심 - 상승 확률 계산:**
```python
# 1. 앙상블 모델 출력 평균
ensemble_outputs = torch.stack(all_outputs).mean(dim=0)

# 2. 소프트맥스로 확률 변환
probabilities = torch.softmax(ensemble_outputs, dim=1)

# 3. 상승 확률 추출
prob_up = probabilities[:, 1]  # 🎯 이게 핵심!
```

### `AccuracyResult` 클래스 - 결과 저장

```python
class AccuracyResult:
    def save_res(self, model_name: str, df: pd.DataFrame)
    def print_summary() -> None
```

**저장되는 DataFrame 구조:**

| 컬럼 | 설명 |
|------|------|
| `StockID` | 종목 ID |
| `ending_date` | 예측 기준일 |
| `label` | 실제 라벨 (0: 하락, 1: 상승) |
| `prediction` | 예측 결과 (0: 하락, 1: 상승) |
| `prob_down` | 하락 확률 (0~1) |
| `prob_up` | **상승 확률 (0~1)** ⭐ |

**저장 파일:**
```
results/
├── test_results_i5_r5.parquet
├── test_results_i20_r20.parquet
└── test_results_i60_r60.parquet
```

---

## 🎯 score.py - 결과 로드 및 Pivot 변환

### `ResultLoader` 클래스 - 결과 로드 및 변환

```python
class ResultLoader:
    def raw_results(i, r) -> pd.DataFrame      # 원본 parquet 로드
    def load_results(i, r) -> Dict[DataFrame]  # 🎯 pivot 변환
    def available_models() -> List[Tuple]      # 사용 가능한 모델 목록
    def avg_prob -> pd.DataFrame              # 평균 상승 확률 계산
```

### 🎯 핵심 - `load_results()` 메서드의 Pivot 변환

```python
def load_results(self, i: int, r: int) -> Dict[str, pd.DataFrame]:
    # 1. 원본 데이터 로드
    df = self.raw_results(i, r)
    
    # 2. 날짜 형식 변환
    df['ending_date'] = pd.to_datetime(df['ending_date'])
    
    # 3. 각 컬럼별로 pivot 변환
    pivot_dict = {}
    for col in ['label', 'prediction', 'prob_down', 'prob_up']:
        pivot_df = df.pivot(
            index='ending_date',    # 행: 날짜
            columns='StockID',      # 열: 종목ID
            values=col             # 값: 해당 컬럼
        )
        pivot_dict[col] = pivot_df
    
    return pivot_dict
```

### 📋 데이터 변환 과정

#### Before (evaluate.py 출력)
```
StockID | ending_date | label | prediction | prob_up
A001    | 20240101    | 1     | 1          | 0.73
A002    | 20240101    | 0     | 1          | 0.68
A001    | 20240102    | 0     | 0          | 0.42
...
```

#### After (score.py pivot 변환)
```
prob_up DataFrame:
ending_date | A001 | A002 | A003 | ...
20240101    | 0.73 | 0.68 | 0.55 | ...
20240102    | 0.42 | 0.71 | 0.63 | ...
20240103    | 0.81 | 0.49 | 0.77 | ...
...
```

### 🎯 `avg_prob` - 앙상블 평균 계산

```python
@cached_property
def avg_prob(self) -> pd.DataFrame:
    # 1. 모든 모델의 prob_up 로드
    prob_ups = []
    for i, r in models:
        data = self.load_results(i, r)
        prob_ups.append(data['prob_up'])
    
    # 2. 공통 날짜/종목만 추출
    common_dates = prob_ups[0].index
    common_ids = prob_ups[0].columns
    for prob_up in prob_ups[1:]:
        common_dates = common_dates.intersection(prob_up.index)
        common_ids = common_ids.intersection(prob_up.columns)
    
    # 3. 평균 계산
    avgs = sum(aligned_probs) / len(aligned_probs)
    return avgs
```

---

## 🔍 read.py - 차트 이미지 뷰어 (기타)

### `ChartViewer` 클래스 - 이미지 시각화

```python
class ChartViewer:
    def __init__(self, intervals: int, base_dir: str = 'Images')
    def load_data() -> None
    def display_charts(self, ticker: str, chart_numbers: List[int]) -> None  # 🎯 핵심
```

**핵심 기능:**
- 생성된 차트 이미지들을 시각화
- 특정 종목의 차트 이미지 확인 가능
- 메타데이터와 함께 표시

**사용법:**
```python
from read import read_charts

# 삼성전자(A005930)의 5일 차트 중 270번, 6090번 차트 보기
read_charts(ticker="A005930", 
            intervals=5, 
            chart_numbers=[270, 6090])
```

**표시 정보:**
- 차트 기간 (start_date ~ end_date)
- 추정 기간 (estimation_start ~ estimation_end)
- 라벨 (Up/Down)
- 실제 차트 이미지 (matplotlib으로 시각화)

---

## 💡 핵심 포인트

### 1. 설정 관리
- **params.py**: 모든 설정을 통합 관리
- **config.json**: 모드별(TEST/PRODUCTION), 윈도우별 하이퍼파라미터 분리

### 2. 상승 확률 계산 지점
- **evaluate.py**: `probabilities[:, 1]` → 소프트맥스 출력에서 상승 확률 추출
- **score.py**: `prob_up` 컬럼을 pivot으로 변환

### 3. 데이터 형태 변환
- **Long format** (evaluate.py) → **Wide format** (score.py)
- 행=날짜, 열=종목ID, 값=상승확률

### 4. 앙상블 처리
- **evaluate.py**: 개별 모델들의 출력 평균 (같은 I/R 조합 내)
- **score.py**: 여러 (I/R) 조합 모델들의 평균 (I5/R5, I20/R20, I60/R60)

### 5. 최종 결과물
- `loader.avg_prob`: 모든 모델의 상승 확률 평균값이 담긴 DataFrame
- 형태: 날짜(행) × 종목ID(열) × 상승확률(값)
- 값 범위: 0~1 (0.5 이상이면 상승 예측)

---

## 🚀 실행 순서

1. **설정 확인**: `params.py` & `config.json` 설정 점검
2. **데이터 준비**: `python loader.py` (엑셀 → parquet)
3. **이미지 생성**: `python image.py` (가격 → 차트 이미지)
4. **모델 훈련**: `python training.py` (CNN 모델 훈련)
5. **모델 평가**: `python evaluate.py` (예측 및 결과 저장)
6. **결과 활용**: `python score.py` (pivot 변환 및 앙상블 평균)
7. **시각화**: `python read.py` (차트 확인 - 선택사항)

```python
# 최종 사용법
from score import ResultLoader
loader = ResultLoader()
final_scores = loader.avg_prob  # 모든 모델의 평균 상승 확률
```

---

## 📝 파일 구조 요약

```
W_price_trends/
├── params.py           # 설정 관리
├── config.json         # 하이퍼파라미터 설정
├── loader.py           # 데이터 로드
├── image.py            # 이미지 생성
├── training.py         # 모델 훈련
├── evaluate.py         # 모델 평가
├── score.py            # 결과 처리
├── read.py             # 이미지 뷰어 (번외)
├── DATA/               # 원본 데이터
├── Images/             # 차트 이미지
├── models/             # 훈련된 모델
└── results/            # 예측 결과
``` 