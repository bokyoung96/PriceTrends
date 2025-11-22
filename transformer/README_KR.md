# Transformer 주가 예측 모델

이 디렉토리는 기술적 지표와 시장 데이터를 활용한 Transformer 기반 딥러닝 주가 예측 모델을 포함합니다.

## 📁 디렉토리 구조

```
transformer/
├── README.md           # 영문 문서
├── README_KR.md        # 이 파일 (한글 문서)
├── model.py           # Transformer 아키텍처
├── features.py        # 기술적 지표 계산
├── pipeline.py        # 데이터 처리 파이프라인 (memmap 적용)
├── params.py          # 설정 로더
├── config.json        # 설정 파일
├── train.py          # 학습 스크립트
├── evaluate.py       # 모델 평가
└── DATA/             # 캐시된 전처리 데이터 (.pt 파일)
```

## 🏗️ 모델 아키텍처

### 개요
Variable Selection Network (VSN)를 활용한 커스텀 Transformer 인코더 모델로, 다양한 기술적 지표를 학습합니다.

### 주요 구성 요소

1. **Variable Selection Network (VSN)**
   - 기술적 지표별 중요도를 자동 학습
   - 변수별 Gated Residual Network (GRN)로 특징 처리
   - 소프트 어텐션으로 변수 간 관계 파악

2. **Transformer Encoder**
   - Multi-head self-attention으로 시계열 의존성 포착
   - GELU 활성화 함수
   - Layer normalization (pre-norm)
   - Residual connections

3. **Output Head**
   - GRN으로 최종 특징 정제
   - Layer normalization
   - 이진 분류 (상승/하락)

## 📊 기술적 지표 (Features)

### 기본 가격 지표
- **`logreturn`**: 로그 수익률
- **`hlspread`**: 고가-저가 스프레드 (종가 대비 정규화)
- **`ocgap`**: 시가-전일종가 갭 (overnight return)
- **`volumez`**: 거래량 z-score

### 추세 지표
- **`ma`**: 이동평균 편차 (기본: 20일)
- **`ema`**: 지수이동평균 편차 (기본: 20일)
- **`magap`**: 주가-이동평균 갭
- **`macd`**: MACD 히스토그램 (주가 대비 정규화)

### 모멘텀 지표
- **`rsi`**: 상대강도지수 (기본: 14일, [-0.5, 0.5]로 정규화)

### 변동성 지표
- **`rollingvol`**: 수익률의 롤링 표준편차 (기본: 20일)
- **`bb`**: 볼린저 밴드 위치 ([-0.5, 0.5]로 정규화)

## ⚙️ 설정 구조

### 새로운 구조 (모드 + 타임프레임 분리)

설정이 두 개의 독립적인 차원으로 분리되었습니다:

```json
{
  "mode_configs": {
    "TEST": { ... },        // 테스트 환경
    "PRODUCTION": { ... }   // 실전 환경
  },
  "timeframe_configs": {
    "SHORT": { ... },       // 단기
    "MEDIUM": { ... },      // 중기
    "LONG": { ... }         // 장기
  }
}
```

### `mode_configs` (환경 설정)

| 모드 | 학습 연도 | 테스트 연도 | 배치 크기 | epoch | lr |
|------|----------|-----------|----------|-------|-----|
| **TEST** | 2000-2011 | 2012-2024 | 64 | 10 | 1e-4 |
| **PRODUCTION** | 2000-2021 | 2022-2024 | 256 | 50 | 1e-5 |

### `timeframe_configs` (예측 기간)

| 타임프레임 | lookback | stride | horizon | 용도 |
|----------|----------|--------|---------|------|
| **SHORT** | 20일 | 1 | 5일 | 단기 모멘텀 |
| **MEDIUM** | 60일 | 1 | 20일 | 월간 리밸런싱 |
| **LONG** | 126일 | 1 | 60일 | 장기 트렌드 |

### 핵심 파라미터 설명

#### **lookback** (관찰 기간)
- 모델이 관찰하는 과거 일수
- 예: `lookback=60` → 과거 60일의 가격/지표 데이터 사용

#### **stride** (윈도우 이동 간격)
- 학습 샘플 생성 시 며칠씩 이동할지
- `stride=1` → 매일 슬라이딩 (최대 데이터, 추천)
- `stride=20` → 20일씩 점프 (적은 데이터, 빠른 학습)

#### **horizon** (예측 목표)
- 며칠 후를 예측할지
- 예: `horizon=20` → 20일 후 가격 변화 예측
- 레이블: `가격[오늘+20일] > 가격[오늘]`이면 1, 아니면 0

#### **min_assets**
- 시점별 최소 유효 종목 수
- 거래 가능한 종목이 너무 적은 날짜 필터링
- 높을수록 엄격한 필터링

#### **norm** (정규화 전략)
- `"asset"`: 종목별 독립 정규화
- `"cross"`: 시점별 종목 간 정규화
- `"none"`: 정규화 없음

## 🚀 사용법

### 학습

```python
from transformer.params import TransformerParams
from transformer.train import Trainer
from transformer.pipeline import Config

params = TransformerParams()

# 모드와 타임프레임 조합
tcfg = params.get_config(mode="TEST", timeframe="MEDIUM")

cfg = Config(
    lookback=tcfg.lookback,
    stride=tcfg.stride,
    horizon=tcfg.horizon,
    features=tuple(tcfg.features),
    min_assets=tcfg.min_assets,
    norm=tcfg.norm,
    train_years=tcfg.train_years,
    test_years=tcfg.test_years
)

trainer = Trainer(cfg, name=f"transformer_{tcfg.mode}")
trainer.train(
    epochs=tcfg.max_epoch,
    batch=tcfg.batch_size,
    lr=tcfg.lr,
    d_model=tcfg.d_model,
    nhead=tcfg.nhead,
    n_layers=tcfg.n_layers,
    d_ff=tcfg.d_ff,
    drop=tcfg.drop
)
```

### 직접 실행

```bash
# train.py에서 mode와 timeframe 설정
# 기본값: mode="TEST", timeframe="MEDIUM"
python transformer/train.py
```

### 설정 조합 예시

```python
# 빠른 검증
params.get_config(mode="TEST", timeframe="SHORT")

# 메인 실험
params.get_config(mode="TEST", timeframe="MEDIUM")

# 장기 패턴
params.get_config(mode="TEST", timeframe="LONG")

# 실전 배포
params.get_config(mode="PRODUCTION", timeframe="MEDIUM")
```

## 📈 데이터 파이프라인

### 흐름

1. **원시 데이터 로드** (`FrameLoader`)
   - `DATA/` 폴더의 parquet 파일에서 OHLCV 데이터 로드
   - 제외 종목 필터링

2. **피처 엔지니어링** (`Featurizer`)
   - 원시 OHLCV로부터 기술적 지표 계산
   - 설정에 따라 정규화 적용

3. **윈도우 생성** (`WindowMaker`) **[메모리 최적화]**
   - **`numpy.memmap` 사용**으로 메모리 효율적 처리
   - `lookback` 길이로 시계열 슬라이딩 윈도우 생성
   - `min_assets` 미만 종목 시점 필터링
   - `horizon` 기반 미래 수익률로 레이블 생성
   - **stride=1에서도 메모리 문제 없음**

4. **캐싱** (`Windows.save/load`)
   - `transformer/DATA/win_lb{lookback}_hz{horizon}.pt`에 저장
   - 대용량 데이터(>4GB)를 위한 pickle protocol 4 사용

5. **데이터셋 & 데이터로더** (`StockDataset`, `get_loaders`)
   - PyTorch Dataset으로 래핑
   - **연도 기반 분할** (train_years vs test_years)
   - 시간순 분할 (랜덤 아님, 미래 정보 유출 방지)

### 메모리 효율성

**CNN의 검증된 방식을 차용**하여 `numpy.memmap` 사용:

```python
# 기존 방식 (메모리 문제):
data = []
for window in windows:
    data.append(window)  # RAM에 누적 → 8GB+
final = np.concatenate(data)  # 크래시!

# 새 방식 (메모리 안전):
mmap = open_memmap(temp_file, shape=(n_windows, lookback, n_features))
for i, window in enumerate(windows):
    mmap[i] = window  # 디스크에 직접 쓰기 → ~2MB RAM만 사용
final = mmap[:actual_count]
```

**결과:**
- ✅ `stride=1` 완벽 작동
- ✅ 최대 학습 데이터 활용
- ✅ 일정한 메모리 사용량 (~2-3MB)

### 데이터 형태

- **입력**: `(batch, lookback, n_features)`
  - `lookback` = 60 (설정 가능)
  - `n_features` = 11 (선택된 지표 개수)
  
- **레이블**: `(batch,)` 
  - 이진 분류: 0 (하락), 1 (상승)

## 📊 학습 세부사항

- **옵티마이저**: Adam
- **손실 함수**: CrossEntropyLoss
- **디바이스**: MPS (Apple Silicon) / CUDA / CPU 자동 감지
- **체크포인트**: 검증 손실 기준 최적 모델 저장
- **데이터 분할**: 연도 기반 시간순 분할 (데이터 유출 없음)

## 🎯 성능 지표

학습 중 로깅:
- **Loss**: Cross-entropy 손실
- **Accuracy**: 분류 정확도

저장 위치:
- **모델**: `models/transformer_{mode}/best.pth`
- **예측 결과**: `scores/price_trends_score_transformer_{mode}_lb{lookback}_hz{horizon}.parquet`

## 🐛 문제 해결

### RuntimeWarning: Mean of empty slice
- **상태**: ✅ 해결됨
- **해결책**: `features.py`에서 경고 억제

### OverflowError: >4GB pickle
- **상태**: ✅ 해결됨
- **해결책**: `pickle_protocol=4` 사용

### MemoryError: Cannot allocate
- **상태**: ✅ 해결됨
- **해결책**: `numpy.memmap` 구현 (CNN 방식)

### RuntimeError: No windows
- **원인**: `min_assets` 너무 엄격하거나 데이터 부족
- **해결책**: `min_assets` 낮추기 (예: 50 → 30)

## 📝 참고사항

- 모델은 고정된 `horizon`으로 미래 수익률 예측 (예: 20일)
- 피처는 자동으로 정규화되어 스케일 문제 방지
- 결측 데이터는 마스킹으로 처리 (NaN → 0, 유효성 마스크 유지)
- **학습/검증 분할은 시간순** (연도 기반), 랜덤 아님
- **stride=1 권장** (최대 데이터 활용)

## 🔮 권장 워크플로우

### 20일 리밸런싱 전략 기준:

1. **학습**
   ```python
   mode="PRODUCTION"
   timeframe="MEDIUM"  # lookback=60, horizon=20, stride=1
   ```

2. **예측**
   ```python
   # 20일마다 예측
   dates = ["2024-01-15", "2024-02-05", "2024-02-26", ...]
   ```

3. **앙상블 (선택사항)**
   ```python
   # 여러 타임프레임 조합
   short = predict(timeframe="SHORT")   # 가중치 0.2
   medium = predict(timeframe="MEDIUM") # 가중치 0.5
   long = predict(timeframe="LONG")     # 가중치 0.3
   final = 0.2*short + 0.5*medium + 0.3*long
   ```

## 🚀 향후 개선 방향

- [ ] 연속 수익률 예측을 위한 회귀 모드 추가
- [ ] 어텐션 시각화 구현
- [ ] Learning rate scheduler 추가
- [ ] 다중 horizon 예측 지원
- [ ] 시계열 교차 검증
- [ ] 하이퍼파라미터 자동 튜닝
