# PriceTrends Pipeline Guide

ì°¨íŠ¸ ê¸°ë°˜ ì£¼ê? ?ˆì¸¡ ?¤í—˜??ì²˜ìŒë¶€???ê¹Œì§€ ?œëˆˆ??ë³????ˆë„ë¡??µì‹¬ ?Œì¼ê³??¤í–‰ ?œì„œë¥??•ë¦¬?ˆìŠµ?ˆë‹¤. ë³µì¡???©ì–´??ìµœì†Œ?”í–ˆê³? ?¤ì œ ì½”ë“œ ?ë¦„??ê·¸ë?ë¡??°ë¼ê°€ë©??„ìš”??ëª…ë ¹ë§??¨ê²¼?µë‹ˆ??

---

## 1. ?„ì²´ ?ë¦„ ?”ì•½

```
DATA.xlsx(or parquet) ??loader.py ??DATA/*.parquet
                   ??image.py ??Images/<interval>/images_*.npy + charts_*.feather
                   ??training.py ??models/korea_cnn_.../checkpoint*.pth.tar
evaluate.py ??results/test_results_*.parquet
score.py     ??results/avg_prob.parquet (pivot table)
```

?„ìš”???°ë¼ `read.py`ë¡?ì°¨íŠ¸ë¥??ˆìœ¼ë¡??•ì¸?˜ê³ , `utils` ?˜ìœ„ ?¤í¬ë¦½íŠ¸ë¡??„ì²˜ë¦¬ë? ?”í•  ???ˆìŠµ?ˆë‹¤.

---

## 2. ?¤ì • (params.py & config.json)

- `params.py`  
  `CNNParams`ê°€ `config.json`???½ì–´ ëª¨ë“œ(TEST/PRODUCTION)?€ ?ˆë„??5Â·20Â·60??ë³??˜ì´?¼íŒŒ?¼ë??°ë? ?Œë ¤ì¤ë‹ˆ??  
  ```python
  params = CNNParams()
  config = params.get_config(mode="PRODUCTION", window_size=20)
  ```
- `config.json`  
  - `mode_configs`: ë°°ì¹˜ ?¬ê¸°, ?¬ë‹?ˆì´?? ?œë¡­?„ì›ƒ, ?©ì„±ê³?ì±„ë„ ë°°ì—´, ?™ìƒë¸??¬ê¸°.  
  - `window_configs`: ?…ë ¥ ?ˆë„??`ws`), ?ˆì¸¡ êµ¬ê°„(`pw`), ?ˆì´?´ë³„ ì»¤ë„ ?¬ì´ì¦?  
  - `test_years`: ?‰ê????¬ìš©???°ë„ ëª©ë¡.

---

## 3. ?°ì´??ì¤€ë¹?(loader.py)

1. **?‘ì? ??Parquet (? íƒ)**  
   `DataConverter(excel_path, output_dir).data_convert()`ê°€ `DATA.xlsx`ë¥??½ì–´ `open/high/low/close/volume.parquet`?€ ?œì™¸ ì¢…ëª© JSON??ë§Œë“­?ˆë‹¤.
2. **?Œì? ë¡œë”**  
   ```python
   loader = DataLoader("DATA")
   close_df = loader.load("close")  # ?„ì²´ ?±ì»¤ Ã— ? ì§œ ?„ë ˆ??   loader.available()               # ?¬ìš© ê°€?¥í•œ ???•ì¸
   ```

---

## 4. ì°¨íŠ¸ ?ì„± (prediction/image.py)

1. **êµ¬ì„± ?”ì†Œ**
   - `ChartConfig`: ê°„ê²©, ?´ë?ì§€ ?’ì´, MA/ê±°ë˜???¬í•¨ ?¬ë?, ?€??ê²½ë¡œ.
   - `MarketData`: OHLCV ?°ì´???•í•©??ê²€ì¦?
   - `ChartGenerator`: ?´ë™?‰ê·  ìºì‹± ???•ê·œ????ìº”ë“¤, MA, ê±°ë˜?‰ì„ ?‘ë°± ?´ë?ì§€ë¡??Œë”ë§????¤ìŒ êµ¬ê°„ ?ìŠ¹ ?¬ë?(0/1) ?ˆì´ë¸?
   - `ChartBatchProcessor`: ?°ì»¤ë³??¬ë¼?´ë”© ?ˆë„?? ì§„í–‰ë¥??œì‹œ, ?ˆì™¸ ë¡œê¹…, ê²°ê³¼ ?€??
   - `GenerateImages`: ??êµ¬ì„±?”ì†Œë¥?ë¬¶ì? ?Œì‚¬??
2. **ë©”ëª¨ë¦??ˆì „ ?€??*
   - `numpy.lib.format.open_memmap`?¼ë¡œ ?„ì‹œ `.tmp` ?Œì¼???´ê³  ?´ë?ì§€ê°€ ë§Œë“¤?´ì§ˆ ?Œë§ˆ??ë°”ë¡œ ê¸°ë¡?©ë‹ˆ??
   - ë§ˆì?ë§‰ì— ?¤ì œ ê°œìˆ˜ë§Œí¼ ?˜ë¼ `images_<interval>d.npy`ë¡??€?¥í•˜ê³?ë©”í??°ì´?°ë? `charts_<interval>d_metadata.feather`ë¡?ê¸°ë¡?©ë‹ˆ??
3. **?¤í–‰ ?ˆì‹œ**
   ```python
   if __name__ == "__main__":
       ma_windows_map = {5: (5, 20, 60), 20: (5, 20, 60), 60: (5, 20, 60)}
       run_batch(frequencies=[5, 20, 60], ma_windows_map=ma_windows_map)
   ```
   ì¶œë ¥ ?”ë ‰?°ë¦¬: `prediction/Images/<interval>/`.

---

## 5. ?™ìŠµ (core/training.py)

1. **?°ì´?°ì…‹**  
   `KoreanEquityDataset(intervals, years)`ê°€ ?„ì—??ë§Œë“  `.npy`?€ `.feather`ë¥?ë©”ëª¨ë¦¬ë§µ?¼ë¡œ ?½ì–´ `(image_tensor, label)`???Œë ¤ì¤ë‹ˆ??
2. **ëª¨ë¸**  
   `CNNModel`?€ ì¸??? ì»¤ë„, ?¨ë”© ?±ì„ ?¤ì • ê°€?¥í•œ 2D CNN ë¶„ë¥˜ê¸°ì…?ˆë‹¤. `Trainer`ê°€ ë°°ì¹˜ ?•ê·œ?”Â·ë“œë¡?•„?ƒÂ·MaxPool ì¡°í•©???ë™ êµ¬ì„±?©ë‹ˆ??
3. **?ˆë ¨ ë£¨í‹´**
   ```python
   params = CNNParams()
   config = params.get_config("PRODUCTION", ws=20)
   trainer = Trainer(ws=20, pw=config["pw"], config=config)
   loaders = trainer.get_dataloaders(train_years=config["train_years"])
   trainer.train_empirical_ensem_model(loaders)
   ```
   ì²´í¬?¬ì¸?¸ëŠ” `core/models/korea_cnn_{ws}d{pw}p_{mode}/checkpoint*.pth.tar` ?•íƒœë¡??€?¥ë©?ˆë‹¤.

---

## 6. ?‰ê? (prediction/evaluate.py)

1. **Evaluator êµ¬ì„±**
   - `ModelEvaluator`ê°€ ?ŒìŠ¤???°ì´??`KoreanEquityDataset`)ë¥?ë§Œë“¤ê³??™ìƒë¸?ì²´í¬?¬ì¸?¸ë? ?œíšŒ?©ë‹ˆ??
   - Softmax ?•ë¥ ???‰ê·  ?´ì–´ ìµœì¢… ?ˆì¸¡???»ê³ , ?¼ë²¨/?•ë¥ /?°ì»¤/? ì§œë¥?DataFrame?¼ë¡œ ë°˜í™˜?©ë‹ˆ??
2. **?¤í–‰**
   ```python
   params = CNNParams()
   config = params.get_config("PRODUCTION", 20)
   config["test_years"] = params.get_test_years()
   evaluator = ModelEvaluator(input_days=20, return_days=20, config=config)
   results = evaluator.predict()
   ```
   ê²°ê³¼??`prediction/results/test_results_I20_R20.parquet`?€ ê°™ì´ ?€?¥ë˜ë©? `AccuracyResult`ê°€ ?”ì•½ ë¦¬í¬?¸ë? ì¶œë ¥?©ë‹ˆ??

---

## 7. ?ìˆ˜ ì§‘ê³„ (utils/score.py)

`ResultLoader`ê°€ ê°?ëª¨ë¸??`prob_up` ì»¬ëŸ¼???½ì–´ ê³µí†µ ? ì§œ/ì¢…ëª©?¼ë¡œ ë§ì¶˜ ???‰ê·  ?•ë¥  ?Œì´ë¸”ì„ ë§Œë“­?ˆë‹¤.
```python
from utils.score import ResultLoader
loader = ResultLoader()
avg_prob = loader.avg_prob   # index=ending_date, columns=tickers
```
?í•œ?¤ë©´ ??ê°’ì„ ?„ëµ ?”ì§„?´ë‚˜ ?¤ê±°???œìŠ¤?œìœ¼ë¡??˜ê¸¸ ???ˆìŠµ?ˆë‹¤.

---

## 8. ë¶€ê°€ ? í‹¸

- `prediction/read.py`: ?¹ì • ?°ì»¤Â·?ˆë„?°ì˜ ì°¨íŠ¸ ?´ë?ì§€ë¥?ì§ì ‘ ?•ì¸.
- `prediction/score.py`: ê°„ë‹¨???”ì•½ ?µê³„ ?ëŠ” ?œê°??
- `results_d/`, `models/`: ?¤í—˜ ?°ì¶œë¬¼ì´ ëª¨ì´??ê¸°ë³¸ ?´ë”. ?©ëŸ‰??ì»¤ì?ë©?ì£¼ê¸°?ìœ¼ë¡??•ë¦¬?˜ì„¸??

---

## 11. Transformer µ¥ÀÌÅÍ ÆÄÀÌÇÁ¶óÀÎ

- 	ransformer/data_pipeline.py: OHLCV parquetÀ» ÀĞ¾î (num_windows, lookback, assets, features) ÅÙ¼­¸¦ ¸¸µé°í windows_lb{lookback}_hz{horizon}.pt·Î ÀúÀåÇÕ´Ï´Ù.
- 	ransformer/features.py: log_ret, hl_spread, oc_gap, olume_z °è»ê°ú Á¤±ÔÈ­¸¦ Àü´ãÇÕ´Ï´Ù. RawTensorConfig.features Æ©ÇÃ¿¡ ÀÌ¸§À» Ãß°¡ÇÏ¸é ÀÚµ¿À¸·Î ½ºÅÃµÇ°í À¯È¿ ¸¶½ºÅ©µµ µ¿±âÈ­µË´Ï´Ù.
- Ä¿½ºÅÒ ÇÇÃ³¸¦ ½ÇÇèÇÏ·Á¸é eatures.py¿¡ °è»ê ·ÎÁ÷À» ÀÛ¼ºÇÑ µÚ ÇØ´ç ÀÌ¸§À» ¼³Á¤¿¡ Ãß°¡ÇÏ¸é µË´Ï´Ù.

---
## 11. Transformerë¡??•ì¥?˜ê³  ?¶ë‹¤ë©?

- **ì°¨íŠ¸ ê¸°ë°˜ ViT**: ?„ì—???ì„±???´ë?ì§€ë¥?ê·¸ë?ë¡??¨ì¹˜ ?œí€€?¤ë¡œ ìª¼ê°œ Transformer Encoder???£ìœ¼ë©??©ë‹ˆ?? ?°ì´???Œì´?„ë¼?¸ì? ?¬ì‚¬??ê°€?¥í•©?ˆë‹¤.
- **?ì‹œ ?œê³„??Transformer**: ?´ë?ì§€ ?ì„± ?¨ê³„ë¥??ëµ?˜ê³  OHLCV ?ì„œë¥?ì§ì ‘ ?¤ë£¨???ˆë¡œ???Œì´?„ë¼?¸ì´ ?„ìš”?©ë‹ˆ?? `loader.py` ì¶œë ¥ ???œí€€???„ì²˜ë¦???Transformer ?™ìŠµ ?œì„œë¡??¬êµ¬?±í•˜ë©??©ë‹ˆ??

?„ìš”??ê²½ìš° ??ë¬¸ì„œë¥??…ë°?´íŠ¸?˜ì—¬ ?ˆë¡œ???¤í—˜ ?ˆì°¨ë¥?ê³„ì† ê¸°ë¡?˜ì„¸??



