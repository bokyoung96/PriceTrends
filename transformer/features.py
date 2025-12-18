from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Union
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Views:
    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    vol_win: int
    
    log_ret: pd.DataFrame = field(init=False)
    hl_spread: pd.DataFrame = field(init=False)
    oc_gap: pd.DataFrame = field(init=False)
    vol_z: pd.DataFrame = field(init=False)

    def __post_init__(self):
        c = self.close.replace(0.0, np.nan)
        prev = c.shift(1)
        
        object.__setattr__(self, "log_ret", np.log(c / prev))
        object.__setattr__(self, "hl_spread", (self.high - self.low) / c)
        object.__setattr__(self, "oc_gap", (self.open - prev) / prev)
        
        m_vol = self.volume.rolling(self.vol_win, min_periods=1).mean().replace(0.0, np.nan)
        object.__setattr__(self, "vol_z", (self.volume / m_vol) - 1.0)


def get_logreturn(v: Views, **kwargs) -> pd.DataFrame: return v.log_ret
def get_hlspread(v: Views, **kwargs) -> pd.DataFrame: return v.hl_spread
def get_ocgap(v: Views, **kwargs) -> pd.DataFrame: return v.oc_gap
def get_volumez(v: Views, **kwargs) -> pd.DataFrame: return v.vol_z

def get_magap(v: Views, window: int = 20, **kwargs) -> pd.DataFrame:
    ma = v.close.rolling(window, min_periods=1).mean()
    return (v.close / ma.replace(0.0, np.nan)) - 1.0

def get_rollingvol(v: Views, window: int = 20, **kwargs) -> pd.DataFrame:
    return v.log_ret.rolling(window, min_periods=1).std()

def get_rsi(v: Views, window: int = 14, **kwargs) -> pd.DataFrame:
    delta = v.close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # NOTE: Normalize to [-0.5, 0.5]
    return (rsi / 100.0) - 0.5 

def get_macd(v: Views, fast: int = 12, slow: int = 26, signal: int = 9, **kwargs) -> pd.DataFrame:
    ema_fast = v.close.ewm(span=fast, adjust=False, min_periods=1).mean()
    ema_slow = v.close.ewm(span=slow, adjust=False, min_periods=1).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=1).mean()
    return (macd_line - signal_line) / v.close.replace(0.0, np.nan)

def get_ma(v: Views, window: int = 20, **kwargs) -> pd.DataFrame:
    ma = v.close.rolling(window, min_periods=1).mean()
    return (v.close - ma) / ma.replace(0.0, np.nan)

def get_ema(v: Views, window: int = 20, **kwargs) -> pd.DataFrame:
    ema = v.close.ewm(span=window, adjust=False, min_periods=1).mean()
    return (v.close - ema) / ema.replace(0.0, np.nan)

def get_bb(v: Views, window: int = 20, num_std: float = 2.0, **kwargs) -> pd.DataFrame:
    ma = v.close.rolling(window, min_periods=1).mean()
    std = v.close.rolling(window, min_periods=1).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    bb_position = (v.close - lower) / (upper - lower).replace(0.0, np.nan)
    # NOTE: Center around 0
    return bb_position - 0.5

def get_atr(v: Views, window: int = 14, **kwargs) -> pd.DataFrame:
    prev_close = v.close.shift(1)
    hl = v.high - v.low
    hc = (v.high - prev_close).abs()
    lc = (v.low - prev_close).abs()
    tr = np.maximum.reduce([hl.to_numpy(), hc.to_numpy(), lc.to_numpy()])
    tr_df = pd.DataFrame(tr, index=v.close.index, columns=v.close.columns)
    atr = tr_df.rolling(window, min_periods=1).mean()
    return atr / v.close.replace(0.0, np.nan)

def get_downsidevol(v: Views, window: int = 20, **kwargs) -> pd.DataFrame:
    neg = v.log_ret.where(v.log_ret < 0.0, 0.0)
    dv = neg.pow(2).rolling(window, min_periods=1).mean().apply(np.sqrt)
    return dv

def get_ulcer(v: Views, window: int = 20, **kwargs) -> pd.DataFrame:
    rolling_max = v.close.rolling(window, min_periods=1).max()
    dd = (v.close / rolling_max) - 1.0
    ulcer = dd.pow(2).rolling(window, min_periods=1).mean().apply(np.sqrt)
    return ulcer

def get_stoch(v: Views, window: int = 14, smooth_k: int = 3, smooth_d: int = 3, **kwargs) -> pd.DataFrame:
    highest_high = v.high.rolling(window, min_periods=1).max()
    lowest_low = v.low.rolling(window, min_periods=1).min()
    raw_k = (v.close - lowest_low) / (highest_high - lowest_low).replace(0.0, np.nan)
    smooth_k_df = raw_k.rolling(smooth_k, min_periods=1).mean()
    smooth_d_df = smooth_k_df.rolling(smooth_d, min_periods=1).mean()
    return smooth_d_df.fillna(0.0) - 0.5

def get_mfi(v: Views, window: int = 14, **kwargs) -> pd.DataFrame:
    tp = (v.high + v.low + v.close) / 3.0
    mf = tp * v.volume
    delta_tp = tp.diff()
    pos_flow = mf.where(delta_tp > 0.0, 0.0)
    neg_flow = mf.where(delta_tp < 0.0, 0.0)
    pos_sum = pos_flow.rolling(window, min_periods=1).sum()
    neg_sum = neg_flow.rolling(window, min_periods=1).sum().replace(0.0, np.nan)
    mfr = pos_sum / neg_sum
    mfi = 100 - (100 / (1 + mfr))
    return (mfi / 100.0) - 0.5


REGISTRY = {
    "logreturn": get_logreturn,
    "hlspread": get_hlspread,
    "ocgap": get_ocgap,
    "volumez": get_volumez,
    "magap": get_magap,
    "rollingvol": get_rollingvol,
    "rsi": get_rsi,
    "macd": get_macd,
    "ma": get_ma,
    "ema": get_ema,
    "bb": get_bb,
    "atr": get_atr,
    "downsidevol": get_downsidevol,
    "ulcer": get_ulcer,
    "stoch": get_stoch,
    "mfi": get_mfi,
}


@dataclass(frozen=True)
class Panel:
    values: np.ndarray
    mask: np.ndarray


def make_features(
    frames: Dict[str, pd.DataFrame],
    features: Tuple[Any, ...],
    norm: str = "none",
    vol_win: int = 20,
    zero_invalid: bool = False,
) -> Panel:
    
    v = Views(
        close=frames["close"],
        open=frames["open"],
        high=frames["high"],
        low=frames["low"],
        volume=frames["volume"],
        vol_win=vol_win
    )
    
    tensors = []
    for feat in features:
        name, params = feat, {}
        if isinstance(feat, (tuple, list)):
            name, params = feat[0], feat[1]
        elif isinstance(feat, dict):
            name = feat.pop("name")
            params = feat
            
        func = REGISTRY.get(name)
        if not func: raise KeyError(f"Unknown feature: {name}")
        
        df = func(v, **params)
        tensors.append(df.to_numpy(copy=True))
        
    vals = np.stack(tensors, axis=-1)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        if norm == "asset":
            m = np.nanmean(vals, axis=0, keepdims=True)
            s = np.nanstd(vals, axis=0, keepdims=True)
            vals = (vals - m) / np.where(s == 0, 1.0, s)
        elif norm == "cross":
            m = np.nanmean(vals, axis=1, keepdims=True)
            s = np.nanstd(vals, axis=1, keepdims=True)
            vals = (vals - m) / np.where(s == 0, 1.0, s)
        
    mask = np.isfinite(vals).all(axis=2)
    if zero_invalid:
        mask &= vals.any(axis=2)
        
    vals = np.nan_to_num(vals, nan=0.0).astype(np.float32)
    
    return Panel(vals, mask)
