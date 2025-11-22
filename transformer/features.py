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
