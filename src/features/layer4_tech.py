"""Layer 4 -- classical technical indicators.

All pure, no-lookahead. Uses Wilder smoothing (EMA with alpha=1/n)
wherever Wilder's original formulation calls for it (RSI, ADX/DI).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute_layer4(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    vol = df["volume"].astype("float64") if "volume" in df.columns else pd.Series(0.0, index=df.index)

    out: dict[str, pd.Series] = {}

    # RSI (Wilder) -----------------------------------------------------
    out["rsi_14"] = _rsi(close, cfg.window_rsi)

    # MACD -------------------------------------------------------------
    fast, slow, sig = cfg.macd_windows
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=sig, adjust=False, min_periods=slow + sig - 1).mean()
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal

    # Bollinger --------------------------------------------------------
    n = cfg.window_bb
    m = cfg.bb_stdev_mult
    bb_mid = close.rolling(n, min_periods=n).mean()
    bb_std = close.rolling(n, min_periods=n).std(ddof=0)
    bb_upper = bb_mid + m * bb_std
    bb_lower = bb_mid - m * bb_std
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower
    out["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
    denom = (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_pct_b"] = (close - bb_lower) / denom

    # EMAs + distances --------------------------------------------------
    for span in cfg.ema_windows:
        col = f"ema_{span}"
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        out[col] = ema
        out[f"{col}_dist"] = (close - ema) / close.replace(0, np.nan) * 100.0

    # ADX / DI ---------------------------------------------------------
    adx_n = cfg.window_adx
    adx_frame = _adx(high, low, close, adx_n)
    out["adx_14"] = adx_frame["adx"]
    out["plus_di_14"] = adx_frame["+di"]
    out["minus_di_14"] = adx_frame["-di"]

    # Rolling VWAP -----------------------------------------------------
    typical = (high + low + close) / 3.0
    pv = typical * vol
    roll_pv = pv.rolling(cfg.window_vwap, min_periods=cfg.window_vwap).sum()
    roll_v = vol.rolling(cfg.window_vwap, min_periods=cfg.window_vwap).sum().replace(0, np.nan)
    vwap = roll_pv / roll_v
    out["vwap_rolling_20_dist"] = (close - vwap) / vwap * 100.0

    # OBV --------------------------------------------------------------
    dc_sign = np.sign(close.diff().fillna(0.0))
    obv = (vol * dc_sign).cumsum()
    out["obv"] = obv
    out["obv_slope_20"] = _rolling_slope(obv, cfg.window_obv_slope)

    # Supertrend (bar-close variant) -----------------------------------
    st, st_dir = _supertrend(
        high, low, close,
        atr_n=cfg.supertrend_atr_window,
        mult=cfg.supertrend_mult,
    )
    out["supertrend"] = st
    out["supertrend_dir"] = st_dir

    return pd.DataFrame(out, index=df.index)


# ---- helpers ----------------------------------------------------------
def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_dn = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(~avg_dn.eq(0), 100.0)
    return rsi


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.DataFrame:
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    plus_dm_s = pd.Series(plus_dm, index=high.index)
    minus_dm_s = pd.Series(minus_dm, index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_di = 100.0 * plus_dm_s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return pd.DataFrame({"+di": plus_di, "-di": minus_di, "adx": adx})


def _rolling_slope(s: pd.Series, n: int) -> pd.Series:
    """Linear-regression slope over a window of n bars.

    Uses the closed-form formula with x = 0..n-1.
    """
    if n < 2:
        return pd.Series(np.nan, index=s.index)
    x = np.arange(n, dtype="float64")
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    return s.rolling(n, min_periods=n).apply(_slope, raw=True)


def _supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    *, atr_n: int, mult: float,
) -> tuple[pd.Series, pd.Series]:
    hl2 = (high + low) / 2.0
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / atr_n, adjust=False, min_periods=atr_n).mean()

    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr

    n = len(close)
    upper = upper_basic.to_numpy().copy()
    lower = lower_basic.to_numpy().copy()
    st = np.full(n, np.nan, dtype="float64")
    direction = np.zeros(n, dtype="float64")
    c = close.to_numpy()

    # Find first fully-warm bar.
    first = int(np.argmax(~np.isnan(atr.to_numpy())))
    if np.isnan(atr.iloc[first]):
        return pd.Series(st, index=close.index), pd.Series(direction, index=close.index)

    # Initialize at first warm bar.
    direction[first] = 1.0
    st[first] = lower[first]
    for i in range(first + 1, n):
        # Final band adjustment.
        if upper[i] > upper[i - 1] and c[i - 1] <= upper[i - 1]:
            upper[i] = upper[i - 1]
        if lower[i] < lower[i - 1] and c[i - 1] >= lower[i - 1]:
            lower[i] = lower[i - 1]
        # Direction flip.
        if direction[i - 1] == 1.0:
            if c[i] < lower[i]:
                direction[i] = -1.0
                st[i] = upper[i]
            else:
                direction[i] = 1.0
                st[i] = lower[i]
        else:
            if c[i] > upper[i]:
                direction[i] = 1.0
                st[i] = lower[i]
            else:
                direction[i] = -1.0
                st[i] = upper[i]

    return (
        pd.Series(st, index=close.index, dtype="float64"),
        pd.Series(direction, index=close.index, dtype="float64"),
    )
