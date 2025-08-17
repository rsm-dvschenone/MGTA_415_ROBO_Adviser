
"""
data_analysis/price_indicators.py
---------------------------------
RSI & MACD analysis helpers (works with a simple DataFrame of dates & close prices).
Defaults match common conventions: RSI(14) and MACD(12, 26, 9).

INPUT SHAPE (df): pandas.DataFrame with REQUIRED columns:
- Date  (datetime-like or parseable string)
- Close (float)

"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any

def _ensure_datetime_index(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    if df.index.dtype.kind in "M":
        return df.sort_index()
    if date_col in df.columns:
        out = df.copy()
        out[date_col] = pd.to_datetime(out[date_col])
        out = out.sort_values(date_col).set_index(date_col)
        return out
    raise ValueError('DataFrame must have a DatetimeIndex or a "Date" column.')

def rsi(series: pd.Series, period: int = 14, method: str = "wilder") -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).astype(float)
    loss = (-delta.where(delta < 0, 0.0)).astype(float)
    if method.lower() == "wilder":
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    else:
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    r = 100 - (100 / (1 + rs))
    return r.fillna(50.0)

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"MACD": macd_line, "MACD_signal": signal_line, "MACD_hist": hist})

def analyze_rsi_macd(
    df: pd.DataFrame,
    *,
    price_col: str = "Close",
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0,
) -> Dict[str, Any]:
    x = _ensure_datetime_index(df)
    if price_col not in x.columns:
        raise ValueError(f'price_col="{price_col}" not in DataFrame.')
    px = x[price_col].astype(float)

    rsi_series = rsi(px, period=rsi_period)
    macd_df = macd(px, fast=macd_fast, slow=macd_slow, signal=macd_signal)

    out = x.copy()
    out["RSI"] = rsi_series
    out = out.join(macd_df, how="left")

    def _rsi_state(v):
        if v >= rsi_overbought: return "overbought"
        if v <= rsi_oversold: return "oversold"
        return "neutral"
    out["rsi_state"] = out["RSI"].apply(_rsi_state)

    def _macd_state(row):
        if pd.isna(row["MACD"]) or pd.isna(row["MACD_signal"]): return "neutral"
        return "bullish" if row["MACD"] > row["MACD_signal"] else "bearish"
    out["macd_state"] = out.apply(_macd_state, axis=1)

    prev = out["MACD"].shift(1) - out["MACD_signal"].shift(1)
    curr = out["MACD"] - out["MACD_signal"]
    cross_up = (prev <= 0) & (curr > 0)
    cross_down = (prev >= 0) & (curr < 0)
    out["macd_cross"] = np.where(cross_up, "up", np.where(cross_down, "down", "none"))

    last = out.iloc[-1]
    summary = {
        "n_obs": int(len(out)),
        "start": str(out.index[0].date()),
        "end": str(out.index[-1].date()),
        "last_close": float(px.iloc[-1]),
        "RSI_last": round(float(last["RSI"]), 2),
        "RSI_state_last": str(last["rsi_state"]),
        "MACD_last": round(float(last["MACD"]), 4) if pd.notnull(last["MACD"]) else None,
        "MACD_signal_last": round(float(last["MACD_signal"]), 4) if pd.notnull(last["MACD_signal"]) else None,
        "MACD_hist_last": round(float(last["MACD_hist"]), 4) if pd.notnull(last["MACD_hist"]) else None,
        "MACD_state_last": str(last["macd_state"]),
        "MACD_cross_last": str(last["macd_cross"]),
        "params": {"RSI_period": rsi_period, "MACD": [macd_fast, macd_slow, macd_signal],
                   "RSI_overbought": rsi_overbought, "RSI_oversold": rsi_oversold},
    }
    return {"summary": summary, "frame": out}
