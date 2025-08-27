#!/usr/bin/env python3
"""
yfinance_stock.py

OUTPUT SHAPE (df): pandas.DataFrame with REQUIRED columns:
- Date  (datetime-like)
- Close (float)

Dependency:
- pip install yfinance pandas numpy
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional


def fetch_stock_data(
    ticker: str = "NVDA",
    period: str = "60d",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance and return a DataFrame
    with exactly ['Date', 'Close'] as required columns.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period, interval=interval)

    # Ensure Date column
    if isinstance(hist.index, pd.DatetimeIndex):
        hist = hist.reset_index()  # gives 'Date' by default
    else:
        hist = hist.reset_index().rename(columns={"index": "Date"})

    # Standardize/keep only required columns
    if "Date" not in hist.columns:
        for cand in ("Datetime", "DateTime", "date", "timestamp"):
            if cand in hist.columns:
                hist = hist.rename(columns={cand: "Date"})
                break

    df = hist[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).reset_index(drop=True)
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: Compute RSI(14) and MACD(12/26/9) for analysis.
    Note: Not required by the output schema.
    """
    out = df.copy()

    # RSI (14)
    delta = out["Close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=out.index).rolling(window=14, min_periods=14).mean()
    avg_loss = pd.Series(loss, index=out.index).rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    out["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema_12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = out["Close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["MACD"] = macd
    out["MACD_Signal"] = signal
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    return out


def nvda_data_pipeline(include_indicators: bool = False,
                       period: str = "60d",
                       interval: str = "1d") -> pd.DataFrame:
    """
    Convenience pipeline for NVDA; returns only ['Date','Close'] per spec.
    """
    df = fetch_stock_data("NVDA", period, interval)
    if include_indicators:
        _ = compute_technical_indicators(df)  # computed but not returned
    return df.loc[:, ["Date", "Close"]].copy()


def collect(
    ticker: str = "NVDA",
    period: str = "60d",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Pipeline-friendly entrypoint. Returns DataFrame with columns ['Date', 'Close'].
    """
    return fetch_stock_data(ticker=ticker, period=period, interval=interval).loc[:, ["Date", "Close"]].copy()


if __name__ == "__main__":
    df = nvda_data_pipeline()
    print(df.tail())
