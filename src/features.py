import pandas as pd
import numpy as np

def add_lagged_returns(df: pd.DataFrame, lags):
    for L in lags:
        df[f"ret_lag{L}"] = df["ret"].shift(L)
    return df

def add_sma(df: pd.DataFrame, windows):
    for w in windows:
        df[f"sma_{w}"] = df["price"].rolling(w).mean()
        df[f"sma_ret_{w}"] = df["ret"].rolling(w).mean()
    return df

def add_ema(df: pd.DataFrame, windows):
    for w in windows:
        df[f"ema_{w}"] = df["price"].ewm(span=w, adjust=False).mean()
    return df

def add_rsi(df: pd.DataFrame, windows):
    # classic Wilder's RSI
    delta = df["price"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    for w in windows:
        roll_up = up.ewm(alpha=1/w, adjust=False).mean()
        roll_down = down.ewm(alpha=1/w, adjust=False).mean()
        rs = roll_up / (roll_down + 1e-12)
        df[f"rsi_{w}"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9):
    ema_fast = df["price"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["price"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = sig
    df["macd_hist"] = macd - sig
    return df

def add_volatility(df: pd.DataFrame, windows):
    for w in windows:
        df[f"vol_{w}"] = df["ret"].rolling(w).std()
    return df

def build_features(df: pd.DataFrame, cfg: dict):
    df = add_lagged_returns(df, cfg.get("lags", []))
    df = add_sma(df, cfg.get("sma_windows", []))
    df = add_ema(df, cfg.get("ema_windows", []))
    df = add_rsi(df, cfg.get("rsi_windows", []))
    macd = cfg.get("macd", None)
    if macd:
        df = add_macd(df, macd.get("fast",12), macd.get("slow",26), macd.get("signal",9))
    df = add_volatility(df, cfg.get("vol_windows", []))
    return df
