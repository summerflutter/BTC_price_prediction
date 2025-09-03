import pandas as pd
import numpy as np

def load_main(csv_path: str, timestamp_col: str, price_col: str, freq: str = "D", tz: str = "UTC"):
    df = pd.read_csv(csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    if tz:
        df[timestamp_col] = df[timestamp_col].dt.tz_convert(tz)
    df = df[[timestamp_col, price_col]].rename(columns={timestamp_col: "timestamp", price_col: "price"})
    df = df.set_index("timestamp").sort_index()
    if freq:
        df = df.resample(freq).last().dropna()
    return df

def compute_returns(df: pd.DataFrame, log_return: bool = True):
    px = df["price"].astype(float)
    if log_return:
        ret = np.log(px).diff()
    else:
        ret = px.pct_change()
    df = df.copy()
    df["ret"] = ret
    return df

def merge_exogenous(base: pd.DataFrame, assets):
    out = base.copy()
    for spec in assets:
        name = spec["name"]
        csv = spec["csv"]
        price_col = spec.get("price_col", "price")
        df = pd.read_csv(csv)
        df[spec.get("timestamp_col", "timestamp")] = pd.to_datetime(df.get(spec.get("timestamp_col", "timestamp")), utc=True)
        df = df.rename(columns={spec.get("timestamp_col", "timestamp"): "timestamp", price_col: f"{name}_price"})
        df = df.set_index("timestamp").sort_index()
        df = df[[f"{name}_price"]]
        out = out.join(df, how="left")
    return out

def train_val_test_indices(n, train, val, test, step=None):
    if step is None:
        step = test
    i = 0
    while True:
        tr_start = i
        tr_end = tr_start + train
        va_end = tr_end + val
        te_end = va_end + test
        if te_end > n:
            break
        yield (slice(tr_start, tr_end), slice(tr_end, va_end), slice(va_end, te_end))
        i += step
