import pandas as pd

def make_supervised(y: pd.Series, lags: int = 28, add_calendar: bool = True):
    y = pd.Series(y).astype(float)
    df = pd.DataFrame({"ds": pd.to_datetime(y.index), "y": y.values}).set_index("ds")

    for i in range(1, int(lags) + 1):
        df[f"lag_{i}"] = df["y"].shift(i)

    df["roll_mean_7"] = df["y"].shift(1).rolling(7).mean()
    df["roll_std_7"] = df["y"].shift(1).rolling(7).std()
    df["roll_mean_28"] = df["y"].shift(1).rolling(28).mean()

    if add_calendar:
        idx = df.index
        df["dow"] = idx.dayofweek
        df["dom"] = idx.day
        df["month"] = idx.month
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    df = df.dropna()
    X = df.drop(columns=["y"])
    y_out = df["y"]
    return X, y_out
