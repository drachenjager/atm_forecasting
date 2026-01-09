import pandas as pd
import numpy as np
from xgboost import XGBRegressor

from .base import ForecastModel
from .lag_features import make_supervised

class XGBoostLagModel(ForecastModel):
    def __init__(
        self,
        lags=28,
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        add_calendar=True,
    ):
        self.lags = int(lags)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.colsample_bytree = float(colsample_bytree)
        self.reg_lambda = float(reg_lambda)
        self.add_calendar = bool(add_calendar)

        self.model_ = None
        self.history_ = None

    def fit(self, y: pd.Series):
        y = pd.Series(y).astype(float)
        X, y_sup = make_supervised(y, lags=self.lags, add_calendar=self.add_calendar)

        self.model_ = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        self.model_.fit(X, y_sup)
        self.history_ = y.copy()
        return self

    def _predict_recursive(self, dates: pd.DatetimeIndex):
        hist = self.history_.copy()
        preds = []
        for d in dates:
            row = {}
            for i in range(1, self.lags + 1):
                row[f"lag_{i}"] = float(hist.iloc[-i])

            row["roll_mean_7"] = float(hist.iloc[-7:].mean()) if len(hist) >= 7 else float(hist.mean())
            row["roll_std_7"] = float(hist.iloc[-7:].std(ddof=0)) if len(hist) >= 7 else 0.0
            row["roll_mean_28"] = float(hist.iloc[-28:].mean()) if len(hist) >= 28 else float(hist.mean())

            if self.add_calendar:
                row["dow"] = d.dayofweek
                row["dom"] = d.day
                row["month"] = d.month
                row["is_weekend"] = int(d.dayofweek >= 5)

            X_row = pd.DataFrame([row], index=[d])
            yhat = float(self.model_.predict(X_row)[0])
            yhat = max(0.0, yhat)
            preds.append(yhat)
            hist = pd.concat([hist, pd.Series([yhat], index=[d])])
        return np.array(preds)

    def predict(self, n_periods: int, last_date, freq: str):
        if self.model_ is None or self.history_ is None:
            raise RuntimeError("Modelo no entrenado.")
        last_date = pd.to_datetime(last_date)
        dates = pd.date_range(start=last_date, periods=int(n_periods) + 1, freq=freq)[1:]
        yhat = self._predict_recursive(dates)
        return pd.DataFrame({"ds": dates, "yhat": yhat})
