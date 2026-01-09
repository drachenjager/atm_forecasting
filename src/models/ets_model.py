import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .base import ForecastModel

class ETSModel(ForecastModel):
    def __init__(self, trend="add", seasonal="add", seasonal_periods=7, damped_trend=False):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.fit_ = None

    def fit(self, y: pd.Series):
        y = pd.Series(y).astype(float)
        model = ExponentialSmoothing(
            y,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=int(self.seasonal_periods) if self.seasonal is not None else None,
            damped_trend=bool(self.damped_trend),
            initialization_method="estimated",
        )
        self.fit_ = model.fit(optimized=True)
        return self

    def predict(self, n_periods: int, last_date, freq: str):
        if self.fit_ is None:
            raise RuntimeError("Modelo no entrenado.")
        last_date = pd.to_datetime(last_date)
        dates = pd.date_range(start=last_date, periods=int(n_periods) + 1, freq=freq)[1:]
        yhat = self.fit_.forecast(int(n_periods)).values
        return pd.DataFrame({"ds": dates, "yhat": yhat})
