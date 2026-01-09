import pandas as pd
from prophet import Prophet
from .base import ForecastModel

class ProphetModel(ForecastModel):
    def __init__(
        self,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = False,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.05,
    ):
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model_ = None

    def fit(self, y: pd.Series):
        y = pd.Series(y).astype(float)
        df = pd.DataFrame({"ds": pd.to_datetime(y.index), "y": y.values})
        self.model_ = Prophet(
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=float(self.changepoint_prior_scale),
        )
        self.model_.fit(df)
        return self

    def predict(self, n_periods: int, last_date, freq: str):
        if self.model_ is None:
            raise RuntimeError("Modelo no entrenado.")
        last_date = pd.to_datetime(last_date)
        future = pd.DataFrame({"ds": pd.date_range(start=last_date, periods=int(n_periods) + 1, freq=freq)[1:]})
        fcst = self.model_.predict(future)
        return fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
