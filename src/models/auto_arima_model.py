import pandas as pd
from pmdarima import auto_arima
from .base import ForecastModel

class AutoARIMAModel(ForecastModel):
    def __init__(
        self,
        seasonal: bool = True,
        m: int = 7,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 1,
        max_P: int = 2,
        max_Q: int = 2,
        max_D: int = 1,
    ):
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_D = max_D
        self.model_ = None

    def fit(self, y: pd.Series):
        y = pd.Series(y).astype(float)
        self.model_ = auto_arima(
            y,
            seasonal=self.seasonal,
            m=int(self.m),
            max_p=int(self.max_p),
            max_q=int(self.max_q),
            max_d=int(self.max_d),
            max_P=int(self.max_P),
            max_Q=int(self.max_Q),
            max_D=int(self.max_D),
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            information_criterion="aicc",
        )
        return self

    def predict(self, n_periods: int, last_date, freq: str):
        if self.model_ is None:
            raise RuntimeError("Modelo no entrenado.")
        yhat, conf = self.model_.predict(n_periods=int(n_periods), return_conf_int=True, alpha=0.2)

        last_date = pd.to_datetime(last_date)
        dates = pd.date_range(start=last_date, periods=int(n_periods) + 1, freq=freq)[1:]

        return pd.DataFrame(
            {
                "ds": dates,
                "yhat": yhat,
                "yhat_lower": conf[:, 0],
                "yhat_upper": conf[:, 1],
            }
        )
