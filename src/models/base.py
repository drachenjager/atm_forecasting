from __future__ import annotations
import pandas as pd

class ForecastModel:
    def fit(self, y: pd.Series) -> "ForecastModel":
        raise NotImplementedError

    def predict(self, n_periods: int, last_date, freq: str) -> pd.DataFrame:
        # Return DataFrame with columns:
        # - ds (datetime)
        # - yhat
        # - yhat_lower (optional)
        # - yhat_upper (optional)
        raise NotImplementedError
