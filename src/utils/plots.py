import pandas as pd
import plotly.graph_objects as go
from typing import Optional

def plot_history_and_forecast(series_df: pd.DataFrame, forecast_df: Optional[pd.DataFrame], title: str = ""):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series_df.index,
            y=series_df["y"],
            mode="lines+markers",
            name="Histórico",
            marker=dict(size=6),
        )
    )

    if forecast_df is not None and len(forecast_df) > 0:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["ds"],
                y=forecast_df["yhat"],
                mode="lines+markers",
                name="Pronóstico",
                marker=dict(size=6),
            )
        )
        if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["ds"], y=forecast_df["yhat_upper"],
                mode="lines", name="Upper", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df["ds"], y=forecast_df["yhat_lower"],
                mode="lines", name="Lower", line=dict(width=0),
                fill="tonexty", showlegend=False,
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Demanda (y)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

def plot_train_test_forecast(y_train, y_test, yhat_test: pd.DataFrame, yhat_future: pd.DataFrame, title: str = ""):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_train.index,
            y=y_train.values,
            mode="lines+markers",
            name="Train",
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_test.index,
            y=y_test.values,
            mode="lines+markers",
            name="Test (real)",
            marker=dict(size=6),
        )
    )

    if "ds" in yhat_test.columns:
        fig.add_trace(
            go.Scatter(
                x=yhat_test["ds"],
                y=yhat_test["yhat"],
                mode="lines+markers",
                name="Test (pred)",
                marker=dict(size=6),
            )
        )

    if yhat_future is not None and len(yhat_future) > 0:
        fig.add_trace(
            go.Scatter(
                x=yhat_future["ds"],
                y=yhat_future["yhat"],
                mode="lines+markers",
                name="Futuro",
                marker=dict(size=6),
            )
        )

        if "yhat_lower" in yhat_future.columns and "yhat_upper" in yhat_future.columns:
            fig.add_trace(go.Scatter(
                x=yhat_future["ds"], y=yhat_future["yhat_upper"],
                mode="lines", name="Upper", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=yhat_future["ds"], y=yhat_future["yhat_lower"],
                mode="lines", name="Lower", line=dict(width=0),
                fill="tonexty", showlegend=False,
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Demanda (y)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
