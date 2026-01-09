import streamlit as st
import pandas as pd

from src.utils.data_prep import (
    load_data,
    build_series_sum,
    build_series_count,
    detect_catalog_error_description,
)
from src.utils.metrics import regression_metrics
from src.utils.plots import plot_history_and_forecast, plot_train_test_forecast

from src.models.auto_arima_model import AutoARIMAModel
from src.models.ets_model import ETSModel
from src.models.prophet_model import ProphetModel
from src.models.rf_lag_model import RandomForestLagModel
from src.models.xgb_lag_model import XGBoostLagModel

st.set_page_config(page_title="ATM Forecasting Lab", layout="wide")

st.title("ATM Forecasting Lab (Streamlit)")
st.caption(
    "Soporta 2 casos: (A) pronóstico de demanda (suma de transacciones) o (B) pronóstico de frecuencia de errores usando un catálogo ERROR/DESCRIPCIÓN + eventos."
)

with st.sidebar:
    st.header("1) Cargar archivos")
    uploaded_main = st.file_uploader("Archivo principal (CSV/Excel)", type=["csv", "xlsx", "xls"])
    uploaded_events = st.file_uploader(
        "Archivo de eventos (opcional; para pronóstico de errores)",
        type=["csv", "xlsx", "xls"],
        help="Si tu archivo principal es un catálogo (ERROR, DESCRIPCIÓN), sube aquí el log/eventos con fecha y código de error.",
    )

    st.header("2) Serie temporal")
    freq = st.selectbox("Frecuencia de agregación", options=["D", "H", "W", "M"], index=0)
    fill_missing = st.selectbox("Tratamiento de faltantes", options=["zeros", "ffill", "interpolate"], index=0)

    st.header("3) Entrenamiento / evaluación")
    test_size = st.number_input("Tamaño de test (n periodos al final)", min_value=7, max_value=365, value=28, step=1)
    horizon = st.number_input("Horizonte futuro (n periodos)", min_value=1, max_value=365, value=14, step=1)

    st.header("4) Modelo")
    model_name = st.selectbox(
        "Algoritmo",
        options=[
            "Auto-ARIMA (pmdarima)",
            "ETS / Holt-Winters",
            "Prophet",
            "Random Forest (lags)",
            "XGBoost (lags)",
        ],
        index=0,
    )

    st.header("5) Parámetros")
    params = {}

    if model_name == "Auto-ARIMA (pmdarima)":
        params["seasonal"] = st.checkbox("Estacional", value=True)
        params["m"] = st.number_input("m (periodicidad estacional)", min_value=1, max_value=365, value=7, step=1)
        params["max_p"] = st.slider("max_p", 1, 10, 5)
        params["max_q"] = st.slider("max_q", 1, 10, 5)
        params["max_d"] = st.slider("max_d", 0, 2, 1)
        params["max_P"] = st.slider("max_P", 0, 5, 2)
        params["max_Q"] = st.slider("max_Q", 0, 5, 2)
        params["max_D"] = st.slider("max_D", 0, 1, 1)

    elif model_name == "ETS / Holt-Winters":
        params["trend"] = st.selectbox("Tendencia", options=["add", "mul", None], index=0)
        params["seasonal"] = st.selectbox("Estacionalidad", options=["add", "mul", None], index=0)
        params["seasonal_periods"] = st.number_input("Periodos estacionales", min_value=2, max_value=365, value=7, step=1)
        params["damped_trend"] = st.checkbox("Tendencia amortiguada", value=False)

    elif model_name == "Prophet":
        params["weekly_seasonality"] = st.checkbox("Estacionalidad semanal", value=True)
        params["yearly_seasonality"] = st.checkbox("Estacionalidad anual", value=False)
        params["daily_seasonality"] = st.checkbox("Estacionalidad diaria", value=False)
        params["seasonality_mode"] = st.selectbox("Modo estacionalidad", options=["additive", "multiplicative"], index=0)
        params["changepoint_prior_scale"] = st.slider("changepoint_prior_scale", 0.001, 0.5, 0.05)

    elif model_name == "Random Forest (lags)":
        params["lags"] = st.number_input("Número de lags", min_value=7, max_value=90, value=28, step=1)
        params["n_estimators"] = st.slider("n_estimators", 100, 1500, 500, step=50)
        params["max_depth"] = st.slider("max_depth", 2, 50, 12, step=1)
        params["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 20, 2, step=1)
        params["add_calendar"] = st.checkbox("Agregar variables calendario", value=True)

    elif model_name == "XGBoost (lags)":
        params["lags"] = st.number_input("Número de lags", min_value=7, max_value=90, value=28, step=1)
        params["n_estimators"] = st.slider("n_estimators", 200, 3000, 800, step=50)
        params["max_depth"] = st.slider("max_depth", 2, 20, 6, step=1)
        params["learning_rate"] = st.slider("learning_rate", 0.005, 0.3, 0.05)
        params["subsample"] = st.slider("subsample", 0.5, 1.0, 0.9)
        params["colsample_bytree"] = st.slider("colsample_bytree", 0.5, 1.0, 0.9)
        params["reg_lambda"] = st.slider("reg_lambda", 0.0, 5.0, 1.0)
        params["add_calendar"] = st.checkbox("Agregar variables calendario", value=True)

st.divider()

if not uploaded_main:
    st.info("Carga un archivo principal para comenzar.")
    st.stop()

df_main = load_data(uploaded_main)
st.subheader("Archivo principal: vista previa")
st.dataframe(df_main.head(30), use_container_width=True)

is_catalog, catalog_error_col, catalog_desc_col = detect_catalog_error_description(df_main)

# Mode selection (auto, but user can override)
mode_default = "Catálogo de errores (ERROR/DESCRIPCIÓN) + eventos" if is_catalog else "Transacciones (serie temporal: suma de monto)"
mode = st.selectbox(
    "Modo de trabajo (auto-detectado)",
    options=[
        "Transacciones (serie temporal: suma de monto)",
        "Catálogo de errores (ERROR/DESCRIPCIÓN) + eventos",
    ],
    index=0 if mode_default.startswith("Transacciones") else 1,
)

if mode == "Transacciones (serie temporal: suma de monto)":
    with st.expander("Configurar columnas y filtros (Transacciones)"):
        cols = list(df_main.columns)
        date_col = st.selectbox("Columna de fecha/hora", options=cols, index=0)
        target_col = st.selectbox("Columna objetivo (monto)", options=cols, index=min(1, len(cols)-1))
        group_col = st.selectbox("Columna de ATM (opcional)", options=["(ninguna)"] + cols, index=0)
        group_col = None if group_col == "(ninguna)" else group_col

        group_value = None
        if group_col:
            values = df_main[group_col].dropna().astype(str).unique().tolist()
            values = sorted(values)[:5000]
            group_value = st.selectbox("Seleccionar ATM (o grupo)", options=["(AGREGADO)"] + values, index=0)
            if group_value == "(AGREGADO)":
                group_value = None

    try:
        series_df = build_series_sum(
            df=df_main,
            date_col=date_col,
            target_col=target_col,
            group_col=group_col,
            group_value=group_value,
            freq=freq,
            fill_missing=fill_missing,
        )
        if len(series_df) <= 1 and freq in {"D", "W", "M"}:
            timestamps = pd.to_datetime(df_main[date_col], errors="coerce").dropna()
            if not timestamps.empty and timestamps.dt.floor("D").nunique() == 1 and timestamps.dt.hour.nunique() > 1:
                st.info("Detecté datos intradía en una sola fecha. Ajusté la frecuencia a horas (H) para graficar.")
                freq = "H"
                series_df = build_series_sum(
                    df=df_main,
                    date_col=date_col,
                    target_col=target_col,
                    group_col=group_col,
                    group_value=group_value,
                    freq=freq,
                    fill_missing=fill_missing,
                )
        series_label = f"Suma de {target_col}"
    except Exception as e:
        st.error(f"No pude construir la serie temporal. Detalle: {e}")
        st.stop()

else:
    # Catalog + events
    if not is_catalog:
        st.warning(
            "Seleccionaste modo de catálogo, pero el archivo principal no parece un catálogo típico (ERROR, DESCRIPCIÓN). "
            "Aun así puedes continuar si indicas columnas manualmente."
        )

    with st.expander("Configurar catálogo (ERROR/DESCRIPCIÓN)"):
        cols = list(df_main.columns)
        error_col = st.selectbox("Columna código de error", options=cols, index=cols.index(catalog_error_col) if catalog_error_col in cols else 0)
        desc_col = st.selectbox("Columna descripción", options=cols, index=cols.index(catalog_desc_col) if catalog_desc_col in cols else min(1, len(cols)-1))

    st.subheader("Catálogo de errores")
    st.dataframe(df_main[[error_col, desc_col]].drop_duplicates().head(1000), use_container_width=True)

    if not uploaded_events:
        st.info("Para pronosticar frecuencia de errores, sube también el archivo de eventos/logs (con fecha y código de error).")
        st.stop()

    df_events = load_data(uploaded_events)
    st.subheader("Archivo de eventos: vista previa")
    st.dataframe(df_events.head(30), use_container_width=True)

    with st.expander("Configurar columnas y filtros (Eventos)"):
        cols_e = list(df_events.columns)
        date_col_e = st.selectbox("Columna fecha/hora (eventos)", options=cols_e, index=0)
        event_col_e = st.selectbox("Columna código de error (eventos)", options=cols_e, index=min(1, len(cols_e)-1))
        group_col_e = st.selectbox("Columna ATM (opcional, eventos)", options=["(ninguna)"] + cols_e, index=0)
        group_col_e = None if group_col_e == "(ninguna)" else group_col_e

        # choose error code to forecast
        codes = df_events[event_col_e].dropna().astype(str).unique().tolist()
        codes = sorted(codes)[:10000]
        chosen_code = st.selectbox("Error a pronosticar", options=codes, index=0)

        group_value_e = None
        if group_col_e:
            values = df_events[group_col_e].dropna().astype(str).unique().tolist()
            values = sorted(values)[:5000]
            group_value_e = st.selectbox("Seleccionar ATM (opcional)", options=["(AGREGADO)"] + values, index=0)
            if group_value_e == "(AGREGADO)":
                group_value_e = None

    # Join to get description (optional)
    try:
        catalog_map = df_main[[error_col, desc_col]].drop_duplicates().copy()
        catalog_map[error_col] = catalog_map[error_col].astype(str)
        desc_lookup = dict(zip(catalog_map[error_col], catalog_map[desc_col]))
        chosen_desc = desc_lookup.get(str(chosen_code), "")
        if chosen_desc:
            st.caption(f"Descripción: {chosen_desc}")
    except Exception:
        chosen_desc = ""

    try:
        series_df = build_series_count(
            df=df_events,
            date_col=date_col_e,
            event_col=event_col_e,
            event_value=str(chosen_code),
            group_col=group_col_e,
            group_value=group_value_e,
            freq=freq,
            fill_missing=fill_missing,
        )
        if len(series_df) <= 1 and freq in {"D", "W", "M"}:
            timestamps = pd.to_datetime(df_events[date_col_e], errors="coerce").dropna()
            if not timestamps.empty and timestamps.dt.floor("D").nunique() == 1 and timestamps.dt.hour.nunique() > 1:
                st.info("Detecté datos intradía en una sola fecha. Ajusté la frecuencia a horas (H) para graficar.")
                freq = "H"
                series_df = build_series_count(
                    df=df_events,
                    date_col=date_col_e,
                    event_col=event_col_e,
                    event_value=str(chosen_code),
                    group_col=group_col_e,
                    group_value=group_value_e,
                    freq=freq,
                    fill_missing=fill_missing,
                )
        series_label = f"Conteo de error {chosen_code}" + (f" ({chosen_desc})" if chosen_desc else "")
    except Exception as e:
        st.error(f"No pude construir la serie de eventos. Detalle: {e}")
        st.stop()

st.subheader("Serie temporal")
fig_hist = plot_history_and_forecast(series_df, forecast_df=None, title=series_label)
st.plotly_chart(fig_hist, use_container_width=True)

if len(series_df) < (int(test_size) + 30):
    st.warning("Pocos datos para el test seleccionado. Considera reducir test_size o cargar más historial.")

if len(series_df) < 3:
    st.error("La serie temporal tiene muy pocos puntos para entrenar un modelo. Ajusta la frecuencia o carga más historial.")
    st.stop()

test_size_eff = int(test_size)
max_test_size = max(1, len(series_df) - 2)
if test_size_eff > max_test_size:
    st.warning(f"test_size ajustado automáticamente a {max_test_size} por falta de datos.")
    test_size_eff = max_test_size

horizon_eff = int(horizon)
max_horizon = max(1, len(series_df))
if horizon_eff > max_horizon:
    st.warning(f"Horizonte ajustado automáticamente a {max_horizon} por falta de datos.")
    horizon_eff = max_horizon

run = st.button("Entrenar y pronosticar", type="primary")
if not run:
    st.stop()

y = series_df["y"].copy()

if test_size_eff >= len(y):
    st.error("test_size es demasiado grande para el tamaño de la serie. Reduce test_size.")
    st.stop()

y_train = y.iloc[:-test_size_eff]
y_test = y.iloc[-test_size_eff:]

if model_name == "Auto-ARIMA (pmdarima)":
    model = AutoARIMAModel(**params)
elif model_name == "ETS / Holt-Winters":
    model = ETSModel(**params)
elif model_name == "Prophet":
    model = ProphetModel(**params)
elif model_name == "Random Forest (lags)":
    model = RandomForestLagModel(**params)
elif model_name == "XGBoost (lags)":
    model = XGBoostLagModel(**params)
else:
    st.error("Modelo no soportado.")
    st.stop()

with st.spinner("Entrenando y evaluando..."):
    model.fit(y_train)
    test_pred = model.predict(n_periods=len(y_test), last_date=y_train.index[-1], freq=freq)
    test_pred = test_pred.set_index("ds").reindex(y_test.index)

    metrics = regression_metrics(y_true=y_test.values, y_pred=test_pred["yhat"].values)
    st.subheader("Desempeño en test")
    st.json(metrics)

    model.fit(y)
    future_fcst = model.predict(n_periods=horizon_eff, last_date=y.index[-1], freq=freq)

st.subheader("Pronóstico sobre test y futuro")
fig_tt = plot_train_test_forecast(
    y_train=y_train,
    y_test=y_test,
    yhat_test=test_pred.reset_index().rename(columns={"index": "ds"}),
    yhat_future=future_fcst,
    title=f"{model_name} | {series_label} | test={test_size_eff} | horizonte={horizon_eff}",
)
st.plotly_chart(fig_tt, use_container_width=True)

st.subheader("Pronóstico futuro (tabla)")
st.dataframe(future_fcst, use_container_width=True)

csv_bytes = future_fcst.to_csv(index=False).encode("utf-8")
st.download_button("Descargar pronóstico (CSV)", data=csv_bytes, file_name="forecast.csv", mime="text/csv")
