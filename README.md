# ATM Forecasting Lab (Streamlit)

Aplicación en Streamlit para pronosticar series temporales de ATMs.

## Casos soportados

### A) Transacciones → pronóstico de demanda (suma de monto)
- Archivo con columna de **fecha/hora** (p. ej. `transaction_date`)
- Columna objetivo numérica (monto de retiros, cash-out, etc.)
- Opcional: columna de **ATM** (`atm_id`) para filtrar o agregar

La app agrega por frecuencia (D/W/M) y pronostica `sum(monto)`.

### B) Catálogo de errores (como tu ejemplo) + eventos/logs → pronóstico de frecuencia de errores
- **Archivo principal (catálogo)** con columnas tipo: `ERROR`, `DESCRIPCIÓN`
- **Archivo de eventos/logs** con:
  - columna de **fecha/hora**
  - columna de **código de error**
  - opcional: columna de ATM

La app genera una serie `count(error)` por periodo y permite pronosticar un código específico.

## Modelos incluidos (>=5)
1. Auto-ARIMA (pmdarima)
2. ETS / Holt-Winters
3. Prophet
4. Random Forest con lags + variables calendario
5. XGBoost con lags + variables calendario

## Instalación
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Ejecutar
```bash
streamlit run app.py
```

## Nota sobre CSVs en español
- Se soportan codificaciones frecuentes: utf-8 / utf-8-sig / cp1252 / latin1
- Se intenta inferir delimitador automáticamente (, o ;)
