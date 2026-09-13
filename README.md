# ANTM Stock Price Prediction with XGBoost and Random Forest

Undergraduate thesis project (Informatics, Gunadarma University, 2026) predicting the closing price and price direction of PT Aneka Tambang Tbk (ANTM.JK) on the Indonesia Stock Exchange, ending in a deployed Streamlit web application.

The project follows the CRISP-DM methodology.

---

## Key finding

The headline result of this project is not the error metric — it is what the error metric hid.

Both tuned models reached a MAPE of roughly 2%, which looks strong. But their **Directional Accuracy was below random chance** (43.78% and 45.13%), meaning the models were poor at predicting whether the price would rise or fall. A low MAPE alone was not sufficient evidence that the models were useful.

Diagnostics traced the cause to an upward-direction bias introduced by features expressed as absolute price levels. After detrending those features into relative returns and applying class weighting, Directional Accuracy rose to:

| Model | Directional Accuracy (after correction) |
|---|---|
| XGBoost | 60.81% |
| Random Forest | 62.01% |
| Random-guess baseline | 50.06% |
| Majority-class baseline | 54.20% |

Both corrected models outperform both baselines.

---

## Data

| Source | Content | Notes |
|---|---|---|
| Investing.com | ANTM daily OHLCV and percentage change | Indonesian locale format: `DD/MM/YYYY` dates, `.` thousands separator, `,` decimal separator, volume as text (`80,41M`) |
| yfinance | World gold price, world nickel price, USD/IDR | ISO dates, float values |

- **Period:** 2 March 2012 – 27 February 2026
- **Records:** 3,389 daily observations
- **Split:** chronological, no shuffling — 80% train (2,672 rows) / 20% test (668 rows)

The two sources use different formats and different trading calendars, so a cleaning and alignment step was required before they could be merged into a single feature set.

### Features (10)

`Open`, `High`, `Low`, `Volume`, `SMA-50`, `SMA-200`, `RSI`, world gold price, world nickel price, USD/IDR exchange rate.

---

## Method

- **Models:** XGBoost and Random Forest, in both regression (price) and classification (direction) framings
- **Validation:** `TimeSeriesSplit` cross-validation, chosen over a random split to avoid lookahead bias on time series data
- **Tuning:** `RandomizedSearchCV`
- **Interpretability:** SHAP values and feature importances
- **Evaluation:** MAPE, RMSE, MAE, R², and Directional Accuracy

## Regression results (after tuning)

| Model | MAPE |
|---|---|
| XGBoost | 2.0443% |
| Random Forest | 2.0902% |

Random Forest performed slightly better on MAE, RMSE, R² and Directional Accuracy. The two models are close enough that neither is decisively superior.

---

## Repository contents

| File | Description |
|---|---|
| `streamlit_app.py` | Streamlit web application for interactive prediction |
| `antm_stock_prediction.ipynb` | Full analysis notebook: data preparation, modelling, evaluation, diagnostics |
| `Data_Historis_ANTM.csv` | ANTM daily historical data (Investing.com) |
| `gold_nickel.csv` | World gold and nickel prices (yfinance) |
| `requirements.txt` | Python dependencies |

---

## Running locally

```bash
git clone https://github.com/Muhammadfahrezi044/antm-stock-prediction.git
cd antm-stock-prediction
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Tech stack

Python · pandas · NumPy · SciPy · scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn · Streamlit · yfinance

---

## Disclaimer

This project was built for academic research. It is not financial advice and should not be used as a basis for investment decisions.

---

**Muhammad Fahrezi** — [GitHub](https://github.com/Muhammadfahrezi044)
