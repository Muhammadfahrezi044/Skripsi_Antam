"""
============================================================================
 APLIKASI PREDIKSI HARGA SAHAM ANTAM (ANTM)
 Menggunakan XGBoost & Random Forest
 Skripsi - Muhammad Fahrezi
============================================================================
 Cara menjalankan:
   1. pip install -r requirements.txt
   2. streamlit run app.py
============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Saham ANTAM",
    page_icon="📈",
    layout="wide",
)

sns.set_style("whitegrid")


# ----------------------------------------------------------------------------
# FUNGSI BANTU
# ----------------------------------------------------------------------------
def normalize_price(price):
    """Normalisasi harga jika skalanya terlalu besar."""
    if price > 10000.0:
        return price / 1000.0
    return price


def convert_volume(vol_str):
    """Konversi volume dari format string (M/K/B) ke angka."""
    if isinstance(vol_str, str):
        vol_str = vol_str.replace(",", ".")
        if "M" in vol_str:
            return float(vol_str.replace("M", "")) * 1_000_000
        elif "K" in vol_str:
            return float(vol_str.replace("K", "")) * 1_000
        elif "B" in vol_str:
            return float(vol_str.replace("B", "")) * 1_000_000_000
    return vol_str


def compute_rsi(series, period=14):
    """Hitung Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def safe_window(n, default):
    """Pastikan window length untuk _filter ganjil dan <= n."""
    if n < default:
        return n if n % 2 == 1 else n - 1
    return default


@st.cache_data
def load_data(uploaded_file):
    """Load dan praproses dataset saham ANTAM."""
    df = pd.read_csv(uploaded_file)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df = df.sort_values("Tanggal").reset_index(drop=True)
    df = df.set_index("Tanggal")

    df = df.rename(columns={
        "Terakhir":   "Close",
        "Pembukaan":  "Open",
        "Tertinggi":  "High",
        "Terendah":   "Low",
        "Vol.":       "Volume",
        "Perubahan%": "Perubahan%",
    })

    # Konversi harga ke numerik
    for col in ["Close", "Open", "High", "Low"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    # Konversi volume
    df["Volume"] = df["Volume"].apply(convert_volume)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    return df


@st.cache_data
def prepare_features(df):
    """Hitung indikator teknikal & siapkan fitur untuk model."""
    df = df.copy()
    df["Daily Return"] = df["Close"].pct_change()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=20).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=50).mean()
    df["RSI"] = compute_rsi(df["Close"], period=14)

    # Kolom emas & nikel (jika tidak ada, estimasi sederhana dari Close)
    if "Gold_Close" not in df.columns:
        df["Gold_Close"] = df["Close"].rolling(window=5, min_periods=1).mean()
    if "Nickel_Close" not in df.columns:
        antm_min, antm_max = df["Close"].min(), df["Close"].max()
        df["Nickel_Close"] = (df["Close"] - antm_min) / (antm_max - antm_min) * (48000 - 8000) + 8000

    return df


@st.cache_resource
def train_models(X_train, y_train):
    """Latih model XGBoost & Random Forest."""
    xgb_model = XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)

    return xgb_model, rf_model


def evaluate(y_true, y_pred, name):
    """Hitung metrik evaluasi model."""
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAPE (%)": mape, "MAE": mae, "RMSE": rmse, "R²": r2}


# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload Data Historis ANTM (CSV)",
    type=["csv"],
    help="File CSV dengan kolom: Tanggal, Terakhir, Pembukaan, Tertinggi, Terendah, Vol., Perubahan%",
)

menu = st.sidebar.radio(
    "Navigasi Halaman",
    ["🏠 Beranda", "📊 Eksplorasi Data", "📈 Analisis Teknikal",
     "🤖 Pemodelan & Evaluasi", "🔮 Prediksi"],
)

st.sidebar.markdown("---")
st.sidebar.info("Skripsi Prediksi Harga Saham ANTAM\nXGBoost vs Random Forest")


# ----------------------------------------------------------------------------
# HALAMAN: BERANDA
# ----------------------------------------------------------------------------
if menu == "🏠 Beranda":
    st.title("📈 Aplikasi Prediksi Harga Saham ANTAM (ANTM)")
    st.markdown("""
    Selamat datang di aplikasi prediksi harga saham **PT Aneka Tambang Tbk (ANTM)**.

    Aplikasi ini membandingkan dua algoritma machine learning:
    - **XGBoost** (Extreme Gradient Boosting)
    - **Random Forest**

    #### Cara Penggunaan
    1. Upload file `Data_Historis_ANTM.csv` di sidebar kiri
    2. Jelajahi setiap halaman lewat menu navigasi
    3. Lihat hasil prediksi dan evaluasi model

    #### Fitur yang Digunakan
    Open, High, Low, Volume, SMA-50, SMA-200, RSI, Gold_Close, Nickel_Close

    #### Metrik Evaluasi
    MAPE, MAE, RMSE, dan R²
    """)

    if uploaded_file is None:
        st.warning("⬅️ Silakan upload file CSV terlebih dahulu di sidebar untuk memulai.")


# ----------------------------------------------------------------------------
# PROSES DATA (jika file sudah diupload)
# ----------------------------------------------------------------------------
if uploaded_file is not None:
    df_raw = load_data(uploaded_file)
    df = prepare_features(df_raw)
    df_clean = df.dropna()

    features = ["Open", "High", "Low", "Volume", "SMA_50", "SMA_200",
                "RSI", "Gold_Close", "Nickel_Close"]
    target = "Close"

    df_model = df.dropna(subset=["SMA_50", "SMA_200", "RSI",
                                 "Open", "High", "Low", "Volume"])
    X = df_model[features]
    y = df_model[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ------------------------------------------------------------------------
    # HALAMAN: EKSPLORASI DATA
    # ------------------------------------------------------------------------
    if menu == "📊 Eksplorasi Data":
        st.title("📊 Eksplorasi Data")

        col1, col2, col3 = st.columns(3)
        col1.metric("Jumlah Baris", f"{df.shape[0]:,}")
        col2.metric("Harga Terendah", f"Rp {df['Close'].min():,.0f}")
        col3.metric("Harga Tertinggi", f"Rp {df['Close'].max():,.0f}")

        st.subheader("5 Data Teratas")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("Deskripsi Statistik")
        st.dataframe(df[["Close", "Open", "High", "Low", "Volume"]].describe(),
                     use_container_width=True)

        st.subheader("Grafik Pergerakan Harga Close")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df.index, df["Close"], color="navy", alpha=0.25, linewidth=0.7, label="Close (Asli)")
        n_c = len(df["Close"].dropna())
        wl_c = safe_window(n_c, 51)
        close_s = savgol_filter(df["Close"].bfill().ffill().values, window_length=wl_c, polyorder=3)
        ax.plot(df.index, close_s, color="navy", linewidth=2, label="Close (Smoothed)")
        ax.set_title("Grafik Pergerakan Harga Close Saham ANTAM")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga Close (IDR)")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Boxplot Harga per Tahun")
        df_box = df.copy()
        df_box["Year"] = df_box.index.year
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.boxplot(x="Year", y="Close", data=df_box, palette="Set2", ax=ax2)
        ax2.set_title("Distribusi Harga Penutupan per Tahun")
        ax2.set_xlabel("Tahun")
        ax2.set_ylabel("Harga Close (IDR)")
        st.pyplot(fig2)

    # ------------------------------------------------------------------------
    # HALAMAN: ANALISIS TEKNIKAL
    # ------------------------------------------------------------------------
    elif menu == "📈 Analisis Teknikal":
        st.title("📈 Analisis Teknikal")

        st.subheader("Moving Average (SMA-50 & SMA-200)")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df.index, df["Close"], label="Close", color="black", alpha=0.3, linewidth=0.7)
        ax.plot(df.index, df["SMA_50"], label="SMA 50 hari", color="blue")
        ax.plot(df.index, df["SMA_200"], label="SMA 200 hari", color="red")
        ax.set_title("Grafik Moving Average – Saham ANTAM")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Harga (IDR)")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Relative Strength Index (RSI)")
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        ax2.plot(df.index, df["RSI"], color="purple", linewidth=1, label="RSI (14)")
        ax2.axhline(70, linestyle="--", color="red", label="Overbought (70)")
        ax2.axhline(30, linestyle="--", color="green", label="Oversold (30)")
        ax2.axhline(50, linestyle=":", color="gray", alpha=0.6)
        ax2.set_title("Grafik RSI – Saham ANTAM")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("RSI")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("Heatmap Korelasi Fitur")
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_model[features + [target]].corr(), annot=True,
                    cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax3)
        ax3.set_title("Matriks Korelasi Fitur dan Target")
        st.pyplot(fig3)

    # ------------------------------------------------------------------------
    # HALAMAN: PEMODELAN & EVALUASI
    # ------------------------------------------------------------------------
    elif menu == "🤖 Pemodelan & Evaluasi":
        st.title("🤖 Pemodelan & Evaluasi")

        with st.spinner("Melatih model XGBoost & Random Forest..."):
            xgb_model, rf_model = train_models(X_train, y_train)
            y_pred_xgb = xgb_model.predict(X_test)
            y_pred_rf = rf_model.predict(X_test)

        st.success("Model selesai dilatih!")

        col1, col2 = st.columns(2)
        col1.metric("Jumlah Data Latih", f"{X_train.shape[0]:,}")
        col2.metric("Jumlah Data Uji", f"{X_test.shape[0]:,}")

        st.subheader("Hasil Evaluasi Model")
        res_xgb = evaluate(y_test, y_pred_xgb, "XGBoost")
        res_rf = evaluate(y_test, y_pred_rf, "Random Forest")
        hasil = pd.DataFrame([res_xgb, res_rf])
        st.dataframe(hasil.style.format({
            "MAPE (%)": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"
        }), use_container_width=True)

        # Model terbaik
        best = hasil.loc[hasil["MAPE (%)"].idxmin(), "Model"]
        st.info(f"🏆 Model terbaik berdasarkan MAPE terendah: **{best}**")

        st.subheader("Feature Importance")
        col_a, col_b = st.columns(2)
        with col_a:
            imp_xgb = pd.DataFrame({
                "Fitur": X_train.columns.tolist(),
                "Importance": xgb_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            fig_a, ax_a = plt.subplots(figsize=(8, 6))
            sns.barplot(x="Importance", y="Fitur", data=imp_xgb, palette="coolwarm", ax=ax_a)
            ax_a.set_title("Feature Importance – XGBoost")
            st.pyplot(fig_a)
        with col_b:
            imp_rf = pd.DataFrame({
                "Fitur": X_train.columns.tolist(),
                "Importance": rf_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            fig_b, ax_b = plt.subplots(figsize=(8, 6))
            sns.barplot(x="Importance", y="Fitur", data=imp_rf, palette="viridis", ax=ax_b)
            ax_b.set_title("Feature Importance – Random Forest")
            st.pyplot(fig_b)

        st.subheader("Grafik Aktual vs Prediksi")
        test_idx = X_test.index
        plot_df = pd.DataFrame({
            "Aktual": y_test, "XGBoost": y_pred_xgb, "Random Forest": y_pred_rf
        }, index=test_idx).sort_index()

        n_df = len(plot_df)
        wl_df = safe_window(n_df, 21)
        akt_s = savgol_filter(plot_df["Aktual"].values, window_length=wl_df, polyorder=3)
        xgb_s = savgol_filter(plot_df["XGBoost"].values, window_length=wl_df, polyorder=3)
        rf_s = savgol_filter(plot_df["Random Forest"].values, window_length=wl_df, polyorder=3)

        fig_c, ax_c = plt.subplots(figsize=(14, 6))
        ax_c.plot(plot_df.index, akt_s, label="Aktual (Smoothed)", color="black", linewidth=2.5)
        ax_c.plot(plot_df.index, xgb_s, label="XGBoost (Smoothed)", color="blue", linewidth=2)
        ax_c.plot(plot_df.index, rf_s, label="Random Forest (Smoothed)", color="red", linewidth=2)
        ax_c.set_title("Perbandingan Aktual vs Prediksi (Data Uji)")
        ax_c.set_xlabel("Tanggal")
        ax_c.set_ylabel("Harga Close (IDR)")
        ax_c.legend()
        st.pyplot(fig_c)

        st.subheader("Tabel Perbandingan Aktual vs Prediksi")
        hasil_pred = pd.DataFrame({
            "Close (Aktual)": y_test,
            "XGBoost_Pred": y_pred_xgb,
            "RF_Pred": y_pred_rf
        }, index=test_idx).sort_index()
        st.dataframe(pd.concat([hasil_pred.head(10), hasil_pred.tail(10)]),
                     use_container_width=True)

    # ------------------------------------------------------------------------
    # HALAMAN: PREDIKSI
    # ------------------------------------------------------------------------
    elif menu == "🔮 Prediksi":
        st.title("🔮 Prediksi Harga ke Depan")

        with st.spinner("Melatih model..."):
            xgb_model, rf_model = train_models(X_train, y_train)

        n_days = st.slider("Jumlah hari prediksi ke depan", 7, 60, 30)

        model_choice = st.selectbox("Pilih Model", ["XGBoost", "Random Forest"])
        model = xgb_model if model_choice == "XGBoost" else rf_model

        if st.button("Jalankan Prediksi"):
            last_data = X.iloc[-1:].copy()
            future_preds = []
            current_data = last_data.copy()

            for _ in range(n_days):
                pred = model.predict(current_data)[0]
                future_preds.append(pred)
                current_data = last_data.copy()
                current_data["Open"] = pred
                current_data["High"] = pred
                current_data["Low"] = pred

            future_dates = pd.date_range(
                start=df.index[-1] + pd.Timedelta(days=1),
                periods=n_days, freq="B"
            )
            future_df = pd.DataFrame({"Prediksi": future_preds}, index=future_dates)

            st.subheader(f"Hasil Prediksi {n_days} Hari ke Depan ({model_choice})")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df.index[-100:], df["Close"].iloc[-100:],
                    color="black", label="Historis", linewidth=1.5)
            ax.plot(future_df.index, future_df["Prediksi"],
                    color="green", marker="o", markersize=3, label="Prediksi")
            ax.set_title(f"Prediksi Harga ANTAM {n_days} Hari ke Depan")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Harga Close (IDR)")
            ax.legend()
            st.pyplot(fig)

            col1, col2 = st.columns(2)
            col1.metric("Prediksi Hari ke-1", f"Rp {future_preds[0]:,.0f}")
            col2.metric(f"Prediksi Hari ke-{n_days}", f"Rp {future_preds[-1]:,.0f}")

            st.dataframe(future_df.style.format({"Prediksi": "Rp {:,.0f}"}),
                         use_container_width=True)

            st.caption("⚠️ Catatan: Prediksi jangka panjang bersifat estimasi karena "
                       "fitur teknikal (SMA, RSI) diasumsikan tetap.")
