"""
============================================================================
 APLIKASI PREDIKSI HARGA SAHAM ANTAM (ANTM)
 XGBoost & Random Forest  -  Skripsi Muhammad Fahrezi
============================================================================
 Disamakan PERSIS dengan notebook:
   - Konversi harga identik dengan notebook (sudah diverifikasi 0 selisih)
   - Data emas (GC=F) & nikel (NI=F / INCO.JK) diunduh asli via yfinance
   - Semua grafik mengikuti notebook
   - Mendukung file .csv maupun .xlsx (termasuk xlsx berisi CSV mentah)

 Jalankan:
   pip install -r requirements.txt
   streamlit run app.py
============================================================================
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import seaborn as sns
from scipy.signal import savgol_filter

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_percentage_error, mean_absolute_error,
    mean_squared_error, r2_score,
)

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Prediksi Harga Saham ANTAM", page_icon="📈", layout="wide")
sns.set_style("whitegrid")


# ============================================================================
# FUNGSI PRAPROSES
# ============================================================================
def convert_volume(vol_str):
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
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def safe_window(n, default, polyorder=3):
    """Pastikan window ganjil, <= n, dan > polyorder (syarat savgol_filter)."""
    w = default if n >= default else (n if n % 2 == 1 else n - 1)
    if w <= polyorder:           # window harus lebih besar dari polyorder
        w = polyorder + 1
    if w % 2 == 0:               # pastikan ganjil
        w += 1
    return w


def _read_any(file_bytes, file_name):
    """Baca CSV/XLSX, termasuk XLSX yang isinya CSV mentah (1 kolom)."""
    expected = ["Tanggal", "Terakhir", "Pembukaan", "Tertinggi",
                "Terendah", "Vol.", "Perubahan%"]
    name = file_name.lower()

    if name.endswith((".xlsx", ".xls")):
        raw = pd.read_excel(io.BytesIO(file_bytes), header=None)
    else:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
            if "Tanggal" in df.columns and "Terakhir" in df.columns:
                return df
        except Exception:
            pass
        raw = pd.read_csv(io.BytesIO(file_bytes), header=None)

    # Sudah rapi (>=7 kolom)
    if raw.shape[1] >= 7:
        if str(raw.iloc[0, 1]).strip().strip('"') in ("Terakhir", "Pembukaan"):
            raw = raw.iloc[1:].reset_index(drop=True)
        raw.columns = expected[:raw.shape[1]]
        return raw

    # File 1 kolom: CSV mentah tergabung
    col0 = raw.iloc[:, 0].astype(str).str.replace('"', "", regex=False)
    col0 = col0[~col0.str.startswith("Tanggal")].reset_index(drop=True)
    rows = []
    for s in col0:
        toks = s.split(",")
        if len(toks) < 9:
            continue
        tgl = toks[0]
        close, open_, high, low = toks[1], toks[2], toks[3], toks[4]
        vol = toks[5] + "." + toks[6]
        chg = toks[7] + "." + toks[8]
        rows.append([tgl, close, open_, high, low, vol, chg])
    return pd.DataFrame(rows, columns=expected)


@st.cache_data(show_spinner=False)
def load_and_process(file_bytes, file_name, fetch_online=True):
    df = _read_any(file_bytes, file_name)

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df = df.sort_values("Tanggal").reset_index(drop=True).set_index("Tanggal")
    df = df.rename(columns={
        "Terakhir": "Close", "Pembukaan": "Open", "Tertinggi": "High",
        "Terendah": "Low", "Vol.": "Volume", "Perubahan%": "Perubahan%",
    })

    # Konversi harga (verified identik dengan notebook)
    for col in ["Close", "Open", "High", "Low"]:
        s = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(s, errors="coerce")

    df["Volume"] = df["Volume"].apply(convert_volume)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    start_date = str(df.index.min().date())
    end_date = str(df.index.max().date())

    log, gold, nickel = [], None, None
    if fetch_online:
        try:
            import yfinance as yf
            g = yf.download("GC=F", start=start_date, end=end_date, progress=False)
            if isinstance(g.columns, pd.MultiIndex):
                g.columns = ["_".join(c).strip() for c in g.columns]
                cc = [c for c in g.columns if "Close" in c][0]
            else:
                cc = "Close"
            gold = g[[cc]].rename(columns={cc: "Gold_Close"})
            gold.index = pd.to_datetime(gold.index)
            log.append(f"Emas (GC=F): {len(gold)} baris")

            for tk in ["NI=F", "INCO.JK"]:
                try:
                    nr = yf.download(tk, start=start_date, end=end_date, progress=False)
                    if isinstance(nr.columns, pd.MultiIndex):
                        nr.columns = ["_".join(c).strip() for c in nr.columns]
                        c2 = [c for c in nr.columns if "Close" in c][0]
                    else:
                        c2 = "Close"
                    nr = nr[[c2]].rename(columns={c2: "Nickel_Close_Raw"})
                    if len(nr) > 100 and nr["Nickel_Close_Raw"].isna().mean() < 0.5:
                        nickel = nr.copy()
                        log.append(f"Nikel ({tk}): {len(nickel)} baris")
                        break
                except Exception:
                    continue
        except Exception as e:
            log.append(f"Gagal yfinance: {e}")

    if gold is None:
        gold = pd.DataFrame({"Gold_Close": df["Close"].rolling(5, min_periods=1).mean()})
        gold.index = df.index
        log.append("Emas: fallback estimasi (offline)")

    if nickel is None:
        amin, amax = df["Close"].min(), df["Close"].max()
        ns = (df["Close"] - amin) / (amax - amin) * (48000 - 8000) + 8000
        nickel = pd.DataFrame({"Nickel_Close_Raw": ns})
        nickel.index = df.index
        log.append("Nikel: fallback estimasi (offline)")

    nmean = nickel["Nickel_Close_Raw"].mean()
    nickel["Nickel_Close"] = nickel["Nickel_Close_Raw"] / 10 if nmean > 10000 else nickel["Nickel_Close_Raw"]

    nickel.index = pd.to_datetime(nickel.index)
    df = df.join(gold[["Gold_Close"]], how="left")
    df = df.join(nickel[["Nickel_Close"]], how="left")
    df["Gold_Close"] = df["Gold_Close"].ffill().bfill()
    df["Nickel_Close"] = df["Nickel_Close"].ffill().bfill()
    return df, log


def add_indicators(df):
    df = df.copy()
    df = df.dropna(subset=["Close", "Open", "High", "Low", "Volume"])
    df["Daily Return"] = df["Close"].pct_change()
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=20).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=50).mean()
    df["RSI"] = compute_rsi(df["Close"], period=14)
    return df


@st.cache_resource(show_spinner=False)
def train_models(X_vals, y_vals, cols):
    X_train = pd.DataFrame(X_vals, columns=cols)
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_vals)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_vals)
    return xgb, rf


def evaluate(y_true, y_pred, name):
    return {
        "Model": name,
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
    }


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload Data ANTM (CSV / XLSX)", type=["csv", "xlsx", "xls"])
fetch_online = st.sidebar.checkbox("Unduh emas & nikel via yfinance", value=True,
                                   help="Matikan jika tanpa internet (pakai estimasi).")
menu = st.sidebar.radio("Navigasi", ["🏠 Beranda", "📊 Eksplorasi Data",
                                     "📈 Analisis Teknikal", "🤖 Pemodelan & Evaluasi", "🔮 Prediksi"])
st.sidebar.markdown("---")
st.sidebar.info("Skripsi Prediksi Saham ANTAM\nXGBoost vs Random Forest")


# ============================================================================
# BERANDA
# ============================================================================
if menu == "🏠 Beranda":
    st.title("📈 Prediksi Harga Saham ANTAM (ANTM)")
    st.markdown("""
    Aplikasi ini disamakan dengan notebook skripsi:
    - Konversi harga identik dengan notebook
    - Data **emas (GC=F)** & **nikel (NI=F / INCO.JK)** diunduh asli via yfinance
    - Semua grafik mengikuti notebook

    Upload `Data_Historis_ANTM.csv` atau `.xlsx` di sidebar untuk memulai.
    """)
    if uploaded_file is None:
        st.warning("⬅️ Upload file terlebih dahulu.")


if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    with st.spinner("Memuat & memproses data (mengunduh emas/nikel bila online)..."):
        df, log = load_and_process(file_bytes, uploaded_file.name, fetch_online)
        df = add_indicators(df)

    features = ["Open", "High", "Low", "Volume", "SMA_50", "SMA_200",
                "RSI", "Gold_Close", "Nickel_Close"]
    target = "Close"
    df_model = df.dropna(subset=["SMA_50", "SMA_200", "RSI"])
    X, y = df_model[features], df_model[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # --------------------------------------------------------------------
    if menu == "📊 Eksplorasi Data":
        st.title("📊 Eksplorasi Data")
        for l in log:
            st.caption("• " + l)
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Baris", f"{df.shape[0]:,}")
        c2.metric("Harga Terendah", f"Rp {df['Close'].min():,.0f}")
        c3.metric("Harga Tertinggi", f"Rp {df['Close'].max():,.0f}")

        st.subheader("2.1 Data Teratas")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("2.2 Data Terbawah")
        st.dataframe(df.tail(), use_container_width=True)

        st.subheader("2.3 Informasi Dataset")
        info = pd.DataFrame({
            "Kolom": df.columns,
            "Non-Null": [df[c].notna().sum() for c in df.columns],
            "Tipe Data": [str(df[c].dtype) for c in df.columns],
        })
        st.dataframe(info, use_container_width=True)
        st.caption(f"Total {df.shape[0]:,} baris dan {df.shape[1]} kolom.")

        st.subheader("2.4 Deskripsi Statistik")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("2.5 Cek Missing Values")
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Kolom", "Jumlah Missing"]
        st.dataframe(miss, use_container_width=True)

        st.subheader("2.6 Grafik Pergerakan Harga Close")
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df.index, df["Close"], color="navy", alpha=0.25, linewidth=0.7, label="Close (Asli)")
        wl_c = safe_window(len(df["Close"]), 51)
        ax.plot(df.index, savgol_filter(df["Close"].values, wl_c, 3), color="navy", linewidth=2, label="Close (Smoothed)")
        ax.set_title("Grafik Pergerakan Harga Close Saham ANTAM")
        ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Close (IDR)"); ax.legend()
        st.pyplot(fig)

        st.subheader("2.7 Volume Perdagangan")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df.index, df["Volume"], color="darkorange", alpha=0.25, linewidth=0.7, label="Volume Asli")
        ax.plot(df.index, df["Volume"].rolling(30, center=True, min_periods=1).mean(),
                color="darkorange", linewidth=2, label="Volume (MA-30)")
        ax.set_title("Volume Perdagangan Saham ANTAM"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Volume")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}"))
        ax.legend(); st.pyplot(fig)

        st.subheader("2.8 Candlestick (Setahun Terakhir)")
        cutoff = df.index.max() - pd.Timedelta(days=365)
        last_year = df[df.index >= cutoff].copy()
        fig, ax = plt.subplots(figsize=(14, 6))
        w = 0.6
        for date, row in last_year.iterrows():
            color = "green" if row["Close"] >= row["Open"] else "red"
            ax.vlines(date, row["Low"], row["High"], color=color, linewidth=1)
            bl, bh = min(row["Open"], row["Close"]), max(row["Open"], row["Close"])
            ax.add_patch(plt.Rectangle((date - pd.Timedelta(days=w/2), bl),
                                       pd.Timedelta(days=w), bh - bl, facecolor=color, edgecolor=color))
        ax.legend(handles=[Patch(facecolor="green", label="Naik"), Patch(facecolor="red", label="Turun")], loc="upper left")
        ax.set_title("Candlestick ANTAM (Setahun Terakhir)"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (IDR)")
        plt.xticks(rotation=45); st.pyplot(fig)

        st.subheader("2.9 Boxplot Harga per Tahun")
        df_box = df.copy(); df_box["Year"] = df_box.index.year
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(x="Year", y="Close", data=df_box, hue="Year", palette="Set2", legend=False, ax=ax)
        ax.set_title("Boxplot Harga Penutupan per Tahun"); ax.set_xlabel("Tahun"); ax.set_ylabel("Harga Close (IDR)")
        st.pyplot(fig)

        st.subheader("2.10 Koefisien Variasi (CV) per Tahun")
        cv_per_year = df_box.groupby("Year")["Close"].agg(["mean", "std"])
        cv_per_year["CV in %"] = (cv_per_year["std"] / cv_per_year["mean"]) * 100
        st.dataframe(cv_per_year.style.format({"mean": "{:.2f}", "std": "{:.2f}", "CV in %": "{:.2f}"}),
                     use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(cv_per_year.index.astype(str), cv_per_year["CV in %"], color="steelblue", edgecolor="black")
        for i, v in enumerate(cv_per_year["CV in %"]):
            ax.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=9)
        ax.set_title("Koefisien Variasi (CV) Harga Saham ANTM per Tahun")
        ax.set_xlabel("Tahun"); ax.set_ylabel("CV (%)")
        st.pyplot(fig)

        st.subheader("2.11 Matriks Korelasi (Mean, Std, CV)")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cv_per_year.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Matriks Korelasi (Mean, Std, CV)")
        st.pyplot(fig)

    # --------------------------------------------------------------------
    elif menu == "📈 Analisis Teknikal":
        st.title("📈 Analisis Teknikal")

        st.subheader("3.3 Grafik Daily Return")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df.index, df["Daily Return"], color="teal", linewidth=0.5, alpha=0.25, label="Daily Return Asli")
        ax.plot(df.index, df["Daily Return"].rolling(10, center=True, min_periods=1).mean(),
                color="teal", linewidth=2, label="Daily Return (MA-10)")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title("Daily Return Saham ANTAM"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Return Harian")
        ax.legend(); st.pyplot(fig)

        st.subheader("3.4 Distribusi Daily Return")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df["Daily Return"].dropna(), bins=60, kde=True, color="teal", ax=ax)
        ax.set_title("Distribusi Daily Return Saham ANTAM")
        ax.set_xlabel("Daily Return"); ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

        st.subheader("3.5 Moving Average (SMA-50 & SMA-200)")
        fig, ax = plt.subplots(figsize=(14, 6))
        wl_c = safe_window(len(df["Close"].dropna()), 51)
        ax.plot(df.index, df["Close"], label="Close (Asli)", color="black", alpha=0.2, linewidth=0.7)
        ax.plot(df.index, savgol_filter(df["Close"].values, wl_c, 3), label="Close (Smoothed)", color="black", linewidth=1.8)
        ax.plot(df.index, df["SMA_50"], label="SMA 50 hari", color="blue")
        ax.plot(df.index, df["SMA_200"], label="SMA 200 hari", color="red")
        ax.set_title("Grafik Moving Average – Saham ANTAM"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (IDR)")
        ax.legend(); st.pyplot(fig)

        st.subheader("3.6 Relative Strength Index (RSI)")
        wl_r = safe_window(len(df["RSI"].dropna()), 21)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df.index, df["RSI"], color="purple", linewidth=0.5, alpha=0.25, label="RSI (14) Asli")
        ax.plot(df.index, savgol_filter(df["RSI"].fillna(50).values, wl_r, 3), color="purple", linewidth=2, label="RSI (14) Smoothed")
        ax.axhline(70, linestyle="--", color="red", label="Overbought (70)")
        ax.axhline(30, linestyle="--", color="green", label="Oversold (30)")
        ax.axhline(50, linestyle=":", color="gray", alpha=0.6)
        ax.set_title("Grafik RSI – Saham ANTAM"); ax.set_xlabel("Tanggal"); ax.set_ylabel("RSI")
        ax.legend(); st.pyplot(fig)

        st.subheader("3.7.1 Korelasi ANTAM vs Emas & Nikel")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df_model[["Close", "Gold_Close", "Nickel_Close"]].corr(),
                    annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Korelasi ANTAM vs Emas & Nikel"); st.pyplot(fig)

        st.subheader("3.7.2 Tren ANTAM, Emas, Nikel (Base = 100)")
        dt = df_model[["Close", "Gold_Close", "Nickel_Close"]].copy()
        dt = dt / dt.iloc[0] * 100
        wl_t = safe_window(len(dt), 51)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(dt.index, savgol_filter(dt["Close"].values, wl_t, 3), label="ANTAM", color="navy", linewidth=2)
        ax.plot(dt.index, savgol_filter(dt["Gold_Close"].values, wl_t, 3), label="Emas", color="goldenrod", linewidth=2)
        ax.plot(dt.index, savgol_filter(dt["Nickel_Close"].values, wl_t, 3), label="Nikel", color="green", linewidth=2)
        ax.set_title("Perbandingan Tren (Base 100)"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Ternormalisasi")
        ax.legend(); st.pyplot(fig)

        st.subheader("3.8 Heatmap Korelasi Fitur")
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(df_model[features + [target]].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Matriks Korelasi Fitur dan Target"); st.pyplot(fig)

    # --------------------------------------------------------------------
    elif menu == "🤖 Pemodelan & Evaluasi":
        st.title("🤖 Pemodelan & Evaluasi")
        with st.spinner("Melatih model..."):
            xgb_model, rf_model = train_models(X_train.values, y_train.values, features)
            y_pred_xgb = xgb_model.predict(X_test)
            y_pred_rf = rf_model.predict(X_test)
        st.success("Model selesai dilatih!")

        c1, c2 = st.columns(2)
        c1.metric("Data Latih", f"{X_train.shape[0]:,}")
        c2.metric("Data Uji", f"{X_test.shape[0]:,}")

        st.subheader("Hasil Evaluasi")
        hasil = pd.DataFrame([evaluate(y_test, y_pred_xgb, "XGBoost"),
                              evaluate(y_test, y_pred_rf, "Random Forest")])
        st.dataframe(hasil.style.format({"MAPE (%)": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"}),
                     use_container_width=True)
        best = hasil.loc[hasil["MAPE (%)"].idxmin(), "Model"]
        st.info(f"🏆 Model terbaik (MAPE terendah): **{best}**")

        st.subheader("Feature Importance")
        ca, cb = st.columns(2)
        with ca:
            imp = pd.DataFrame({"Fitur": features, "Importance": xgb_model.feature_importances_}).sort_values("Importance", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x="Importance", y="Fitur", data=imp, hue="Fitur", palette="coolwarm", legend=False, ax=ax)
            ax.set_title("XGBoost"); st.pyplot(fig)
        with cb:
            imp = pd.DataFrame({"Fitur": features, "Importance": rf_model.feature_importances_}).sort_values("Importance", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x="Importance", y="Fitur", data=imp, hue="Fitur", palette="viridis", legend=False, ax=ax)
            ax.set_title("Random Forest"); st.pyplot(fig)

        st.subheader("4.4 Visualisasi Pohon Random Forest")
        with st.spinner("Merender pohon keputusan (agak lambat)..."):
            from sklearn.tree import plot_tree
            rf_vis = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
            rf_vis.fit(X_train, y_train)
            fig, ax = plt.subplots(figsize=(22, 10))
            plot_tree(rf_vis.estimators_[0], feature_names=features,
                      filled=True, rounded=True, fontsize=9, ax=ax)
            ax.set_title("Visualisasi Random Forest Regressor (max_depth=3) – Pohon Pertama")
            st.pyplot(fig)

        st.subheader("4.6 Tabel Perbandingan Aktual vs Prediksi")
        idx = X_test.index
        hasil_pred = pd.DataFrame({
            "Close (Aktual)": y_test, "XGBoost_Pred": y_pred_xgb, "RF_Pred": y_pred_rf
        }, index=idx).sort_index()
        st.dataframe(pd.concat([hasil_pred.head(5), hasil_pred.tail(5)]).style.format("{:,.2f}"),
                     use_container_width=True)

        st.subheader("5.1 Aktual vs Prediksi – XGBoost")
        idx = X_test.index
        wl_p = safe_window(len(y_test), 21)
        p_xgb = pd.DataFrame({"Aktual": y_test, "Prediksi": y_pred_xgb}, index=idx).sort_index()
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(p_xgb.index, p_xgb["Aktual"], color="black", alpha=0.2, linewidth=0.7)
        ax.plot(p_xgb.index, p_xgb["Prediksi"], color="blue", alpha=0.2, linewidth=0.7)
        ax.plot(p_xgb.index, savgol_filter(p_xgb["Aktual"].values, wl_p, 3), label="Aktual (Smoothed)", color="black", linewidth=2.5)
        ax.plot(p_xgb.index, savgol_filter(p_xgb["Prediksi"].values, wl_p, 3), label="Prediksi XGBoost (Smoothed)", color="blue", linewidth=2)
        ax.set_title("Aktual vs Prediksi – XGBoost (Data Uji)"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Close (IDR)")
        ax.legend(); st.pyplot(fig)

        st.subheader("5.2 Aktual vs Prediksi – Random Forest")
        p_rf = pd.DataFrame({"Aktual": y_test, "Prediksi": y_pred_rf}, index=idx).sort_index()
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(p_rf.index, p_rf["Aktual"], color="black", alpha=0.2, linewidth=0.7)
        ax.plot(p_rf.index, p_rf["Prediksi"], color="red", alpha=0.2, linewidth=0.7)
        ax.plot(p_rf.index, savgol_filter(p_rf["Aktual"].values, wl_p, 3), label="Aktual (Smoothed)", color="black", linewidth=2.5)
        ax.plot(p_rf.index, savgol_filter(p_rf["Prediksi"].values, wl_p, 3), label="Prediksi Random Forest (Smoothed)", color="red", linewidth=2)
        ax.set_title("Aktual vs Prediksi – Random Forest (Data Uji)"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Close (IDR)")
        ax.legend(); st.pyplot(fig)

        st.subheader("5.3 Aktual vs Prediksi (Gabungan)")
        pdf = pd.DataFrame({"Aktual": y_test, "XGBoost": y_pred_xgb, "Random Forest": y_pred_rf}, index=idx).sort_index()
        wl = safe_window(len(pdf), 21)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(pdf.index, savgol_filter(pdf["Aktual"].values, wl, 3), label="Aktual (Smoothed)", color="black", linewidth=2.5)
        ax.plot(pdf.index, savgol_filter(pdf["XGBoost"].values, wl, 3), label="XGBoost (Smoothed)", color="blue", linewidth=2)
        ax.plot(pdf.index, savgol_filter(pdf["Random Forest"].values, wl, 3), label="Random Forest (Smoothed)", color="red", linewidth=2)
        ax.set_title("Aktual vs Prediksi (Data Uji)"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Close (IDR)")
        ax.legend(); st.pyplot(fig)

        st.subheader("5.4 Scatter Plot Aktual vs Prediksi")
        ca, cb = st.columns(2)
        with ca:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_test, y_pred_xgb, alpha=0.4, color="blue", s=12)
            lims = [min(y_test.min(), y_pred_xgb.min()), max(y_test.max(), y_pred_xgb.max())]
            ax.plot(lims, lims, "--", color="black", linewidth=1)
            ax.set_title("Scatter – XGBoost"); ax.set_xlabel("Aktual"); ax.set_ylabel("Prediksi")
            st.pyplot(fig)
        with cb:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_test, y_pred_rf, alpha=0.4, color="red", s=12)
            lims = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
            ax.plot(lims, lims, "--", color="black", linewidth=1)
            ax.set_title("Scatter – Random Forest"); ax.set_xlabel("Aktual"); ax.set_ylabel("Prediksi")
            st.pyplot(fig)

        st.subheader("5.5 Distribusi Residual / Error")
        res_xgb = y_test.values - y_pred_xgb
        res_rf = y_test.values - y_pred_rf
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.histplot(res_xgb, bins=40, kde=True, color="blue", alpha=0.5, label="XGBoost", ax=ax)
        sns.histplot(res_rf, bins=40, kde=True, color="red", alpha=0.5, label="Random Forest", ax=ax)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Distribusi Residual (Error) Kedua Model")
        ax.set_xlabel("Residual (Aktual - Prediksi)"); ax.set_ylabel("Frekuensi")
        ax.legend(); st.pyplot(fig)

        st.subheader("5.6 Perbandingan MAPE")
        fig, ax = plt.subplots(figsize=(8, 5))
        mape_vals = [hasil.loc[0, "MAPE (%)"], hasil.loc[1, "MAPE (%)"]]
        bars = ax.bar(["XGBoost", "Random Forest"], mape_vals, color=["blue", "red"], edgecolor="black")
        for b, v in zip(bars, mape_vals):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}%", ha="center", va="bottom", fontsize=10)
        ax.set_title("Perbandingan MAPE Kedua Model"); ax.set_ylabel("MAPE (%)")
        st.pyplot(fig)

    # --------------------------------------------------------------------
    elif menu == "🔮 Prediksi":
        st.title("🔮 Prediksi Harga ke Depan")
        with st.spinner("Melatih model..."):
            xgb_model, rf_model = train_models(X_train.values, y_train.values, features)
        n_days = st.slider("Jumlah hari prediksi", 7, 60, 30)
        model_choice = st.selectbox("Pilih Model", ["XGBoost", "Random Forest"])
        model = xgb_model if model_choice == "XGBoost" else rf_model

        if st.button("Jalankan Prediksi"):
            last_data = X.iloc[-1:].copy()
            preds, current = [], last_data.copy()
            for _ in range(n_days):
                p = model.predict(current)[0]
                preds.append(p)
                current = last_data.copy()
                current["Open"] = p; current["High"] = p; current["Low"] = p
            fdates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days, freq="B")
            fdf = pd.DataFrame({"Prediksi": preds}, index=fdates)

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df.index[-100:], df["Close"].iloc[-100:], color="black", label="Historis", linewidth=1.5)
            ax.plot(fdf.index, fdf["Prediksi"], color="green", marker="o", markersize=3, label="Prediksi")
            ax.set_title(f"Prediksi {n_days} Hari ke Depan ({model_choice})")
            ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Close (IDR)"); ax.legend()
            st.pyplot(fig)

            c1, c2 = st.columns(2)
            c1.metric("Hari ke-1", f"Rp {preds[0]:,.0f}")
            c2.metric(f"Hari ke-{n_days}", f"Rp {preds[-1]:,.0f}")
            st.dataframe(fdf.style.format({"Prediksi": "Rp {:,.0f}"}), use_container_width=True)
            st.caption("⚠️ Prediksi jangka panjang bersifat estimasi (SMA, RSI diasumsikan tetap).")
