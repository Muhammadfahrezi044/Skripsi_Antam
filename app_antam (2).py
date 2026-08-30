"""
============================================================================
 APLIKASI PREDIKSI HARGA SAHAM ANTAM (ANTM)
 XGBoost & Random Forest  -  Skripsi Muhammad Fahrezi
============================================================================
 Disamakan PERSIS dengan notebook FINAL (setelah seluruh revisi dosen
 penguji), mencakup:
   - 10 fitur: Open, High, Low, Volume, SMA_50, SMA_200, RSI,
     Gold_Close, Nickel_Close, USDIDR_Close
   - Target = return harian (bukan harga langsung), dikonversi balik ke
     harga memakai Harga_prediksi(t+1) = Harga_aktual(t) x (1 + return)
   - Sumber data nikel = harga komoditas nikel dunia (NI=F) atau file
     manual (investing.com dsb.) -- TIDAK memakai proksi saham
     perusahaan tambang (mis. INCO.JK), karena itu tidak valid secara
     konseptual
   - Hyperparameter tuning (RandomizedSearchCV + TimeSeriesSplit)
   - 4 metrik evaluasi (MAPE, MAE, RMSE, R2) + Directional Accuracy
   - Validasi silang time-series (k-fold walk-forward)
   - Prediksi 30 hari ke depan dengan SMA/RSI dihitung ulang tiap
     iterasi, dan Gold/Nickel/USDIDR diekstrapolasi memakai naive drift
     (bukan dibekukan konstan)
   - Deduplikasi baris tanggal ganda pada data sumber

 Jalankan:
   pip install -r requirements.txt
   streamlit run app_antam.py
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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
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

FEATURES = ["Open", "High", "Low", "Volume", "SMA_50", "SMA_200", "RSI",
            "Gold_Close", "Nickel_Close", "USDIDR_Close"]
FEATURE_DESC = [
    "Harga pembukaan saham", "Harga tertinggi saham", "Harga terendah saham",
    "Volume perdagangan", "Rata-rata bergerak 50 hari", "Rata-rata bergerak 200 hari",
    "Relative Strength Index", "Harga penutupan emas dunia",
    "Harga penutupan nikel dunia", "Nilai tukar Rupiah terhadap Dolar AS",
]


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


def parse_price_string(series):
    """Parser angka yang tahan terhadap dua format penulisan:
    - Format AS/Internasional : 16,050.00  (koma=ribuan, titik=desimal)
    - Format Indonesia/Eropa  : 16.050,00  (titik=ribuan, koma=desimal)
    Tanpa penanganan ini, angka format Indonesia (umum pada file yang
    diunduh dari investing.com versi ID) akan salah diparse menjadi NaN
    untuk seluruh baris."""
    s = series.astype(str).str.strip()

    def _parse_one(v):
        if v in ("", "nan", "None"):
            return np.nan
        has_comma, has_dot = "," in v, "." in v
        if has_comma and has_dot:
            if v.rfind(",") > v.rfind("."):
                v = v.replace(".", "").replace(",", ".")   # format ID/EU
            else:
                v = v.replace(",", "")                      # format AS
        elif has_comma and not has_dot:
            parts = v.split(",")
            v = v.replace(",", "") if (len(parts) > 1 and len(parts[-1]) == 3) else v.replace(",", ".")
        elif has_dot and not has_comma:
            parts = v.split(".")
            if len(parts) > 1 and len(parts[-1]) == 3:
                v = v.replace(".", "")
        return v

    return pd.to_numeric(s.apply(_parse_one), errors="coerce")


def safe_window(n, default, polyorder=3):
    """Pastikan window ganjil, <= n, dan > polyorder (syarat savgol_filter)."""
    w = default if n >= default else (n if n % 2 == 1 else n - 1)
    if w <= polyorder:
        w = polyorder + 1
    if w % 2 == 0:
        w += 1
    return w


def _read_any(file_bytes, file_name):
    """Baca CSV/XLSX data ANTM, termasuk XLSX yang isinya CSV mentah (1 kolom)."""
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

    if raw.shape[1] >= 7:
        if str(raw.iloc[0, 1]).strip().strip('"') in ("Terakhir", "Pembukaan"):
            raw = raw.iloc[1:].reset_index(drop=True)
        raw.columns = expected[:raw.shape[1]]
        return raw

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


def parse_nickel_manual(file_bytes, antm_index):
    """Baca file harga nikel dunia yang diunggah manual (mis. dari
    investing.com), dengan deteksi kolom & parsing angka yang tahan
    terhadap format Indonesia/Eropa maupun Amerika, plus diagnostik
    keterisisan tanggal terhadap rentang data ANTM."""
    raw = pd.read_csv(io.BytesIO(file_bytes))
    log = [f"Kolom terbaca: {raw.columns.tolist()}"]

    date_candidates = [c for c in raw.columns if any(k in c.lower() for k in ["date", "tanggal"])]
    price_candidates = [c for c in raw.columns if any(k in c.lower() for k in
                         ["price", "close", "terakhir", "value", "harga"])]
    if not date_candidates or not price_candidates:
        return None, log + ["[GAGAL] Kolom tanggal/harga tidak ditemukan."]

    date_col, price_col = date_candidates[0], price_candidates[0]
    log.append(f"Kolom tanggal: '{date_col}', kolom harga: '{price_col}'")

    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=True, errors="coerce")
    nickel = raw.set_index(date_col)[[price_col]].rename(columns={price_col: "Nickel_Close_Raw"})
    nickel["Nickel_Close_Raw"] = parse_price_string(nickel["Nickel_Close_Raw"])
    nickel = nickel.dropna()

    if len(nickel) == 0:
        return None, log + ["[GAGAL] Semua baris gagal diparse."]

    overlap = nickel.index.intersection(antm_index)
    log.append(f"Rentang file nikel: {nickel.index.min().date()} s.d. {nickel.index.max().date()} "
               f"({len(nickel)} baris)")
    log.append(f"Beririsan dengan data ANTM: {len(overlap)} tanggal")
    if len(overlap) == 0:
        return None, log + ["[GAGAL] Rentang tanggal file nikel tidak beririsan sama sekali dengan data ANTM."]

    return nickel.sort_index(), log


@st.cache_data(show_spinner=False)
def load_and_process(file_bytes, file_name, fetch_online, nickel_manual_bytes=None):
    df = _read_any(file_bytes, file_name)

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df = df.sort_values("Tanggal").reset_index(drop=True)

    # Deduplikasi tanggal ganda: index tanggal yang tidak unik dapat
    # menyebabkan error saat reindexing pada perhitungan Directional
    # Accuracy maupun validasi silang time-series.
    n_before = len(df)
    dup_mask = df.duplicated(subset="Tanggal", keep=False)
    log = []
    if dup_mask.any():
        log.append(f"[PERINGATAN] {dup_mask.sum()} baris bertanggal duplikat ditemukan & dibuang "
                    f"(menyisakan kemunculan pertama).")
    df = df.drop_duplicates(subset="Tanggal", keep="first")
    df = df.set_index("Tanggal")

    df = df.rename(columns={
        "Terakhir": "Close", "Pembukaan": "Open", "Tertinggi": "High",
        "Terendah": "Low", "Vol.": "Volume", "Perubahan%": "Perubahan%",
    })

    for col in ["Close", "Open", "High", "Low"]:
        raw = pd.to_numeric(df[col], errors="coerce")
        if raw.isna().mean() > 0.5:
            s = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")
        else:
            raw = raw * 1000
            df[col] = raw.apply(lambda x: x / 1000.0 if pd.notna(x) and x > 10000.0 else x)

    df["Volume"] = df["Volume"].apply(convert_volume)
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    start_date = str(df.index.min().date())
    end_date = str(df.index.max().date())

    gold, nickel, usdidr = None, None, None
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

            # PENTING: hanya NI=F (harga komoditas nikel dunia) yang dicoba.
            # TIDAK ada fallback ke saham perusahaan tambang (mis. INCO.JK),
            # karena itu bukan representasi valid dari "harga nikel dunia".
            try:
                nr = yf.download("NI=F", start=start_date, end=end_date, progress=False)
                if isinstance(nr.columns, pd.MultiIndex):
                    nr.columns = ["_".join(c).strip() for c in nr.columns]
                    c2 = [c for c in nr.columns if "Close" in c][0]
                else:
                    c2 = "Close"
                nr = nr[[c2]].rename(columns={c2: "Nickel_Close_Raw"})
                if len(nr) > 100 and nr["Nickel_Close_Raw"].isna().mean() < 0.5:
                    nickel = nr.copy()
                    log.append(f"Nikel dunia (NI=F): {len(nickel)} baris")
            except Exception:
                pass

            u = yf.download("IDR=X", start=start_date, end=end_date, progress=False)
            if isinstance(u.columns, pd.MultiIndex):
                u.columns = ["_".join(c).strip() for c in u.columns]
                cu = [c for c in u.columns if "Close" in c][0]
            else:
                cu = "Close"
            usdidr = u[[cu]].rename(columns={cu: "USDIDR_Close"})
            usdidr.index = pd.to_datetime(usdidr.index)
            log.append(f"USD/IDR (IDR=X): {len(usdidr)} baris")
        except Exception as e:
            log.append(f"Gagal yfinance: {e}")

    # Jika NI=F gagal, satu-satunya jalur pengganti yang sah adalah file
    # harga nikel dunia yang diunggah manual -- BUKAN proksi saham maupun
    # estimasi sintetis dari harga ANTM sendiri (itu akan menjadi
    # kebocoran data / data leakage).
    if nickel is None and nickel_manual_bytes is not None:
        nickel, nlog = parse_nickel_manual(nickel_manual_bytes, df.index)
        log.extend(nlog)

    if gold is None:
        gold = pd.DataFrame({"Gold_Close": np.nan}, index=df.index)
        log.append("[PERINGATAN] Emas tidak tersedia (offline & tidak ada file manual).")

    if usdidr is None:
        usdidr = pd.DataFrame({"USDIDR_Close": np.nan}, index=df.index)
        log.append("[PERINGATAN] USD/IDR tidak tersedia (offline).")

    if nickel is None:
        nickel = pd.DataFrame({"Nickel_Close_Raw": np.nan}, index=df.index)
        log.append("[PERINGATAN] Data nikel dunia tidak tersedia. Silakan unggah file "
                    "harga nikel dunia manual (mis. dari investing.com) di sidebar.")
    else:
        nmean = nickel["Nickel_Close_Raw"].mean()
        nickel["Nickel_Close"] = nickel["Nickel_Close_Raw"] / 10 if nmean > 10000 else nickel["Nickel_Close_Raw"]

    nickel.index = pd.to_datetime(nickel.index)
    df = df.join(gold[["Gold_Close"]], how="left")
    df = df.join(nickel[["Nickel_Close"]] if "Nickel_Close" in nickel.columns
                 else nickel[["Nickel_Close_Raw"]].rename(columns={"Nickel_Close_Raw": "Nickel_Close"}),
                 how="left")
    df = df.join(usdidr[["USDIDR_Close"]], how="left")
    for c in ["Gold_Close", "Nickel_Close", "USDIDR_Close"]:
        df[c] = df[c].ffill().bfill()

    return df, log


def add_indicators(df):
    df = df.copy()
    df = df.dropna()
    df["Daily Return"] = df["Close"].pct_change()
    # min_periods disamakan dengan notebook final (Cell 44): tanpa ini,
    # SMA_200 baru terisi setelah 200 baris penuh, membuang ~150 baris
    # lebih banyak dari yang seharusnya dan menghasilkan jumlah data
    # latih/uji yang TIDAK sinkron dengan Abstrak skripsi (2.672/668).
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=20).mean()
    df["SMA_200"] = df["Close"].rolling(window=200, min_periods=50).mean()
    df["RSI"] = compute_rsi(df["Close"], period=14)
    return df


def directional_accuracy(hasil_df, price_before):
    """Persentase hari model menebak arah (naik/turun) dengan benar,
    dibandingkan harga hari sebelumnya. Penyelarasan posisional (bukan
    reindex berbasis label) karena hasil_df selalu berasal dari
    price_before.shift(-1).dropna() -- memotong tepat satu baris
    terakhir tanpa mengubah urutan sisanya."""
    aligned_before = price_before.iloc[:len(hasil_df)].values
    actual_dir = np.sign(hasil_df["Aktual"].values - aligned_before)
    pred_dir = np.sign(hasil_df["Prediksi"].values - aligned_before)
    return (actual_dir == pred_dir).mean() * 100


@st.cache_resource(show_spinner=False)
def tune_models(X_train_vals, y_train_vals, cols, n_iter=10, n_splits=3):
    """Hyperparameter tuning dengan RandomizedSearchCV + TimeSeriesSplit,
    identik dengan pendekatan pada notebook (Bab 3.11)."""
    X_train = pd.DataFrame(X_train_vals, columns=cols)
    y_train = pd.Series(y_train_vals)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    xgb_param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    }
    xgb_search = RandomizedSearchCV(
        XGBRegressor(random_state=42), xgb_param_dist, n_iter=n_iter,
        scoring="neg_mean_absolute_error", cv=tscv, random_state=42, n_jobs=1,
    )
    xgb_search.fit(X_train, y_train)

    rf_param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 8, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(random_state=42), rf_param_dist, n_iter=n_iter,
        scoring="neg_mean_absolute_error", cv=tscv, random_state=42, n_jobs=1,
    )
    rf_search.fit(X_train, y_train)

    return xgb_search.best_estimator_, rf_search.best_estimator_, xgb_search.best_params_, rf_search.best_params_


def evaluate(y_true, y_pred, name, da=None):
    result = {
        "Model": name,
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R²": r2_score(y_true, y_pred),
    }
    if da is not None:
        result["Directional Accuracy (%)"] = da
    return result


def predict_price(model, X_te, close_te):
    ret_pred = model.predict(X_te)
    price_pred = close_te.values * (1 + ret_pred)
    out = pd.DataFrame({
        "Aktual": close_te.shift(-1),
        "Prediksi": price_pred
    }, index=close_te.index).dropna()
    return out


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("⚙️ Pengaturan")
uploaded_file = st.sidebar.file_uploader("Upload Data ANTM (CSV / XLSX)", type=["csv", "xlsx", "xls"])
fetch_online = st.sidebar.checkbox("Unduh emas, nikel & USD/IDR via yfinance", value=True,
                                    help="Matikan jika tanpa internet.")
nickel_manual_file = st.sidebar.file_uploader(
    "Upload data nikel dunia manual (opsional)", type=["csv"],
    help="Dipakai HANYA jika NI=F gagal diunduh otomatis. Unduh dari investing.com "
         "dengan rentang tanggal yang sama dengan data ANTM Anda.")
menu = st.sidebar.radio("Navigasi", ["🏠 Beranda", "📊 Eksplorasi Data",
                                      "📈 Analisis Teknikal", "🤖 Pemodelan & Evaluasi",
                                      "🎯 Perbaikan Directional Accuracy", "🔮 Prediksi"])
st.sidebar.markdown("---")
st.sidebar.info("Skripsi Prediksi Saham ANTAM\nXGBoost vs Random Forest (Tuned)")


# ============================================================================
# BERANDA
# ============================================================================
if menu == "🏠 Beranda":
    st.title("📈 Prediksi Harga Saham ANTAM (ANTM)")
    st.markdown("""
    Aplikasi ini disamakan dengan notebook skripsi final:
    - **10 fitur** input: Open, High, Low, Volume, SMA_50, SMA_200, RSI,
      Gold_Close, Nickel_Close, **USDIDR_Close**
    - Target = **return** harian, dikonversi balik ke harga
    - Data **emas (GC=F)**, **nikel dunia (NI=F)**, dan **USD/IDR (IDR=X)**
      diunduh asli via yfinance -- bukan proksi saham
    - Model dituning otomatis (**RandomizedSearchCV**)
    - Evaluasi memakai **4 metrik + Directional Accuracy**
    - Validasi silang **time-series** (k-fold walk-forward)

    Upload `Data_Historis_ANTM.csv` atau `.xlsx` di sidebar untuk memulai.
    """)
    if uploaded_file is None:
        st.warning("⬅️ Upload file terlebih dahulu.")


if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    nickel_bytes = nickel_manual_file.getvalue() if nickel_manual_file is not None else None
    with st.spinner("Memuat & memproses data (mengunduh emas/nikel/USD-IDR bila online)..."):
        df, log = load_and_process(file_bytes, uploaded_file.name, fetch_online, nickel_bytes)
        df = add_indicators(df)

    features = FEATURES

    df_model = df.dropna(subset=["SMA_50", "SMA_200", "RSI",
                                  "Gold_Close", "Nickel_Close", "USDIDR_Close"]).copy()
    df_model["Target_Return"] = df_model["Close"].pct_change().shift(-1)
    df_model = df_model.dropna(subset=["Target_Return"])

    X = df_model[features]
    y = df_model["Target_Return"]
    close_price = df_model["Close"]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    close_train, close_test = close_price.iloc[:split_idx], close_price.iloc[split_idx:]

    # --------------------------------------------------------------------
    if menu == "📊 Eksplorasi Data":
        st.title("📊 Eksplorasi Data")
        for l in log:
            st.caption(("⚠️ " if "PERINGATAN" in l or "GAGAL" in l else "• ") + l)
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
        st.pyplot(fig); plt.close(fig)

        st.subheader("2.7 Volume Perdagangan")
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df.index, df["Volume"], color="darkorange", alpha=0.25, linewidth=0.7, label="Volume Asli")
        ax.plot(df.index, df["Volume"].rolling(30, center=True, min_periods=1).mean(),
                color="darkorange", linewidth=2, label="Volume (MA-30)")
        ax.set_title("Volume Perdagangan Saham ANTAM"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Volume")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e9:.1f}"))
        ax.legend(); st.pyplot(fig); plt.close(fig)

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
        plt.xticks(rotation=45); st.pyplot(fig); plt.close(fig)

        st.subheader("2.9 Boxplot Harga per Tahun")
        df_box = df.copy(); df_box["Year"] = df_box.index.year
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(x="Year", y="Close", data=df_box, hue="Year", palette="Set2", legend=False, ax=ax)
        ax.set_title("Boxplot Harga Penutupan per Tahun"); ax.set_xlabel("Tahun"); ax.set_ylabel("Harga Close (IDR)")
        st.pyplot(fig); plt.close(fig)

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
        st.pyplot(fig); plt.close(fig)

        st.subheader("2.11 Matriks Korelasi (Mean, Std, CV)")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cv_per_year.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Matriks Korelasi (Mean, Std, CV)")
        st.pyplot(fig); plt.close(fig)

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
        ax.legend(); st.pyplot(fig); plt.close(fig)

        st.subheader("3.4 Distribusi & Rata-Rata Daily Return")
        dr = df["Daily Return"].dropna()
        avg_dr, std_dr = dr.mean(), dr.std()
        c1, c2, c3 = st.columns(3)
        c1.metric("Average Daily Return", f"{avg_dr*100:.4f}%")
        c2.metric("Std Daily Return", f"{std_dr*100:.4f}%")
        c3.metric("Jumlah Hari", f"{len(dr):,}")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(dr, bins=60, kde=True, color="teal", ax=ax)
        ax.axvline(avg_dr, color="red", linestyle="--", linewidth=1.8,
                   label=f"Average Daily Return = {avg_dr:.4f}")
        ax.set_title("Distribusi Daily Return Saham ANTAM")
        ax.set_xlabel("Daily Return"); ax.set_ylabel("Frekuensi"); ax.legend()
        st.pyplot(fig); plt.close(fig)

        st.subheader("3.4.1 Average Daily Return per Tahun")
        dfy = df.copy(); dfy["Year"] = dfy.index.year
        avg_per_year = dfy.groupby("Year")["Daily Return"].mean()
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["green" if v >= 0 else "red" for v in avg_per_year.values]
        bars = ax.bar(avg_per_year.index.astype(str), avg_per_year.values * 100,
                       color=colors, edgecolor="black", alpha=0.7)
        for b, v in zip(bars, avg_per_year.values * 100):
            ax.text(b.get_x() + b.get_width()/2, v, f"{v:.3f}%",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Average Daily Return Saham ANTAM per Tahun")
        ax.set_xlabel("Tahun"); ax.set_ylabel("Average Daily Return (%)")
        st.pyplot(fig); plt.close(fig)

        st.subheader("3.5 Moving Average (SMA-50 & SMA-200)")
        fig, ax = plt.subplots(figsize=(14, 6))
        wl_c = safe_window(len(df["Close"].dropna()), 51)
        ax.plot(df.index, df["Close"], label="Close (Asli)", color="black", alpha=0.2, linewidth=0.7)
        ax.plot(df.index, savgol_filter(df["Close"].values, wl_c, 3), label="Close (Smoothed)", color="black", linewidth=1.8)
        ax.plot(df.index, df["SMA_50"], label="SMA 50 hari", color="blue")
        ax.plot(df.index, df["SMA_200"], label="SMA 200 hari", color="red")
        ax.set_title("Grafik Moving Average – Saham ANTAM"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga (IDR)")
        ax.legend(); st.pyplot(fig); plt.close(fig)

        st.subheader("3.6 Relative Strength Index (RSI)")
        wl_r = safe_window(len(df["RSI"].dropna()), 21)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df.index, df["RSI"], color="purple", linewidth=0.5, alpha=0.25, label="RSI (14) Asli")
        ax.plot(df.index, savgol_filter(df["RSI"].fillna(50).values, wl_r, 3), color="purple", linewidth=2, label="RSI (14) Smoothed")
        ax.axhline(70, linestyle="--", color="red", label="Overbought (70)")
        ax.axhline(30, linestyle="--", color="green", label="Oversold (30)")
        ax.axhline(50, linestyle=":", color="gray", alpha=0.6)
        ax.set_title("Grafik RSI – Saham ANTAM"); ax.set_xlabel("Tanggal"); ax.set_ylabel("RSI")
        ax.legend(); st.pyplot(fig); plt.close(fig)

        st.subheader("3.7 Pemilihan Fitur")
        st.markdown(f"Fitur yang digunakan sebagai variabel input (X) terdiri dari **{len(features)} fitur** berikut. "
                    f"Target (y) adalah **return** — bukan harga langsung — untuk mengatasi keterbatasan "
                    f"ekstrapolasi model berbasis pohon.")
        fitur_df = pd.DataFrame({
            "No": range(1, len(features) + 1),
            "Nama Fitur": features,
            "Keterangan": FEATURE_DESC,
        })
        st.dataframe(fitur_df, use_container_width=True, hide_index=True)

        st.subheader("3.7.1 Korelasi ANTAM vs Emas, Nikel & USD/IDR")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df_model[["Close", "Gold_Close", "Nickel_Close", "USDIDR_Close"]].corr(),
                    annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Korelasi ANTAM vs Emas, Nikel & USD/IDR"); st.pyplot(fig); plt.close(fig)

        st.subheader("3.7.2 Tren ANTAM, Emas, Nikel, USD/IDR (Base = 100)")
        dt = df_model[["Close", "Gold_Close", "Nickel_Close", "USDIDR_Close"]].copy()
        dt = dt / dt.iloc[0] * 100
        wl_t = safe_window(len(dt), 51)
        fig, ax = plt.subplots(figsize=(14, 5))
        for col, color, label in [("Close", "navy", "ANTAM"), ("Gold_Close", "goldenrod", "Emas"),
                                   ("Nickel_Close", "green", "Nikel"), ("USDIDR_Close", "purple", "USD/IDR")]:
            ax.plot(dt.index, savgol_filter(dt[col].values, wl_t, 3), label=label, color=color, linewidth=2)
        ax.set_title("Perbandingan Tren (Base 100)"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Ternormalisasi")
        ax.legend(); st.pyplot(fig); plt.close(fig)

        st.subheader("3.8 Heatmap Korelasi Fitur")
        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(df_model[features + ["Close"]].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        ax.set_title("Matriks Korelasi Fitur dan Target"); st.pyplot(fig); plt.close(fig)

        st.subheader("3.9 Split Data Latih dan Data Uji (80:20)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Data", f"{len(X):,}")
        c2.metric("Data Latih (80%)", f"{X_train.shape[0]:,}")
        c3.metric("Data Uji (20%)", f"{X_test.shape[0]:,}")
        st.caption("Pembagian dilakukan secara berurutan (shuffle=False) agar urutan waktu "
                   "tetap terjaga sesuai karakteristik data time-series.")

        st.subheader("3.10 Validasi Silang Time-Series (K-Fold Walk-Forward)")
        st.caption("Satu titik pemisahan 80:20 saja rawan bias karena kondisi pasar di awal "
                   "dan akhir rentang data bisa sangat berbeda. TimeSeriesSplit menguji model "
                   "pada 5 fold berurutan tanpa mengacak data, sehingga tidak terjadi kebocoran "
                   "data (data leakage).")
        if st.checkbox("Jalankan validasi silang time-series (agak lambat)"):
            with st.spinner("Menjalankan 5-fold walk-forward validation..."):
                tscv = TimeSeriesSplit(n_splits=5)
                cv_rows = []
                for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
                    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
                    close_te = close_price.iloc[te_idx]
                    for name, mdl in [("XGBoost", XGBRegressor(n_estimators=100, random_state=42)),
                                       ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))]:
                        mdl.fit(X_tr, y_tr)
                        hasil_fold = predict_price(mdl, X_te, close_te)
                        mape_fold = mean_absolute_percentage_error(hasil_fold["Aktual"], hasil_fold["Prediksi"]) * 100
                        da_fold = directional_accuracy(hasil_fold, close_te)
                        cv_rows.append({"Fold": fold, "Model": name, "MAPE (%)": mape_fold,
                                         "Directional Accuracy (%)": da_fold})
                cv_result = pd.DataFrame(cv_rows)
            st.dataframe(cv_result.style.format({"MAPE (%)": "{:.4f}", "Directional Accuracy (%)": "{:.2f}"}),
                         use_container_width=True)
            cv_summary = cv_result.groupby("Model")[["MAPE (%)", "Directional Accuracy (%)"]].agg(["mean", "std"])
            st.markdown("**Ringkasan (rata-rata ± std lintas fold):**")
            st.dataframe(cv_summary, use_container_width=True)

    # --------------------------------------------------------------------
    elif menu == "🤖 Pemodelan & Evaluasi":
        st.title("🤖 Pemodelan & Evaluasi")

        do_tune = st.checkbox("Aktifkan hyperparameter tuning (RandomizedSearchCV) — lebih lambat, lebih akurat",
                               value=False)

        with st.spinner("Melatih model..."):
            if do_tune:
                xgb_model, rf_model, xgb_best, rf_best = tune_models(
                    X_train.values, y_train.values, features)
            else:
                xgb_model = XGBRegressor(n_estimators=100, random_state=42)
                xgb_model.fit(X_train, y_train)
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                rf_model.fit(X_train, y_train)
                xgb_best = rf_best = None

            hasil_xgb = predict_price(xgb_model, X_test, close_test)
            hasil_rf = predict_price(rf_model, X_test, close_test)
            y_actual_xgb, y_pred_xgb = hasil_xgb["Aktual"].values, hasil_xgb["Prediksi"].values
            y_actual_rf, y_pred_rf = hasil_rf["Aktual"].values, hasil_rf["Prediksi"].values
            da_xgb = directional_accuracy(hasil_xgb, close_test)
            da_rf = directional_accuracy(hasil_rf, close_test)
        st.success("Model selesai dilatih!")

        if do_tune and xgb_best is not None:
            with st.expander("Lihat parameter hasil tuning"):
                st.write("**XGBoost terbaik:**", xgb_best)
                st.write("**Random Forest terbaik:**", rf_best)

        st.subheader("4.1 & 4.3 Pelatihan Model (XGBoost & Random Forest)")
        c1, c2 = st.columns(2)
        c1.metric("Data Latih", f"{X_train.shape[0]:,}")
        c2.metric("Data Uji", f"{X_test.shape[0]:,}")

        st.markdown("**Hasil Evaluasi — 4 Metrik + Directional Accuracy**")
        hasil = pd.DataFrame([
            evaluate(y_actual_xgb, y_pred_xgb, "XGBoost", da_xgb),
            evaluate(y_actual_rf, y_pred_rf, "Random Forest", da_rf),
        ])
        st.dataframe(
            hasil.style.format({"MAPE (%)": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}",
                                 "R²": "{:.4f}", "Directional Accuracy (%)": "{:.2f}"}),
            use_container_width=True)

        best_mape = hasil.loc[hasil["MAPE (%)"].idxmin(), "Model"]
        best_da = hasil.loc[hasil["Directional Accuracy (%)"].idxmax(), "Model"]
        st.info(f"🏆 MAPE terendah: **{best_mape}**  |  🎯 Directional Accuracy tertinggi: **{best_da}**")
        if hasil["Directional Accuracy (%)"].max() < 50:
            st.warning("⚠️ Directional Accuracy kedua model berada di bawah 50% — keduanya lebih sering "
                       "salah menebak arah pergerakan harga dibandingkan tebakan acak. MAPE yang rendah "
                       "saja tidak cukup untuk menyimpulkan model layak dipakai dalam keputusan investasi.")

        st.subheader("4.2 & 4.5 Feature Importance")
        ca, cb = st.columns(2)
        with ca:
            imp = pd.DataFrame({"Fitur": features, "Importance": xgb_model.feature_importances_}).sort_values("Importance", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x="Importance", y="Fitur", data=imp, hue="Fitur", palette="coolwarm", legend=False, ax=ax)
            ax.set_title("Feature Importance – XGBoost"); st.pyplot(fig); plt.close(fig)
        with cb:
            imp = pd.DataFrame({"Fitur": features, "Importance": rf_model.feature_importances_}).sort_values("Importance", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(x="Importance", y="Fitur", data=imp, hue="Fitur", palette="viridis", legend=False, ax=ax)
            ax.set_title("Feature Importance – Random Forest"); st.pyplot(fig); plt.close(fig)

        st.subheader("4.6 Tabel Perbandingan Aktual vs Prediksi")
        hasil_pred = pd.DataFrame({
            "Close (Aktual)": hasil_xgb["Aktual"],
            "XGBoost_Pred": hasil_xgb["Prediksi"],
            "RF_Pred": hasil_rf["Prediksi"]
        }).sort_index()
        st.dataframe(pd.concat([hasil_pred.head(5), hasil_pred.tail(5)]).style.format("{:,.2f}"),
                     use_container_width=True)

        st.subheader("5.3 Aktual vs Prediksi (Gabungan)")
        pdf_ = pd.DataFrame({"Aktual": hasil_xgb["Aktual"], "XGBoost": hasil_xgb["Prediksi"],
                              "Random Forest": hasil_rf["Prediksi"]}).sort_index()
        wl = safe_window(len(pdf_), 21)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(pdf_.index, savgol_filter(pdf_["Aktual"].values, wl, 3), label="Aktual (Smoothed)", color="black", linewidth=2.5)
        ax.plot(pdf_.index, savgol_filter(pdf_["XGBoost"].values, wl, 3), label="XGBoost (Smoothed)", color="blue", linewidth=2)
        ax.plot(pdf_.index, savgol_filter(pdf_["Random Forest"].values, wl, 3), label="Random Forest (Smoothed)", color="red", linewidth=2)
        ax.set_title("Aktual vs Prediksi (Data Uji)"); ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Close (IDR)")
        ax.legend(); st.pyplot(fig); plt.close(fig)

        st.subheader("5.4 Scatter Plot Aktual vs Prediksi")
        ca, cb = st.columns(2)
        with ca:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_actual_xgb, y_pred_xgb, alpha=0.4, color="blue", s=12)
            lims = [min(y_actual_xgb.min(), y_pred_xgb.min()), max(y_actual_xgb.max(), y_pred_xgb.max())]
            ax.plot(lims, lims, "--", color="black", linewidth=1)
            ax.set_title("Scatter – XGBoost"); ax.set_xlabel("Aktual"); ax.set_ylabel("Prediksi")
            st.pyplot(fig); plt.close(fig)
        with cb:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_actual_rf, y_pred_rf, alpha=0.4, color="red", s=12)
            lims = [min(y_actual_rf.min(), y_pred_rf.min()), max(y_actual_rf.max(), y_pred_rf.max())]
            ax.plot(lims, lims, "--", color="black", linewidth=1)
            ax.set_title("Scatter – Random Forest"); ax.set_xlabel("Aktual"); ax.set_ylabel("Prediksi")
            st.pyplot(fig); plt.close(fig)

        st.subheader("5.5 Distribusi Residual / Error")
        res_xgb, res_rf = y_actual_xgb - y_pred_xgb, y_actual_rf - y_pred_rf
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.histplot(res_xgb, bins=40, kde=True, color="blue", alpha=0.5, label="XGBoost", ax=ax)
        sns.histplot(res_rf, bins=40, kde=True, color="red", alpha=0.5, label="Random Forest", ax=ax)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Distribusi Residual (Error) Kedua Model")
        ax.set_xlabel("Residual (Aktual - Prediksi)"); ax.set_ylabel("Frekuensi")
        ax.legend(); st.pyplot(fig); plt.close(fig)

        st.subheader("5.6 Perbandingan MAPE & Directional Accuracy")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        mape_vals = [hasil.loc[0, "MAPE (%)"], hasil.loc[1, "MAPE (%)"]]
        da_vals = [hasil.loc[0, "Directional Accuracy (%)"], hasil.loc[1, "Directional Accuracy (%)"]]
        bars0 = axes[0].bar(["XGBoost", "Random Forest"], mape_vals, color=["blue", "red"], edgecolor="black")
        for b, v in zip(bars0, mape_vals):
            axes[0].text(b.get_x() + b.get_width()/2, v, f"{v:.4f}%", ha="center", va="bottom", fontsize=10)
        axes[0].set_title("Perbandingan MAPE"); axes[0].set_ylabel("MAPE (%)")
        bars1 = axes[1].bar(["XGBoost", "Random Forest"], da_vals, color=["blue", "red"], edgecolor="black")
        axes[1].axhline(50, color="gray", linestyle="--", label="Tebakan Acak (50%)")
        for b, v in zip(bars1, da_vals):
            axes[1].text(b.get_x() + b.get_width()/2, v, f"{v:.2f}%", ha="center", va="bottom", fontsize=10)
        axes[1].set_title("Perbandingan Directional Accuracy"); axes[1].set_ylabel("Directional Accuracy (%)")
        axes[1].legend()
        st.pyplot(fig); plt.close(fig)

        st.session_state["xgb_model"] = xgb_model
        st.session_state["rf_model"] = rf_model

    # --------------------------------------------------------------------
    elif menu == "🎯 Perbaikan Directional Accuracy":
        st.title("🎯 Perbaikan Directional Accuracy")
        st.caption("Diagnostik menunjukkan model bias memprediksi arah \"naik\" karena fitur "
                   "berupa level harga absolut yang membawa informasi tren jangka panjang. "
                   "Perbaikan dilakukan dengan detrending (semua fitur diubah jadi return/rasio "
                   "relatif) dan class balancing.")

        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix

        df_dir = df_model.copy()
        df_dir["ret_1"]      = df_dir["Close"].pct_change(1)
        df_dir["ret_5"]      = df_dir["Close"].pct_change(5)
        df_dir["ret_10"]     = df_dir["Close"].pct_change(10)
        df_dir["vol_10"]     = df_dir["Close"].pct_change().rolling(10).std()
        df_dir["hl_range"]   = (df_dir["High"] - df_dir["Low"]) / df_dir["Close"]
        df_dir["co_ret"]     = (df_dir["Close"] - df_dir["Open"]) / df_dir["Open"]
        df_dir["sma50_rel"]  = df_dir["Close"] / df_dir["SMA_50"] - 1
        df_dir["sma200_rel"] = df_dir["Close"] / df_dir["SMA_200"] - 1
        df_dir["rsi_n"]      = df_dir["RSI"] / 100
        df_dir["vol_chg"]    = df_dir["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
        df_dir["gold_ret"]   = df_dir["Gold_Close"].pct_change()
        df_dir["nickel_ret"] = df_dir["Nickel_Close"].pct_change()
        df_dir["usd_ret"]    = df_dir["USDIDR_Close"].pct_change()

        FITUR_DETREND = ["ret_1", "ret_5", "ret_10", "vol_10", "hl_range", "co_ret",
                         "sma50_rel", "sma200_rel", "rsi_n", "vol_chg",
                         "gold_ret", "nickel_ret", "usd_ret"]
        df_dir = df_dir.replace([np.inf, -np.inf], np.nan).dropna(subset=FITUR_DETREND + ["Target_Return"])

        X_dir = df_dir[FITUR_DETREND]
        y_dir = (df_dir["Target_Return"] > 0).astype(int)
        split_dir = int(len(X_dir) * 0.8)
        X_dtr, X_dte = X_dir.iloc[:split_dir], X_dir.iloc[split_dir:]
        y_dtr, y_dte = y_dir.iloc[:split_dir].values, y_dir.iloc[split_dir:].values
        spw = (y_dtr == 0).sum() / max((y_dtr == 1).sum(), 1)

        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah Fitur Detrended", len(FITUR_DETREND))
        c2.metric("Data Latih / Uji", f"{len(X_dtr)} / {len(X_dte)}")
        c3.metric("scale_pos_weight", f"{spw:.3f}")

        with st.spinner("Melatih model klasifikasi arah..."):
            hasil_dir, preds_dir = [], {}
            for nama, model in [
                ("XGBoost", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                                          subsample=0.8, colsample_bytree=0.8,
                                          scale_pos_weight=spw, random_state=42, eval_metric="logloss")),
                ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=4,
                                                         max_features="log2", class_weight="balanced",
                                                         random_state=42)),
            ]:
                model.fit(X_dtr, y_dtr)
                pred = model.predict(X_dte)
                preds_dir[nama] = pred
                hasil_dir.append({
                    "Model": nama,
                    "Directional Accuracy (%)": accuracy_score(y_dte, pred) * 100,
                    "Proporsi Prediksi Naik (%)": pred.mean() * 100,
                })

        rng_cmp = np.random.default_rng(42)
        da_acak = np.mean([(y_dte == rng_cmp.choice([0, 1], size=len(y_dte))).mean() * 100
                            for _ in range(500)])
        da_mayoritas = max(y_dte.mean(), 1 - y_dte.mean()) * 100

        st.subheader("Hasil Setelah Perbaikan")
        tabel_dir = pd.DataFrame(hasil_dir)
        st.dataframe(tabel_dir.style.format({"Directional Accuracy (%)": "{:.2f}",
                                              "Proporsi Prediksi Naik (%)": "{:.2f}"}),
                     use_container_width=True)
        st.caption(f"Proporsi arah naik aktual pada data uji: {y_dte.mean()*100:.2f}%")

        b1, b2 = st.columns(2)
        b1.metric("Baseline Tebakan Acak", f"{da_acak:.2f}%")
        b2.metric("Baseline Kelas Mayoritas", f"{da_mayoritas:.2f}%")

        best_da = tabel_dir["Directional Accuracy (%)"].max()
        if best_da > da_mayoritas:
            st.success(f"✅ Model terbaik ({best_da:.2f}%) melampaui baseline tebakan acak "
                       f"({da_acak:.2f}%) DAN baseline kelas mayoritas ({da_mayoritas:.2f}%).")
        elif best_da > da_acak:
            st.warning(f"⚠️ Model terbaik ({best_da:.2f}%) melampaui tebakan acak tetapi belum "
                       f"melampaui baseline kelas mayoritas ({da_mayoritas:.2f}%).")
        else:
            st.error(f"❌ Model terbaik ({best_da:.2f}%) belum melampaui baseline tebakan acak.")

        st.subheader("Confusion Matrix Setelah Perbaikan")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, nama in zip(axes, ["XGBoost", "Random Forest"]):
            cm = confusion_matrix(y_dte, preds_dir[nama])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=["Turun", "Naik"], yticklabels=["Turun", "Naik"], ax=ax)
            ax.set_title(f"Confusion Matrix — {nama}")
            ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.subheader("Validasi Walk-Forward (TimeSeriesSplit 5-fold)")
        if st.checkbox("Jalankan validasi silang time-series (agak lambat)"):
            with st.spinner("Menjalankan 5-fold walk-forward..."):
                tscv_dir = TimeSeriesSplit(n_splits=5)
                baris_cv = []
                for nama, buat in [
                    ("XGBoost", lambda w: XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03,
                                                        subsample=0.8, colsample_bytree=0.8,
                                                        scale_pos_weight=w, random_state=42,
                                                        eval_metric="logloss")),
                    ("Random Forest", lambda w: RandomForestClassifier(n_estimators=200, max_depth=5,
                                                                       min_samples_leaf=4, max_features="log2",
                                                                       class_weight="balanced", random_state=42)),
                ]:
                    for k, (tr, te) in enumerate(tscv_dir.split(X_dir), start=1):
                        ytr_k, yte_k = y_dir.iloc[tr].values, y_dir.iloc[te].values
                        w = (ytr_k == 0).sum() / max((ytr_k == 1).sum(), 1)
                        m = buat(w); m.fit(X_dir.iloc[tr], ytr_k)
                        baris_cv.append({
                            "Fold": k,
                            "Periode": f"{df_dir.index[te].min().date()} s.d. {df_dir.index[te].max().date()}",
                            "Model": nama,
                            "Directional Accuracy (%)": accuracy_score(yte_k, m.predict(X_dir.iloc[te])) * 100,
                        })
                cv_dir = pd.DataFrame(baris_cv)
            st.dataframe(cv_dir.style.format({"Directional Accuracy (%)": "{:.2f}"}),
                         use_container_width=True)
            st.markdown("**Ringkasan (rata-rata ± std lintas fold):**")
            st.dataframe(cv_dir.groupby("Model")["Directional Accuracy (%)"].agg(["mean", "std"]),
                         use_container_width=True)

    # --------------------------------------------------------------------
    elif menu == "🔮 Prediksi":
        st.title("🔮 Prediksi Harga ke Depan")
        st.subheader("5.7 Prediksi N Hari ke Depan")
        st.caption("SMA_50, SMA_200, dan RSI dihitung ULANG setiap iterasi dari riwayat harga "
                   "(aktual + prediksi sejauh ini). Gold_Close, Nickel_Close, dan USDIDR_Close "
                   "diekstrapolasi memakai naive drift (rata-rata log-return historis), bukan "
                   "dibekukan konstan.")

        xgb_model = st.session_state.get("xgb_model")
        rf_model = st.session_state.get("rf_model")
        if xgb_model is None or rf_model is None:
            with st.spinner("Melatih model dasar..."):
                xgb_model = XGBRegressor(n_estimators=100, random_state=42)
                xgb_model.fit(X_train, y_train)
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                rf_model.fit(X_train, y_train)

        n_days = st.slider("Jumlah hari prediksi", 7, 60, 30)
        drift_window = st.slider("Jendela drift (hari) untuk emas/nikel/USD-IDR", 20, 120, 60)

        if st.button("Jalankan Prediksi"):
            close_history_base = df_model["Close"].copy()
            last_volume = df_model["Volume"].iloc[-1]

            def estimate_daily_drift(series, window):
                recent = series.iloc[-window:]
                if len(recent) < 10 or (recent <= 0).any():
                    return 0.0
                log_returns = np.log(recent / recent.shift(1)).dropna()
                return log_returns.mean() if len(log_returns) > 0 else 0.0

            def project_forward(last_value, drift, n_steps):
                return [last_value * np.exp(drift * step) for step in range(1, n_steps + 1)]

            gold_drift = estimate_daily_drift(df_model["Gold_Close"], drift_window)
            nickel_drift = estimate_daily_drift(df_model["Nickel_Close"], drift_window)
            usdidr_drift = estimate_daily_drift(df_model["USDIDR_Close"], drift_window)

            gold_fc = project_forward(df_model["Gold_Close"].iloc[-1], gold_drift, n_days)
            nickel_fc = project_forward(df_model["Nickel_Close"].iloc[-1], nickel_drift, n_days)
            usdidr_fc = project_forward(df_model["USDIDR_Close"].iloc[-1], usdidr_drift, n_days)

            def build_next_features(close_history, next_price, day_idx):
                sma_50 = close_history.rolling(window=50, min_periods=20).mean().iloc[-1]
                sma_200 = close_history.rolling(window=200, min_periods=50).mean().iloc[-1]
                rsi_val = compute_rsi(close_history, period=14).iloc[-1]
                row = pd.DataFrame({
                    "Open": [next_price], "High": [next_price], "Low": [next_price],
                    "Volume": [last_volume],
                    "SMA_50": [sma_50], "SMA_200": [sma_200], "RSI": [rsi_val],
                    "Gold_Close": [gold_fc[day_idx]], "Nickel_Close": [nickel_fc[day_idx]],
                    "USDIDR_Close": [usdidr_fc[day_idx]],
                })[features]
                return row

            def run_forecast(model):
                price = close_price.iloc[-1]
                close_hist = close_history_base.copy()
                preds = []
                for i in range(n_days):
                    feat = build_next_features(close_hist, price, i)
                    ret = model.predict(feat)[0]
                    price = price * (1 + ret)
                    preds.append(price)
                    close_hist = pd.concat([close_hist, pd.Series(
                        [price], index=[close_hist.index[-1] + pd.Timedelta(days=1)])])
                return preds

            with st.spinner("Menjalankan prediksi rekursif..."):
                preds_xgb = run_forecast(xgb_model)
                preds_rf = run_forecast(rf_model)

            fdates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days, freq="B")
            fdf = pd.DataFrame({"XGBoost": preds_xgb, "Random Forest": preds_rf}, index=fdates)

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df.index[-100:], df["Close"].iloc[-100:], color="black", label="Historis", linewidth=1.5)
            ax.plot(fdf.index, fdf["XGBoost"], color="blue", marker="o", markersize=3, label="Prediksi XGBoost")
            ax.plot(fdf.index, fdf["Random Forest"], color="red", marker="o", markersize=3, label="Prediksi Random Forest")
            ax.set_title(f"Prediksi {n_days} Hari ke Depan")
            ax.set_xlabel("Tanggal"); ax.set_ylabel("Harga Close (IDR)"); ax.legend()
            st.pyplot(fig); plt.close(fig)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("XGBoost — Hari ke-1", f"Rp {preds_xgb[0]:,.0f}")
                st.metric(f"XGBoost — Hari ke-{n_days}", f"Rp {preds_xgb[-1]:,.0f}")
            with c2:
                st.metric("Random Forest — Hari ke-1", f"Rp {preds_rf[0]:,.0f}")
                st.metric(f"Random Forest — Hari ke-{n_days}", f"Rp {preds_rf[-1]:,.0f}")

            st.markdown("**Drift harian rata-rata variabel eksogen:**")
            st.caption(f"Emas: {gold_drift*100:+.4f}%/hari · Nikel: {nickel_drift*100:+.4f}%/hari · "
                       f"USD/IDR: {usdidr_drift*100:+.4f}%/hari (dihitung dari {drift_window} hari terakhir)")

            st.dataframe(fdf.style.format("Rp {:,.0f}"), use_container_width=True)
            st.caption("Prediksi menggunakan metode return yang direkonstruksi secara iteratif. "
                       "SMA/RSI dihitung ulang tiap hari; Gold/Nickel/USD-IDR diekstrapolasi "
                       "dengan naive drift, bukan dibekukan konstan.")
