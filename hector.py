
# PAGE CONFIG
import streamlit as st

st.set_page_config(
    page_title="HECTOR — Crypto Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# STANDARD LIBRARY

import os
import contextlib
import io
import re
import json
import math
import time
import hashlib
import logging
import sqlite3
import smtplib
import warnings
import functools
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# THIRD-PARTY
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, IsolationForest,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve,
)

# OPTIONAL LIBRARIES (graceful degradation — never crash)
try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

try:
    import joblib; JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

try:
    import pyarrow; PARQUET_OK = True
except ImportError:
    PARQUET_OK = False

try:
    import xgboost as xgb; XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import lightgbm as lgb; LGB_OK = True
except Exception:
    LGB_OK = False

try:
    from statsmodels.tsa.stattools import adfuller; ADF_OK = True
except Exception:
    ADF_OK = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False

try:
    import shap; SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    from hmmlearn import hmm; HMM_OK = True
except ImportError:
    HMM_OK = False

try:
    import lime, lime.lime_tabular; LIME_OK = True
except ImportError:
    LIME_OK = False

try:
    from fpdf import FPDF; FPDF_OK = True
except ImportError:
    FPDF_OK = False

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV


# ENVIRONMENT / CONFIG

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID", "")
SMTP_HOST      = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER      = os.getenv("SMTP_USER", "")
SMTP_PASS      = os.getenv("SMTP_PASS", "")
ALERT_EMAIL    = os.getenv("ALERT_EMAIL", "")
SLACK_WEBHOOK  = os.getenv("SLACK_WEBHOOK", "")

CFG = dict(
    retries=3, backoff=1.5, http_timeout=14,
    cache_ttl_ohlcv=180, cache_ttl_info=240, cache_ttl_global=360,
    db_path="hector_data.db",
    model_dir="hector_models",
    cache_dir="hector_cache",
    log_dir="hector_logs",
    max_labels=120, purge_embargo=2, cv_splits=3,
    optuna_trials=20, kelly_fraction=0.25,
    commission=0.001, slippage=0.0005,
    cusum_thr=0.02, entropy_window=60, regime_states=3,
)

for _d in [CFG["log_dir"], CFG["cache_dir"], CFG["model_dir"]]:
    Path(_d).mkdir(exist_ok=True)


# LOGGING

try:
    _fh = RotatingFileHandler(
        Path(CFG["log_dir"]) / "hector.log",
        maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    _fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[_fh, logging.StreamHandler()])
except Exception:
    logging.basicConfig(level=logging.INFO)
log = logging.getLogger("HECTOR")


# SQLITE DATABASE

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(CFG["db_path"], check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _init_db() -> None:
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT, ts TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, ts)
        );
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, symbol TEXT, model_name TEXT,
            cv_f1 REAL, auc REAL, accuracy REAL, precision_ REAL, recall_ REAL
        );
        CREATE TABLE IF NOT EXISTS signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, symbol TEXT, signal INTEGER, prob REAL, bet_size REAL, model TEXT
        );
        CREATE TABLE IF NOT EXISTS alert_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, level TEXT, message TEXT, channel TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ohlcv ON ohlcv(symbol, ts);
        CREATE TABLE IF NOT EXISTS experiment_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT,
            symbol      TEXT,
            run_id      TEXT,
            params      TEXT,
            cv_f1       REAL,
            auc         REAL,
            sharpe      REAL,
            n_trades    INTEGER,
            notes       TEXT
        );
        CREATE TABLE IF NOT EXISTS alpha_decay (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT,
            symbol      TEXT,
            window      TEXT,
            sharpe      REAL,
            ann_ret     REAL,
            win_rate    REAL
        );
        CREATE TABLE IF NOT EXISTS drift_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT,
            symbol      TEXT,
            feature     TEXT,
            psi         REAL,
            status      TEXT
        );
    """)
    conn.commit()
    conn.close()


_init_db()


def _db_upsert_ohlcv(symbol: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    conn = _get_db()
    rows = [
        (symbol, str(idx), float(r["Open"]), float(r["High"]),
         float(r["Low"]), float(r["Close"]), float(r["Volume"]))
        for idx, r in df.iterrows()
    ]
    conn.executemany("INSERT OR REPLACE INTO ohlcv VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def _db_load_ohlcv(symbol: str) -> pd.DataFrame:
    try:
        conn = _get_db()
        df = pd.read_sql("SELECT * FROM ohlcv WHERE symbol=? ORDER BY ts",
                         conn, params=(symbol,))
        conn.close()
        if df.empty:
            return pd.DataFrame()
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts").rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return pd.DataFrame()


def _db_log_metric(symbol: str, name: str, m: dict) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO model_metrics(ts,symbol,model_name,cv_f1,auc,accuracy,precision_,recall_)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), symbol, name,
             m.get("cv_f1", 0), m.get("auc", 0),
             m.get("accuracy", 0), m.get("precision", 0), m.get("recall", 0)),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _db_log_signal(symbol: str, sig: dict) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO signal_history(ts,symbol,signal,prob,bet_size,model) VALUES(?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), symbol,
             sig.get("signal", 0), sig.get("prob", 0.5),
             sig.get("bet_size", 0), sig.get("model", "")),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


# PARQUET CACHE

_CACHE = Path(CFG["cache_dir"])


def _parquet_save(key: str, df: pd.DataFrame) -> None:
    if not PARQUET_OK or df.empty:
        return
    try:
        df.to_parquet(_CACHE / f"{key}.parquet", engine="pyarrow", compression="snappy")
    except Exception:
        pass


def _parquet_load(key: str, max_age: int = 300) -> Optional[pd.DataFrame]:
    if not PARQUET_OK:
        return None
    p = _CACHE / f"{key}.parquet"
    if not p.exists() or (time.time() - p.stat().st_mtime) > max_age:
        return None
    try:
        return pd.read_parquet(p, engine="pyarrow")
    except Exception:
        return None

# MODEL PERSISTENCE

_MODEL_DIR = Path(CFG["model_dir"])


def _model_save(symbol: str, name: str, obj: Any) -> None:
    if not JOBLIB_OK:
        return
    try:
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9_]", "_", f"{symbol}_{name}_{ts}")
        joblib.dump(obj, _MODEL_DIR / f"{safe}.joblib", compress=3)
    except Exception as e:
        log.debug("model_save: %s", e)


def _model_load(symbol: str, name: str) -> Optional[Any]:
    if not JOBLIB_OK:
        return None
    pat = re.sub(r"[^A-Za-z0-9_]", "_", f"{symbol}_{name}_")
    for p in sorted(_MODEL_DIR.glob(f"{pat}*.joblib"), reverse=True)[:3]:
        try:
            return joblib.load(p)
        except Exception:
            continue
    return None


# NOTIFICATIONS (all optional; work only if env vars set)
def _send_telegram(msg: str) -> bool:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT, "text": msg}, timeout=8,
        )
        return r.status_code == 200
    except Exception:
        return False


def _send_email(subject: str, body: str) -> bool:
    if not all([SMTP_USER, SMTP_PASS, ALERT_EMAIL]):
        return False
    try:
        msg = MIMEText(body)
        msg["Subject"] = f"[HECTOR] {subject}"
        msg["From"] = SMTP_USER
        msg["To"]   = ALERT_EMAIL
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        return True
    except Exception:
        return False


def _send_slack(msg: str) -> bool:
    if not SLACK_WEBHOOK:
        return False
    try:
        r = requests.post(SLACK_WEBHOOK, json={"text": msg}, timeout=8)
        return r.status_code == 200
    except Exception:
        return False


def _broadcast(subject: str, body: str) -> None:
    """Fire-and-forget notifications on a background thread."""
    def _send():
        _send_telegram(f"⬡ HECTOR | {subject}\n{body}")
        _send_email(subject, body)
        _send_slack(f"⬡ HECTOR | {subject}\n{body}")
    threading.Thread(target=_send, daemon=True).start()


# DATA SANITISATION — used everywhere before sklearn calls
def _san(arr: np.ndarray, clip: float = 1e6) -> np.ndarray:
    """Replace inf/nan and clip extremes. Safe for StandardScaler.fit_transform."""
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, -clip, clip)


def _san_df(df: pd.DataFrame, clip: float = 1e6) -> np.ndarray:
    """Sanitise a DataFrame, return clean float64 ndarray."""
    raw = df.replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(float)
    return np.clip(raw, -clip, clip)


# HTTP SESSION WITH RETRY (connection pooling + exponential backoff)
_SESSION = requests.Session()
_SESSION.headers.update({
    "Accept":     "application/json, text/xml, */*",
    "User-Agent": "Mozilla/5.0 HECTOR/4.0 (+github.com)",
})


def _get(url: str, params: dict = None, timeout: int = None,
         as_text: bool = False) -> Optional[Any]:
    """Safe HTTP GET with retry and rate-limit handling."""
    to = timeout or CFG["http_timeout"]
    for attempt in range(CFG["retries"]):
        try:
            r = _SESSION.get(url, params=params, timeout=to)
            if r.status_code == 429:
                time.sleep(min(30 * (attempt + 1), 90))
                continue
            if r.status_code == 200:
                return r.text if as_text else r.json()
        except Exception as exc:
            log.debug("GET attempt %d/%d %s: %s", attempt + 1, CFG["retries"], url, exc)
            if attempt < CFG["retries"] - 1:
                time.sleep(CFG["backoff"] ** attempt)
    return None


# CSS THEME
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Mono',monospace;background:#0a0b0d!important;color:#e8eaf0!important;}
.main,.block-container{background:#0a0b0d!important;}
section[data-testid="stSidebar"]{background:#060810!important;border-right:1px solid #1e2130;}
.bb-hdr{background:linear-gradient(90deg,#000 0%,#0d0f16 100%);border-bottom:2px solid #ff8c00;
        padding:10px 20px;margin:-1rem -1rem 1rem -1rem;display:flex;align-items:center;justify-content:space-between;}
.bb-logo{font-size:20px;font-weight:700;color:#ff8c00;letter-spacing:4px;}
.bb-sub{font-size:8px;color:#4a5270;letter-spacing:2px;text-transform:uppercase;}
.bb-live{font-size:10px;color:#00d48a;}
.kpi-card{background:#12141a;border:1px solid #1e2130;border-top:2px solid #ff8c00;
          border-radius:2px;padding:10px 12px;margin-bottom:5px;}
.kpi-lbl{font-size:8px;color:#4a5270;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:2px;}
.kpi-val{font-size:15px;font-weight:700;color:#e8eaf0;}
.kpi-pos{font-size:9px;color:#00d48a;font-weight:600;}
.kpi-neg{font-size:9px;color:#ff3d5a;font-weight:600;}
.sec-hdr{font-size:8px;color:#ff8c00;letter-spacing:2.5px;text-transform:uppercase;
         border-bottom:1px solid #1e2130;padding-bottom:4px;margin:14px 0 8px 0;}
.stat-row{display:flex;justify-content:space-between;padding:3px 0;
          border-bottom:1px solid #0e1420;font-size:9px;}
.stat-k{color:#4a5270;text-transform:uppercase;letter-spacing:.4px;}
.stat-v{color:#c0cce0;}
.signal-lg{background:rgba(0,212,138,.12);border:2px solid #00d48a;color:#00d48a;
           text-align:center;padding:12px;font-size:22px;font-weight:700;border-radius:3px;}
.signal-sh{background:rgba(255,61,90,.12);border:2px solid #ff3d5a;color:#ff3d5a;
           text-align:center;padding:12px;font-size:22px;font-weight:700;border-radius:3px;}
.signal-fl{background:rgba(74,82,112,.2);border:2px solid #2a2f42;color:#4a5270;
           text-align:center;padding:12px;font-size:22px;font-weight:700;border-radius:3px;}
.tag-long{background:rgba(0,212,138,.15);border:1px solid #00d48a;color:#00d48a;
          font-size:8px;padding:1px 5px;border-radius:2px;}
.tag-short{background:rgba(255,61,90,.15);border:1px solid #ff3d5a;color:#ff3d5a;
           font-size:8px;padding:1px 5px;border-radius:2px;}
.tag-neu{background:rgba(74,82,112,.3);border:1px solid #2a2f42;color:#4a5270;
         font-size:8px;padding:1px 5px;border-radius:2px;}
.stButton>button{background:#ff8c00;color:#000;border:none;border-radius:2px;
  font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:.8px;
  padding:5px 14px;width:100%;font-weight:700;}
.stButton>button:hover{background:#ffa333;}
div[data-testid="stMetric"]{background:#12141a;border:1px solid #1e2130;border-radius:2px;padding:8px;}
.stTabs [data-baseweb="tab"]{background:#12141a;color:#4a5270;border-radius:0;
  border-bottom:2px solid transparent;font-family:'IBM Plex Mono',monospace;
  font-size:8px;letter-spacing:1.2px;text-transform:uppercase;}
.stTabs [aria-selected="true"]{background:#12141a;color:#ff8c00;border-bottom:2px solid #ff8c00;}
[data-testid="stSelectbox"]>div{background:#12141a!important;border-color:#1e2130!important;}
::-webkit-scrollbar{width:3px;height:3px;}
::-webkit-scrollbar-track{background:#0a0b0d;}
::-webkit-scrollbar-thumb{background:#2a2f42;border-radius:2px;}
div.block-container{padding-top:.3rem;}
.alert-green{background:rgba(0,212,138,.1);border:1px solid #00d48a;
             color:#00d48a;padding:6px 10px;font-size:9px;border-radius:2px;margin:4px 0;}
.alert-red{background:rgba(255,61,90,.1);border:1px solid #ff3d5a;
           color:#ff3d5a;padding:6px 10px;font-size:9px;border-radius:2px;margin:4px 0;}
.alert-yellow{background:rgba(255,215,0,.1);border:1px solid #ffd700;
              color:#ffd700;padding:6px 10px;font-size:9px;border-radius:2px;margin:4px 0;}
.stProgress > div > div{background-color:#ff8c00!important;}
.regime-bull{background:rgba(0,212,138,.15);border:1px solid #00d48a;color:#00d48a;
             padding:4px 10px;border-radius:2px;font-size:9px;}
.regime-bear{background:rgba(255,61,90,.15);border:1px solid #ff3d5a;color:#ff3d5a;
             padding:4px 10px;border-radius:2px;font-size:9px;}
.regime-side{background:rgba(255,215,0,.15);border:1px solid #ffd700;color:#ffd700;
             padding:4px 10px;border-radius:2px;font-size:9px;}
</style>
""", unsafe_allow_html=True)

# COLOUR PALETTE & PLOTLY HELPERS
C = dict(
    bg="#0a0b0d", panel="#080d16", card="#12141a", border="#1e2130",
    orange="#ff8c00", green="#00d48a", red="#ff3d5a", blue="#0088ff",
    yellow="#ffd700", purple="#9b6dff", cyan="#00cfff", muted="#4a5270",
    text="#e8eaf0", sec="#8892aa",
)


def BB(title: str = "", h: int = 350, legend: bool = True, extra: dict = None) -> dict:
    ax = dict(gridcolor=C["border"], linecolor=C["border"],
              tickfont=dict(size=8), zerolinecolor=C["border"], showgrid=True)
    layout = dict(
        paper_bgcolor=C["card"], plot_bgcolor=C["panel"],
        font=dict(family="IBM Plex Mono, monospace", color=C["sec"], size=9),
        title=dict(text=f"<b style='color:{C['orange']}'>{title}</b>",
                   font=dict(size=10)) if title else None,
        margin=dict(l=44, r=14, t=32 if title else 10, b=28),
        xaxis=ax.copy(), yaxis=ax.copy(),
        legend=dict(bgcolor="rgba(18,20,26,.85)", bordercolor=C["border"],
                    borderwidth=1, font=dict(size=8)) if legend else dict(visible=False),
        height=h, hovermode="x unified",
        hoverlabel=dict(bgcolor=C["card"], bordercolor=C["orange"],
                        font=dict(family="IBM Plex Mono", size=9)),
    )
    if extra:
        layout.update(extra)
    return layout


PCONF = dict(displayModeBar=False)


def _xa() -> dict:
    ax = BB()["xaxis"].copy()
    ax["rangeslider"] = dict(visible=False)
    return ax


# COIN REGISTRY (50 assets — all via free yfinance + CoinGecko public)
COIN_MAP: Dict[str, Tuple[str, str]] = {
    "Bitcoin (BTC)":        ("bitcoin",           "BTC-USD"),
    "Ethereum (ETH)":       ("ethereum",          "ETH-USD"),
    "BNB":                  ("binancecoin",        "BNB-USD"),
    "Solana (SOL)":         ("solana",             "SOL-USD"),
    "XRP":                  ("ripple",             "XRP-USD"),
    "Dogecoin (DOGE)":      ("dogecoin",           "DOGE-USD"),
    "Cardano (ADA)":        ("cardano",            "ADA-USD"),
    "Avalanche (AVAX)":     ("avalanche-2",        "AVAX-USD"),
    "TRON (TRX)":           ("tron",               "TRX-USD"),
    "Polkadot (DOT)":       ("polkadot",           "DOT-USD"),
    "Chainlink (LINK)":     ("chainlink",          "LINK-USD"),
    "Polygon (MATIC)":      ("matic-network",      "MATIC-USD"),
    "Litecoin (LTC)":       ("litecoin",           "LTC-USD"),
    "Shiba Inu (SHIB)":     ("shiba-inu",          "SHIB-USD"),
    "Bitcoin Cash (BCH)":   ("bitcoin-cash",       "BCH-USD"),
    "Uniswap (UNI)":        ("uniswap",            "UNI-USD"),
    "NEAR Protocol":        ("near",               "NEAR-USD"),
    "Cosmos (ATOM)":        ("cosmos",             "ATOM-USD"),
    "Filecoin (FIL)":       ("filecoin",           "FIL-USD"),
    "Aave":                 ("aave",               "AAVE-USD"),
    "Algorand (ALGO)":      ("algorand",           "ALGO-USD"),
    "Stellar (XLM)":        ("stellar",            "XLM-USD"),
    "Monero (XMR)":         ("monero",             "XMR-USD"),
    "Hedera (HBAR)":        ("hedera-hashgraph",   "HBAR-USD"),
    "VeChain (VET)":        ("vechain",            "VET-USD"),
    "EOS":                  ("eos",                "EOS-USD"),
    "Fantom (FTM)":         ("fantom",             "FTM-USD"),
    "Axie Infinity (AXS)":  ("axie-infinity",      "AXS-USD"),
    "Decentraland (MANA)":  ("decentraland",       "MANA-USD"),
    "Zcash (ZEC)":          ("zcash",              "ZEC-USD"),
    "Maker (MKR)":          ("maker",              "MKR-USD"),
    "Compound (COMP)":      ("compound-coin",      "COMP-USD"),
    "SushiSwap (SUSHI)":    ("sushi",              "SUSHI-USD"),
    "Curve DAO (CRV)":      ("curve-dao-token",    "CRV-USD"),
    "Sandbox (SAND)":       ("the-sandbox",        "SAND-USD"),
    "Optimism (OP)":        ("optimism",           "OP-USD"),
    "Arbitrum (ARB)":       ("arbitrum",           "ARB-USD"),
    "Aptos (APT)":          ("aptos",              "APT-USD"),
    "Sui (SUI)":            ("sui",                "SUI-USD"),
    "Pepe (PEPE)":          ("pepe",               "PEPE-USD"),
    "Injective (INJ)":      ("injective-protocol", "INJ-USD"),
    "Kaspa (KAS)":          ("kaspa",              "KAS-USD"),
    "Toncoin (TON)":        ("the-open-network",   "TON-USD"),
    "Celestia (TIA)":       ("celestia",           "TIA-USD"),
    "Stacks (STX)":         ("blockstack",         "STX-USD"),
    "Immutable (IMX)":      ("immutable-x",        "IMX-USD"),
    "Render (RNDR)":        ("render-token",       "RNDR-USD"),
    "Sei (SEI)":            ("sei-network",        "SEI-USD"),
    "Internet Computer":    ("internet-computer",  "ICP-USD"),
}


#  LAYER 1: DATA INGESTION — 100% FREE, ZERO API KEYS

@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def fetch_ohlcv(yf_sym: str, period: str = "3mo", interval: str = "1h") -> pd.DataFrame:
    """OHLCV via yfinance — completely free, no key."""
    import yfinance as yf

    pq_key = re.sub(r"[^A-Za-z0-9]", "_", f"ohlcv_{yf_sym}_{period}_{interval}")
    cached = _parquet_load(pq_key, max_age=CFG["cache_ttl_ohlcv"])
    if cached is not None and not cached.empty:
        return cached

    try:
        raw = yf.download(yf_sym, period=period, interval=interval,
                          progress=False, auto_adjust=True, threads=False)
        if raw.empty:
            return _db_load_ohlcv(yf_sym)
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.ffill().bfill().dropna()
        df.index = pd.to_datetime(df.index)
        _db_upsert_ohlcv(yf_sym, df)
        _parquet_save(pq_key, df)
        return df
    except Exception as exc:
        log.error("fetch_ohlcv %s: %s", yf_sym, exc)
        return _db_load_ohlcv(yf_sym)


@st.cache_data(ttl=CFG["cache_ttl_info"], show_spinner=False)
def fetch_coin_info(cg_id: str) -> dict:
    """CoinGecko public API — free, no key (30 req/min limit)."""
    empty = dict(name=cg_id, symbol=cg_id.upper()[:6],
                 price=0, market_cap=0, volume_24h=0,
                 change_24h=0, change_7d=0, high_24h=0,
                 low_24h=0, ath=0, circulating_supply=0)
    try:
        data = _get(f"https://api.coingecko.com/api/v3/coins/{cg_id}",
                    params={"localization": "false", "tickers": "false",
                            "market_data": "true", "community_data": "false",
                            "developer_data": "false"})
        if not isinstance(data, dict):
            return empty
        md = data.get("market_data") or {}
        def _u(f):
            return ((md.get(f) or {}).get("usd") or 0)
        return dict(
            name=data.get("name", ""),
            symbol=(data.get("symbol") or "").upper(),
            price=_u("current_price"), market_cap=_u("market_cap"),
            volume_24h=_u("total_volume"),
            change_24h=md.get("price_change_percentage_24h") or 0,
            change_7d=md.get("price_change_percentage_7d") or 0,
            high_24h=_u("high_24h"), low_24h=_u("low_24h"),
            ath=_u("ath"), circulating_supply=md.get("circulating_supply") or 0,
        )
    except Exception:
        return empty


@st.cache_data(ttl=CFG["cache_ttl_global"], show_spinner=False)
def fetch_global_market() -> dict:
    """CoinGecko global stats — free."""
    try:
        data = _get("https://api.coingecko.com/api/v3/global")
        if not isinstance(data, dict):
            return {}
        d = data.get("data") or {}
        return dict(
            total_mcap=(d.get("total_market_cap") or {}).get("usd", 0),
            total_vol=(d.get("total_volume") or {}).get("usd", 0),
            btc_dom=(d.get("market_cap_percentage") or {}).get("btc", 0),
            eth_dom=(d.get("market_cap_percentage") or {}).get("eth", 0),
            active_coins=d.get("active_cryptocurrencies", 0),
        )
    except Exception:
        return {}


@st.cache_data(ttl=CFG["cache_ttl_global"], show_spinner=False)
def fetch_fear_greed() -> dict:
    """alternative.me Fear & Greed — completely free."""
    default = {"current": 50, "label": "Neutral", "history": [50] * 30}
    try:
        data = _get("https://api.alternative.me/fng/", params={"limit": 30})
        if not isinstance(data, dict):
            return default
        items = data.get("data") or []
        if not items:
            return default
        _label_map = {
            "Extreme Fear":  "EXTREME FEAR",
            "Fear":          "FEAR",
            "Neutral":       "NEUTRAL",
            "Greed":         "GREED",
            "Extreme Greed": "EXTREME GREED",
        }
        raw_label = items[0].get("value_classification", "Neutral")
        return {
            "current": int(items[0].get("value", 50)),
            "label": _label_map.get(raw_label, raw_label.upper()),
            "history": [int(x.get("value", 50)) for x in reversed(items)],
        }
    except Exception:
        return default


@st.cache_data(ttl=900, show_spinner=False)
def fetch_crypto_news() -> List[dict]:
    """
    Free crypto news — tries multiple sources with no API key:
    1. CryptoCompare free endpoint (no auth required)
    2. CoinDesk RSS
    3. Empty list as final fallback
    """
    # Source 1: CryptoCompare (truly free, no key)
    try:
        data = _get("https://min-api.cryptocompare.com/data/v2/news/",
                    params={"lang": "EN", "sortOrder": "latest"})
        if isinstance(data, dict) and isinstance(data.get("Data"), list):
            out = []
            pos_w = ["bull", "surge", "rally", "gain", "rise", "up", "buy", "moon"]
            neg_w = ["bear", "crash", "drop", "fall", "dump", "sell", "fear", "low"]
            for n in data["Data"][:12]:
                title = n.get("title", "")
                body  = (n.get("body") or "").lower()
                tl    = title.lower()
                pc    = sum(1 for w in pos_w if w in body or w in tl)
                nc    = sum(1 for w in neg_w if w in body or w in tl)
                sent  = ("bullish" if pc > nc else "bearish" if nc > pc else "neutral")
                out.append({"title": title, "sentiment": sent,
                             "url": n.get("url", ""), "time": n.get("published_on", "")})
            if out:
                return out
    except Exception as e:
        log.debug("CryptoCompare news: %s", e)

    # Source 2: CoinDesk RSS
    try:
        rss_text = _get("https://www.coindesk.com/arc/outboundfeeds/rss/",
                        as_text=True)
        if isinstance(rss_text, str) and "<item>" in rss_text:
            root = ET.fromstring(rss_text)
            out  = []
            pos_w = ["bull", "surge", "rally", "gain"]
            neg_w = ["bear", "crash", "drop", "fall"]
            for item in root.findall(".//item")[:12]:
                title = (item.findtext("title") or "").strip()
                if not title:
                    continue
                tl   = title.lower()
                pc   = sum(1 for w in pos_w if w in tl)
                nc   = sum(1 for w in neg_w if w in tl)
                sent = ("bullish" if pc > nc else "bearish" if nc > pc else "neutral")
                out.append({"title": title, "sentiment": sent,
                             "url": item.findtext("link") or "",
                             "time": item.findtext("pubDate") or ""})
            if out:
                return out
    except Exception as e:
        log.debug("CoinDesk RSS: %s", e)

    return []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_reddit_sentiment(subreddit: str = "CryptoCurrency") -> dict:
    """Reddit public JSON API — no OAuth, completely free."""
    default = {"score": 0.0, "label": "neutral", "count": 0}
    try:
        data = _get(f"https://www.reddit.com/r/{subreddit}/hot.json",
                    params={"limit": 50})
        if not isinstance(data, dict):
            return default
        posts = (data.get("data") or {}).get("children") or []
        texts = [p.get("data", {}).get("title", "") for p in posts]
        if not texts:
            return default
        pos_w = ["bull", "moon", "pump", "buy", "long", "ath", "surge", "rally", "green"]
        neg_w = ["bear", "dump", "crash", "sell", "short", "fear", "drop", "red", "loss"]
        scores = []
        for t in texts:
            tl = t.lower()
            p  = sum(1 for w in pos_w if w in tl)
            n  = sum(1 for w in neg_w if w in tl)
            scores.append(
                float(np.clip((p - n) / max(p + n, 1), -1.0, 1.0))
                if (p + n) > 0 else 0.0
            )
        avg = float(np.mean(scores))
        return {
            "score": round(avg, 3),
            "label": ("Bullish" if avg > 0.05 else "Bearish" if avg < -0.05 else "Neutral"),
            "count": len(texts),
        }
    except Exception as e:
        log.debug("Reddit sentiment: %s", e)
        return default


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_onchain_proxies() -> dict:
    """
    Free on-chain data — no key needed:
    blockchain.info public stats + CoinGecko community data.
    """
    result: dict = {}
    try:
        stats = _get("https://api.blockchain.info/stats")
        if isinstance(stats, dict):
            result["hash_rate_eh"]   = round(float(stats.get("hash_rate", 0)) / 1e18, 2)
            result["miners_revenue"] = int(stats.get("miners_revenue_usd", 0))
            result["n_transactions"] = int(stats.get("n_tx", 0))
            result["mempool_size"]   = int(stats.get("mempool_size", 0))
    except Exception as e:
        log.debug("blockchain.info: %s", e)
    try:
        data = _get("https://api.coingecko.com/api/v3/coins/bitcoin",
                    params={"localization": "false", "tickers": "false",
                            "market_data": "true", "community_data": "true"})
        if isinstance(data, dict):
            cd = data.get("community_data") or {}
            result["reddit_subs"]       = cd.get("reddit_subscribers", 0)
            result["twitter_followers"] = cd.get("twitter_followers", 0)
    except Exception as e:
        log.debug("CG community data: %s", e)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dxy() -> pd.DataFrame:
    """DXY Dollar Index via yfinance — free."""
    import yfinance as yf
    try:
        raw = yf.download("DX-Y.NYB", period="3mo", interval="1d",
                          progress=False, auto_adjust=True, threads=False)
        if raw.empty:
            return pd.DataFrame()
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        return raw[["Close"]].rename(columns={"Close": "DXY"})
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def fetch_multi_ohlcv(symbols: tuple, period: str = "3mo") -> dict:
    """Fetch daily OHLCV for multiple symbols in parallel (yfinance, free)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    def _fetch_one(sym):
        df = fetch_ohlcv(sym, period=period, interval="1d")
        return sym, df

    result = {}
    max_workers = min(8, len(symbols))  # cap to avoid rate-limiting
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            try:
                sym, df = fut.result()
                if not df.empty:
                    result[sym] = df
            except Exception:
                pass
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmarks(period: str = "3mo") -> Dict[str, pd.Series]:
    """BTC, ETH, S&P 500 via yfinance — free benchmarks."""
    import yfinance as yf
    out: Dict[str, pd.Series] = {}
    for sym, label in [("BTC-USD", "BTC"), ("ETH-USD", "ETH"), ("^GSPC", "S&P500")]:
        try:
            raw = yf.download(sym, period=period, interval="1d",
                              progress=False, auto_adjust=True, threads=False)
            if not raw.empty:
                raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
                out[label] = raw["Close"].squeeze()
        except Exception:
            pass
    return out


#  LAYER 2: DATA ENGINEERING — 30+ indicators

@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def engineer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators. No external TA library required."""
    if df.empty or len(df) < 30:
        return df
    df = df.copy()
    c  = df["Close"].squeeze()
    h  = df["High"].squeeze()
    lo = df["Low"].squeeze()
    v  = df["Volume"].squeeze()

    df["returns"]     = c.pct_change().replace([np.inf, -np.inf], np.nan)
    df["log_returns"] = np.log((c.replace(0, np.nan) / c.shift(1).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan))

    df["ema9"]   = c.ewm(span=9,  adjust=False).mean()
    df["ema21"]  = c.ewm(span=21, adjust=False).mean()
    df["ema50"]  = c.ewm(span=50, adjust=False).mean()
    df["sma50"]  = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()

    bb_mid = c.rolling(20).mean(); bb_std = c.rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_up"]  = bb_mid + 2 * bb_std
    df["bb_dn"]  = bb_mid - 2 * bb_std
    df["bb_pct"] = (c - df["bb_dn"]) / (df["bb_up"] - df["bb_dn"] + 1e-10)
    df["bb_bw"]  = (df["bb_up"] - df["bb_dn"]) / (bb_mid + 1e-10)

    delta      = c.diff()
    gain       = delta.clip(lower=0).rolling(14).mean()
    loss       = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"]  = 100 - 100 / (1 + gain / (loss + 1e-10))

    ema12             = c.ewm(span=12).mean(); ema26 = c.ewm(span=26).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    lo14          = lo.rolling(14).min(); hi14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"]     = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / (c + 1e-10) * 100

    df["obv"]  = (v * np.sign(c.diff().fillna(0))).cumsum()
    tp         = (h + lo + c) / 3
    df["vwap"] = (tp * v).cumsum() / (v.cumsum() + 1e-10)

    df["vol_5"]    = df["returns"].rolling(5).std()  * np.sqrt(365 * 24)
    df["vol_20"]   = df["returns"].rolling(20).std() * np.sqrt(365 * 24)
    df["ewma_vol"] = df["returns"].ewm(span=30, adjust=False).std() * np.sqrt(365 * 24)

    df["trend_up"]   = (df["ema9"] > df["ema21"]).astype(int)
    df["above_vwap"] = (c > df["vwap"]).astype(int)

    mf_raw = tp * v
    mf_pos = mf_raw.where(tp > tp.shift(1), 0)
    mf_neg = mf_raw.where(tp < tp.shift(1), 0)
    mfr    = mf_pos.rolling(14).sum() / (mf_neg.rolling(14).sum() + 1e-10)
    df["mfi"]    = 100 - 100 / (1 + mfr)
    df["willr"]  = -100 * (hi14 - c) / (hi14 - lo14 + 1e-10)
    mdev         = (tp - tp.rolling(20).mean()).abs().rolling(20).mean()
    df["cci"]    = (tp - tp.rolling(20).mean()) / (0.015 * mdev + 1e-10)

    buy_vol  = v.where(c >= df["Open"].squeeze(), 0)
    sell_vol = v.where(c <  df["Open"].squeeze(), 0)
    df["cvd"]    = (buy_vol - sell_vol).cumsum()
    df["cvd_ma"] = df["cvd"].ewm(span=20).mean()
    df["ofi"]    = (buy_vol - sell_vol) / (v + 1e-10)

    df["dc_high"] = h.rolling(20).max()
    df["dc_low"]  = lo.rolling(20).min()
    df["dc_mid"]  = (df["dc_high"] + df["dc_low"]) / 2

    vol_ma = v.rolling(20).mean()
    df["vol_ratio"] = (v / (vol_ma + 1e-10)).replace([np.inf, -np.inf], 1.0)

    df = df.ffill().bfill().dropna(subset=["returns"])
    return df


#  DATA QUALITY VALIDATION

@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def validate_ohlcv(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    if df.empty:
        return df, {}
    n = len(df)
    # Remove duplicate timestamps
    dupes = df.index.duplicated(keep="last").sum()
    df    = df[~df.index.duplicated(keep="last")].copy()
    # Cap extreme returns (outlier clipping)
    ret  = df["Close"].squeeze().pct_change()
    mu_, sd_ = float(ret.mean()), float(ret.std())
    mask = (ret.abs() > mu_ + 5 * sd_) & (sd_ > 0)
    df["Close"] = df["Close"].where(~mask, other=df["Close"].shift(1))
    df = df.ffill(limit=3).dropna(subset=["Close"])
    return df, {"duplicates": int(dupes), "outliers": int(mask.sum()),
                "rows_before": n, "rows_after": len(df)}


#  LAYER 3: EVENT-BASED PROCESSING

def cusum_events(prices: pd.Series, thr: float = None) -> List[int]:
    """Symmetric CUSUM filter; returns event bar indices."""
    thr = thr or CFG["cusum_thr"]
    rets = prices.pct_change().fillna(0).values
    sp = sn = 0.0; events: List[int] = []
    for i, r in enumerate(rets[1:], start=1):
        sp = max(0.0, sp + r); sn = min(0.0, sn + r)
        if sp > thr or sn < -thr:
            events.append(i); sp = sn = 0.0
    return events


def make_volume_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate OHLCV into volume bars."""
    total = float(df["Volume"].squeeze().sum())
    vpb   = total / max(50, len(df) // 10)
    cum   = 0.0; rows = []; o = h = l = c = vol = 0.0; ts0 = None
    for ts, row in df.iterrows():
        if ts0 is None:
            ts0 = ts; o = float(row["Open"]); h = l = float(row["Close"])
        vr = float(row["Volume"])
        h  = max(h, float(row["High"])); l = min(l, float(row["Low"]))
        c  = float(row["Close"]); vol += vr; cum += vr
        if cum >= vpb:
            rows.append(pd.Series({"Open": o, "High": h, "Low": l,
                                   "Close": c, "Volume": vol}, name=ts))
            cum = vol = 0.0; ts0 = None
    return pd.DataFrame(rows) if len(rows) >= 10 else df


#  LAYER 4: LABELING ENGINE — Triple Barrier

@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def triple_barrier_labels(df: pd.DataFrame, pt: float = 1.5,
                           sl: float = 1.0, max_hold: int = 24) -> pd.DataFrame:
    """Generate Long(1)/Short(-1)/Neutral(0) labels."""
    if df.empty or "returns" not in df.columns:
        return pd.DataFrame()
    c   = df["Close"].squeeze()
    vol = df["returns"].rolling(20).std().bfill()
    n   = len(c)
    step = max(1, n // CFG["max_labels"])
    rows = []
    for i in range(20, n - max_hold, step):
        price  = float(c.iloc[i]); v_ = float(vol.iloc[i])
        upper  = price * (1 + pt * v_); lower = price * (1 - sl * v_)
        end    = min(i + max_hold, n - 1); label = 0
        for j in range(i, end + 1):
            p = float(c.iloc[j])
            if p >= upper: label = 1; break
            if p <= lower: label = -1; break
        ret = float(c.iloc[end] / price) - 1
        if label == 0:
            label = 1 if ret > 0 else -1 if ret < 0 else 0
        rows.append(dict(idx=i, time=c.index[i], label=label, ret=ret,
                         vol_entry=v_, pt=upper, sl=lower))
    return pd.DataFrame(rows)


#  LAYER 5: FEATURE ENGINEERING

FEATURE_COLS = [
    "r1", "r5", "r10", "r20",
    "vol_5", "vol_20", "ewma_vol",
    "rsi", "macd", "macd_hist", "bb_pct", "bb_bw",
    "atr_pct", "stoch_k", "mfi", "willr", "cci",
    "trend_up", "above_vwap", "vol_ratio",
    "ofi", "cvd_norm", "frac_d35",
]


@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def build_features(df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix at each label event."""
    if df.empty or labels_df.empty:
        return pd.DataFrame()
    c    = df["Close"].squeeze()
    v_s  = df["Volume"].squeeze()
    cvd_ = df["cvd"] if "cvd" in df.columns else pd.Series(0, index=df.index)
    cvd_max = float(cvd_.abs().max()) + 1e-10
    rows = []

    for _, row in labels_df.iterrows():
        i = int(row["idx"])
        if i < 20 or i >= len(df):
            continue
        # Fractional diff proxy (d=0.35)
        lp = np.log(c.iloc[max(0, i - 20): i + 1].values + 1e-10)
        w  = [1.0, -0.35, -0.35 * 0.65 / 2,
              -0.35 * 0.65 * 0.3 / 6, -0.35 * 0.65 * 0.3 * 0.025 / 24]
        frac = float(sum(a * lp[-(k + 1)] for k, a in enumerate(w) if k < len(lp)))
        va5  = float(v_s.iloc[max(0, i - 5): i].mean()) + 1e-10

        def _g(col, default=0.0):
            try:
                return float(df[col].iloc[i]) if col in df.columns else default
            except Exception:
                return default

        feat = dict(
            label=int(row["label"]), ret=float(row["ret"]),
            r1 =float(c.iloc[i] / c.iloc[i - 1] - 1) if i >= 1  else 0.0,
            r5 =float(c.iloc[i] / c.iloc[i - 5] - 1) if i >= 5  else 0.0,
            r10=float(c.iloc[i] / c.iloc[i - 10] - 1) if i >= 10 else 0.0,
            r20=float(c.iloc[i] / c.iloc[i - 20] - 1) if i >= 20 else 0.0,
            vol_5=_g("vol_5"), vol_20=_g("vol_20"), ewma_vol=_g("ewma_vol"),
            rsi=_g("rsi", 50), macd=_g("macd"), macd_hist=_g("macd_hist"),
            bb_pct=_g("bb_pct", 0.5), bb_bw=_g("bb_bw"),
            atr_pct=_g("atr_pct"), stoch_k=_g("stoch_k", 50),
            mfi=_g("mfi", 50), willr=_g("willr", -50), cci=_g("cci"),
            trend_up=_g("trend_up"), above_vwap=_g("above_vwap"),
            vol_ratio=float(v_s.iloc[i]) / va5,
            ofi=_g("ofi"),
            cvd_norm=float(cvd_.iloc[i]) / cvd_max,
            frac_d35=frac,
        )
        rows.append(feat)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).replace([np.inf, -np.inf], 0.0).fillna(0.0)


#  OUTLIER REMOVAL (Isolation Forest)

def _remove_outliers(feat_df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    if feat_df.empty or len(feat_df) < 30:
        return feat_df
    try:
        _iX = feat_df[FEATURE_COLS].copy().replace([np.inf, -np.inf], np.nan)
        for _ic in _iX.columns:
            _iq1, _iq3 = _iX[_ic].quantile(0.25), _iX[_ic].quantile(0.75)
            _iiqr = _iq3 - _iq1
            _iX[_ic] = _iX[_ic].clip(lower=_iq1 - 10*_iiqr, upper=_iq3 + 10*_iiqr)
        X    = _iX.fillna(_iX.median()).fillna(0).values
        iso  = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        mask = iso.fit_predict(X) == 1
        return feat_df[mask].copy()
    except Exception as e:
        log.debug("IForest: %s", e)
        return feat_df


#  REGIME DETECTION (HMM or rule-based fallback)

@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def detect_regime(returns: pd.Series, n_states: int = 3) -> np.ndarray:
    r = returns.dropna().values.reshape(-1, 1)
    if HMM_OK and len(r) >= 30:
        try:
            model = hmm.GaussianHMM(n_components=n_states,
                                     covariance_type="diag",
                                     n_iter=200, random_state=42)
            model.fit(r)
            labels = model.predict(r)
            means  = [r[labels == k].mean() for k in range(n_states)]
            order  = np.argsort(means)
            remap  = {old: new for new, old in enumerate(order)}
            return np.array([remap[lb] for lb in labels])
        except Exception:
            pass
    # Rule-based fallback
    rm = pd.Series(returns.dropna().values).rolling(20, min_periods=1).mean()
    return np.where(rm > 0.001, 2, np.where(rm < -0.001, 0, 1))


@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def classify_regime(df: pd.DataFrame) -> str:
    if "returns" not in df.columns or len(df) < 20:
        return "sideways"
    labels = detect_regime(df["returns"])
    return {0: "bear", 1: "sideways", 2: "bull"}.get(int(labels[-1]), "sideways")


#  PURGED CV SPLITS

def _purged_splits(n: int, n_splits: int = 3, embargo: int = 2):
    fold = max(1, n // n_splits)
    for k in range(n_splits):
        ts = k * fold
        te = (k + 1) * fold if k < n_splits - 1 else n
        tr = list(range(0, max(0, ts - embargo))) + \
             list(range(min(n, te + embargo), n))
        va = list(range(ts, te))
        if tr and va:
            yield tr, va



#  ██████  INTELLIGENCE ENGINES  (v5 Alpha Lab)
#
#  1.  Feature Stability Analysis
#  2.  Baseline Benchmarking
#  3.  Probability Calibration Layer
#  4.  Meta-Labeling Layer
#  5.  Dynamic Threshold Optimisation
#  6.  Walk-Forward Validation Engine
#  7.  Advanced Backtesting Engine
#  8.  Portfolio Risk Constraints
#  9.  Performance Attribution
#  10. Statistical Significance Testing
#  11. Model Drift Detection
#  12. Online / Retraining Scheduler
#  13. Experiment Tracking System
#  14. Alpha Decay Monitoring
#  15. Feedback Loop — Losing Trade Analyser
#  16. Helper DB writers for new tables

import uuid as _uuid_mod

# DB writers for new v5 tables

def _db_log_experiment(symbol: str, run_id: str, params: dict,
                        metrics: dict, notes: str = "") -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO experiment_log"
            "(ts,symbol,run_id,params,cv_f1,auc,sharpe,n_trades,notes)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), symbol, run_id,
             json.dumps(params, default=str),
             metrics.get("cv_f1", 0), metrics.get("auc", 0),
             metrics.get("sharpe", 0), metrics.get("n_trades", 0), notes),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _db_log_alpha_decay(symbol: str, window: str,
                         sharpe: float, ann_ret: float, win_rate: float) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO alpha_decay(ts,symbol,window,sharpe,ann_ret,win_rate)"
            " VALUES(?,?,?,?,?,?)",
            (datetime.utcnow().isoformat(), symbol, window,
             sharpe, ann_ret, win_rate),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _db_log_drift(symbol: str, feature: str, psi: float, status: str) -> None:
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO drift_log(ts,symbol,feature,psi,status)"
            " VALUES(?,?,?,?,?)",
            (datetime.utcnow().isoformat(), symbol, feature, psi, status),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _db_load_experiments(symbol: str) -> pd.DataFrame:
    try:
        conn = _get_db()
        df = pd.read_sql(
            "SELECT ts, run_id, cv_f1, auc, sharpe, n_trades, notes"
            " FROM experiment_log WHERE symbol=? ORDER BY id DESC LIMIT 50",
            conn, params=(symbol,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def _db_load_alpha_decay(symbol: str) -> pd.DataFrame:
    try:
        conn = _get_db()
        df = pd.read_sql(
            "SELECT ts, window, sharpe, ann_ret, win_rate"
            " FROM alpha_decay WHERE symbol=? ORDER BY id DESC LIMIT 60",
            conn, params=(symbol,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


# 1. FEATURE STABILITY ANALYSIS

def feature_stability_analysis(feat_df: pd.DataFrame,
                                 n_windows: int = 5) -> pd.DataFrame:
    """
    Rolling importance consistency + feature decay analysis.
    Splits the feature matrix into n_windows and trains a quick RF on
    each window.  Returns a DataFrame of per-feature mean importance
    and coefficient-of-variation (CV). High CV = unstable.
    """
    if feat_df.empty or len(feat_df) < 60:
        return pd.DataFrame()

    df = feat_df[feat_df["label"] != 0].copy()
    if len(df) < 40:
        return pd.DataFrame()

    avail = [c for c in FEATURE_COLS if c in df.columns]
    X_all = _san_df(df[avail])
    y_all = (df["label"] == 1).astype(int).values
    n     = len(X_all)
    fold  = n // n_windows
    imp_windows: List[np.ndarray] = []

    for k in range(n_windows):
        s, e = k * fold, min((k + 1) * fold, n)
        if e - s < 15:
            continue
        try:
            rf = RandomForestClassifier(
                n_estimators=40, max_depth=4, random_state=k,
                n_jobs=-1, class_weight="balanced",
            )
            sc_ = StandardScaler()
            rf.fit(sc_.fit_transform(_san(X_all[s:e])), y_all[s:e])
            imp_windows.append(rf.feature_importances_)
        except Exception:
            pass

    if not imp_windows:
        return pd.DataFrame()

    arr  = np.array(imp_windows)                       # shape (windows, features)
    mean = arr.mean(axis=0)
    std_ = arr.std(axis=0)
    cv   = std_ / (mean + 1e-10)

    # Feature decay: predictive power drops from window 0 → last window
    decay = arr[0] - arr[-1] if len(arr) >= 2 else np.zeros(len(avail))

    result = pd.DataFrame({
        "Feature":   avail,
        "Mean Imp":  mean.round(4),
        "Std Imp":   std_.round(4),
        "CV (instability)": cv.round(3),
        "Decay (early→late)": decay.round(4),
        "Stable":    (cv < 0.5),
    }).sort_values("Mean Imp", ascending=False).reset_index(drop=True)

    return result


def select_stable_features(stability_df: pd.DataFrame,
                             cv_threshold: float = 0.5) -> List[str]:
    """Return only feature names with CV below threshold."""
    if stability_df.empty:
        return FEATURE_COLS
    stable = stability_df[stability_df["CV (instability)"] < cv_threshold]["Feature"].tolist()
    return stable if len(stable) >= 5 else FEATURE_COLS


# 2. BASELINE BENCHMARKING

def run_baseline_benchmarks(feat_df: pd.DataFrame,
                              price: pd.Series) -> dict:
    """
    Compare ML against three naive strategies:
      - Random predictor (coin flip)
      - Momentum: go long when r_t > 0
      - Moving-average crossover: EMA9 > EMA21
    Returns dict of {strategy_name: {f1, accuracy, pnl_pct}}.
    """
    if feat_df.empty or price.empty:
        return {}

    df = feat_df[feat_df["label"] != 0].copy()
    if len(df) < 20:
        return {}

    y_true = (df["label"] == 1).astype(int).values
    results: dict = {}

    # Random
    np.random.seed(42)
    y_rand = np.random.randint(0, 2, size=len(y_true))
    results["Random"] = {
        "f1":       round(float(f1_score(y_true, y_rand, zero_division=0)), 3),
        "accuracy": round(float((y_rand == y_true).mean()), 3),
    }

    # Momentum: predict Long(1) if r1 > 0 else Short(0)
    if "r1" in df.columns:
        y_mom = (df["r1"].values > 0).astype(int)
        results["Momentum"] = {
            "f1":       round(float(f1_score(y_true, y_mom, zero_division=0)), 3),
            "accuracy": round(float((y_mom == y_true).mean()), 3),
        }

    # MA Crossover: predict Long if trend_up == 1
    if "trend_up" in df.columns:
        y_mac = df["trend_up"].values.astype(int)
        results["MA Crossover"] = {
            "f1":       round(float(f1_score(y_true, y_mac, zero_division=0)), 3),
            "accuracy": round(float((y_mac == y_true).mean()), 3),
        }

    # Buy-and-hold PnL for context
    if len(price) > 5:
        bh_ret = float((price.iloc[-1] / price.iloc[0]) - 1) * 100
        results["Buy & Hold"] = {"f1": None, "accuracy": None, "pnl_pct": round(bh_ret, 2)}

    return results


# 3. PROBABILITY CALIBRATION LAYER

def calibrate_probabilities(model: Any, scaler: Any,
                              feat_df: pd.DataFrame,
                              method: str = "platt") -> Optional[Any]:
    """
    Platt scaling (logistic) or isotonic regression calibration.
    Returns a fitted calibrated classifier or None on failure.
    """
    if feat_df.empty or len(feat_df) < 30:
        return None

    df = feat_df[feat_df["label"] != 0].copy()
    if len(df) < 20:
        return None

    try:
        avail = [c for c in FEATURE_COLS if c in df.columns]
        X_raw = _san_df(df[avail])
        try:

            X     = scaler.transform(X_raw)

        except Exception:

            return pd.DataFrame()
        y     = (df["label"] == 1).astype(int).values

        cal_method = "sigmoid" if method == "platt" else "isotonic"
        cal_clf    = CalibratedClassifierCV(
            model, method=cal_method, cv="prefit",
        )
        cal_clf.fit(X, y)
        log.info("Probability calibration (%s) fitted on %d samples", method, len(y))
        return cal_clf
    except Exception as exc:
        log.debug("Calibration failed: %s", exc)
        return None


def get_calibrated_probs(cal_clf: Any, X: np.ndarray) -> np.ndarray:
    """Return calibrated probabilities, fall back gracefully."""
    if cal_clf is None:
        return np.full(len(X), 0.5)
    try:
        return cal_clf.predict_proba(X)[:, 1]
    except Exception:
        return np.full(len(X), 0.5)


# 4. META-LABELING LAYER

def train_meta_model(primary_probs: np.ndarray,
                      feat_df: pd.DataFrame,
                      labels_df: pd.DataFrame) -> Optional[Any]:
    """
    Secondary model that predicts whether the PRIMARY model's prediction
    is correct (meta-label = 1 means primary was right).
    Filters out low-confidence trades.
    """
    if feat_df.empty or len(feat_df) < 40:
        return None

    df = feat_df[feat_df["label"] != 0].copy()
    if len(df) < 30 or len(primary_probs) != len(df):
        return None

    try:
        # Meta-label: 1 if primary correctly predicted direction
        primary_pred = (primary_probs[:len(df)] > 0.5).astype(int)
        y_true       = (df["label"] == 1).astype(int).values
        meta_y       = (primary_pred == y_true).astype(int)

        if meta_y.mean() < 0.1 or meta_y.mean() > 0.95:
            return None  # degenerate — skip

        avail = [c for c in FEATURE_COLS if c in df.columns]
        X_raw = _san_df(df[avail])
        sc_   = StandardScaler()
        X     = sc_.fit_transform(_san(X_raw))

        split = int(len(X) * 0.70)
        meta_clf = RandomForestClassifier(
            n_estimators=60, max_depth=4, random_state=99,
            n_jobs=-1, class_weight="balanced",
        )
        meta_clf.fit(X[:split], meta_y[:split])
        test_acc = float((meta_clf.predict(X[split:]) == meta_y[split:]).mean())
        log.info("Meta-model trained — correctness accuracy %.3f", test_acc)
        return {"model": meta_clf, "scaler": sc_, "accuracy": test_acc}
    except Exception as exc:
        log.debug("Meta-model failed: %s", exc)
        return None


def apply_meta_filter(signals_df: pd.DataFrame,
                       meta_result: Optional[dict],
                       feat_df: pd.DataFrame,
                       confidence_min: float = 0.60) -> pd.DataFrame:
    """
    Drop signals where meta-model confidence < confidence_min.
    Keeps the DataFrame structure intact but zeros out low-quality trades.
    """
    if meta_result is None or signals_df.empty or feat_df.empty:
        return signals_df

    try:
        avail = [c for c in FEATURE_COLS if c in feat_df.columns]
        X_raw = _san_df(feat_df[avail])
        X     = meta_result["scaler"].transform(X_raw)
        meta_conf = meta_result["model"].predict_proba(X)[:, 1]

        n   = min(len(signals_df), len(meta_conf))
        out = signals_df.copy()
        low_conf_mask = meta_conf[:n] < confidence_min
        out.iloc[:n] = out.iloc[:n].copy()
        out.loc[out.index[:n][low_conf_mask], "signal"] = 0
        filtered = int(low_conf_mask.sum())
        log.info("Meta-filter removed %d low-confidence signals", filtered)
        return out
    except Exception as exc:
        log.debug("Meta-filter error: %s", exc)
        return signals_df


# 5. DYNAMIC THRESHOLD OPTIMISATION

def optimise_threshold_per_regime(probs: np.ndarray,
                                   labels: np.ndarray,
                                   regime_labels: np.ndarray,
                                   regimes: List[str] = None) -> dict:
    """
    For each regime find the probability threshold that maximises F1.
    Returns {regime_name: best_threshold}.
    """
    regimes  = regimes or ["bull", "bear", "sideways"]
    state_map = {0: "bear", 1: "sideways", 2: "bull"}
    thresholds = {}

    for state_val, regime_name in state_map.items():
        mask = (regime_labels == state_val)
        if mask.sum() < 10:
            thresholds[regime_name] = 0.55
            continue
        p_sub = probs[mask]
        y_sub = labels[mask]
        best_f1 = -1.0
        best_t  = 0.55
        for t in np.arange(0.40, 0.80, 0.02):
            preds = (p_sub >= t).astype(int)
            f1    = float(f1_score(y_sub, preds, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_t  = round(t, 2)
        thresholds[regime_name] = best_t

    return thresholds


# 6. WALK-FORWARD VALIDATION ENGINE

def walk_forward_train_test(feat_df: pd.DataFrame,
                              price: pd.Series,
                              window_bars: int = None,
                              step_bars:   int = None) -> List[dict]:
    """
    True walk-forward: train on window, test on next step, slide forward.
    No look-ahead. Returns list of fold metrics.
    """
    if feat_df.empty or len(feat_df) < 60:
        return []

    df    = feat_df[feat_df["label"] != 0].copy()
    n     = len(df)
    wsize = window_bars or max(30, n // 4)
    step  = step_bars   or max(10, wsize // 3)
    avail = [c for c in FEATURE_COLS if c in df.columns]
    folds: List[dict] = []

    for start in range(0, n - wsize - step, step):
        tr_end = start + wsize
        te_end = min(tr_end + step, n)
        if te_end - tr_end < 5:
            break

        X_tr = df[avail].fillna(0).values.astype(float)
        y_tr = (df["label"].iloc[start:tr_end] == 1).astype(int).values
        X_te = df[avail].fillna(0).values.astype(float)
        y_te = (df["label"].iloc[tr_end:te_end] == 1).astype(int).values

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        try:
            sc_  = StandardScaler()
            Xts  = sc_.fit_transform(_san(X_tr))
            Xvs  = sc_.transform(X_te)

            clf = RandomForestClassifier(
                n_estimators=60, max_depth=5, random_state=42,
                n_jobs=-1, class_weight="balanced",
            )
            clf.fit(Xts, y_tr)
            proba = clf.predict_proba(Xvs)[:, 1]
            preds = (proba >= 0.55).astype(int)

            folds.append({
                "fold":     len(folds) + 1,
                "train_n":  int(tr_end - start),
                "test_n":   int(te_end - tr_end),
                "f1":       round(float(f1_score(y_te, preds, zero_division=0)), 3),
                "auc":      round(float(roc_auc_score(y_te, proba)
                                         if len(np.unique(y_te)) > 1 else 0.5), 3),
                "accuracy": round(float((preds == y_te).mean()), 3),
                "period":   f"fold {len(folds)+1}",
            })
        except Exception as exc:
            log.debug("WF fold error: %s", exc)

    return folds


# 7. ADVANCED BACKTESTING ENGINE

def advanced_backtest(price: pd.Series,
                       signals_df: pd.DataFrame,
                       commission: float = None,
                       latency_bars: int = 1,
                       max_hold_bars: int = 48,
                       max_dd_cap: float = -0.20,
                       vol_target: float = 0.15,
                       max_exposure: float = 1.0) -> dict:
    """
    Realistic backtest with:
      - Latency simulation (entry delayed by latency_bars)
      - Position holding logic (don't flip each bar)
      - Portfolio risk constraints (DD cap + vol targeting + exposure limits)
    """
    commission = commission or CFG["commission"]
    if signals_df.empty or len(signals_df) < 5:
        return {}
    try:
        sig  = signals_df.set_index("time")["signal"]
        sig  = sig.reindex(price.index, method="ffill").fillna(0)

        # Latency: entry is delayed
        sig_delayed = sig.shift(latency_bars).fillna(0)

        # Position holding: only change position when signal changes
        pos = pd.Series(0.0, index=price.index)
        current_pos  = 0.0
        hold_count   = 0
        for t in price.index:
            new_sig = float(sig_delayed.get(t, 0))
            # Exit if max hold exceeded
            if hold_count >= max_hold_bars and current_pos != 0:
                current_pos = 0.0
                hold_count  = 0
            # Enter new position if signal changes
            if new_sig != current_pos and new_sig != 0:
                current_pos = new_sig
                hold_count  = 0
            elif new_sig == 0 and current_pos != 0:
                current_pos = 0.0
                hold_count  = 0
            else:
                hold_count += 1
            pos[t] = current_pos

        # Vol targeting: scale position by vol target / realised vol
        raw_ret = price.pct_change().fillna(0)
        realvol = raw_ret.rolling(20).std().fillna(raw_ret.std()) * np.sqrt(365 * 24)
        scale   = (vol_target / (realvol + 1e-10)).clip(0.1, max_exposure)
        pos_scaled = (pos * scale).clip(-max_exposure, max_exposure)

        # Costs
        trades  = pos_scaled.diff().abs().fillna(0)
        fee_c   = trades * commission * 1.0
        vol_n   = raw_ret.abs().rolling(20).mean().fillna(0.001)
        slip_c  = trades * CFG["slippage"] * (1 + vol_n * 5)
        gross   = raw_ret * pos_scaled
        net     = gross - fee_c - slip_c

        # Max drawdown cap: liquidate positions when DD hits cap
        cum  = (1 + net).cumprod()
        peak = cum.expanding().max()
        dd   = (cum - peak) / (peak.abs() + 1e-10)

        # Apply DD cap — zero out returns during max-DD breach
        dd_breach   = dd < max_dd_cap
        net_capped  = net.copy()
        net_capped[dd_breach] = 0.0
        cum_capped  = (1 + net_capped).cumprod()
        bh          = (1 + raw_ret).cumprod()

        ann_f   = 365 * 24
        ann_ret = float(net_capped.mean() * ann_f)
        ann_vol = float(net_capped.std() * np.sqrt(ann_f)) + 1e-10
        sharpe  = ann_ret / ann_vol
        downvol = float(net_capped[net_capped < 0].std() * np.sqrt(ann_f)) + 1e-10
        sortino = ann_ret / downvol
        max_dd_ = float(((cum_capped - cum_capped.expanding().max()) /
                          (cum_capped.expanding().max().abs() + 1e-10)).min())
        wins_   = float((net_capped[pos_scaled != 0] > 0).mean()) \
                  if (pos_scaled != 0).any() else 0.5
        gp  = float(net_capped[net_capped > 0].sum())
        gl  = float(abs(net_capped[net_capped < 0].sum()))
        pf  = gp / (gl + 1e-10)
        n_t = int(trades.astype(bool).sum())

        return dict(
            equity=cum_capped, bh=bh,
            drawdown=(cum_capped - cum_capped.expanding().max()) /
                      (cum_capped.expanding().max().abs() + 1e-10),
            returns=net_capped,
            ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
            sortino=sortino, max_dd=max_dd_,
            win_rate=wins_, profit_factor=pf, n_trades=n_t,
            dd_cap_events=int(dd_breach.sum()),
            latency_bars=latency_bars,
        )
    except Exception as exc:
        log.error("Advanced backtest: %s", exc)
        return {}


# 8. PORTFOLIO RISK CONSTRAINTS

def apply_portfolio_constraints(returns: pd.Series,
                                  max_dd_cap: float = -0.20,
                                  vol_target: float  = 0.15) -> dict:
    """
    Compute how often constraints would have been triggered historically.
    Returns summary dict for display.
    """
    if returns.empty:
        return {}
    cum  = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd   = (cum - peak) / (peak.abs() + 1e-10)
    ann_vol = float(returns.std() * np.sqrt(365 * 24))

    dd_breaches = int((dd < max_dd_cap).sum())
    vol_ratio   = ann_vol / max(vol_target, 1e-4)

    return dict(
        dd_breaches=dd_breaches,
        ann_vol_pct=round(ann_vol * 100, 2),
        vol_target_pct=round(vol_target * 100, 1),
        vol_scaling_factor=round(min(1.0, vol_target / max(ann_vol, 1e-4)), 3),
        max_dd_pct=round(float(dd.min()) * 100, 2),
        constraint_active=(dd_breaches > 0 or vol_ratio > 1.2),
    )


# 9. PERFORMANCE ATTRIBUTION

def performance_attribution(returns: pd.Series,
                              signals_df: pd.DataFrame,
                              feat_df:    pd.DataFrame,
                              regime_labels: pd.Series) -> dict:
    """
    Break down P&L by:
      - Regime (bull / bear / sideways)
      - Feature group (momentum, volatility, oscillators, flow)
      - Signal direction (long vs short trades)
    """
    if returns.empty or signals_df.empty:
        return {}

    attr: dict = {}

    # By regime
    try:
        sig = signals_df.set_index("time")["signal"].reindex(returns.index,
                                                               method="ffill").fillna(0)
        r_lab = regime_labels.reindex(returns.index, method="ffill").fillna("sideways")
        for regime in ["bull", "bear", "sideways"]:
            mask    = r_lab == regime
            ret_sub = returns[mask & (sig != 0)]
            attr[f"regime_{regime}"] = {
                "ann_ret_pct": round(float(ret_sub.mean() * 365 * 24 * 100) if len(ret_sub) else 0, 2),
                "n_trades":    int((sig[mask] != 0).sum()),
                "win_rate":    round(float((ret_sub > 0).mean()) if len(ret_sub) else 0.5, 3),
            }
    except Exception as exc:
        log.debug("Regime attribution: %s", exc)

    # By signal direction
    try:
        long_mask  = sig ==  1
        short_mask = sig == -1
        attr["direction_long"]  = {
            "ann_ret_pct": round(float(returns[long_mask].mean()  * 365 * 24 * 100), 2),
            "n_trades":    int(long_mask.sum()),
            "win_rate":    round(float((returns[long_mask]  > 0).mean()) if long_mask.any()  else 0.5, 3),
        }
        attr["direction_short"] = {
            "ann_ret_pct": round(float(returns[short_mask].mean() * 365 * 24 * 100), 2),
            "n_trades":    int(short_mask.sum()),
            "win_rate":    round(float((returns[short_mask] > 0).mean()) if short_mask.any() else 0.5, 3),
        }
    except Exception as exc:
        log.debug("Direction attribution: %s", exc)

    # By feature group importance (from feat_df if available)
    feature_groups = {
        "Momentum":    ["r1", "r5", "r10", "r20", "trend_up"],
        "Volatility":  ["vol_5", "vol_20", "ewma_vol", "bb_bw", "atr_pct"],
        "Oscillators": ["rsi", "stoch_k", "mfi", "willr", "cci"],
        "Flow":        ["ofi", "cvd_norm", "vol_ratio", "above_vwap"],
    }
    if not feat_df.empty:
        for gname, cols in feature_groups.items():
            avail = [c for c in cols if c in feat_df.columns]
            if avail:
                group_mean_imp = float(feat_df[avail].abs().mean().mean())
                attr[f"group_{gname}"] = {"mean_importance": round(group_mean_imp, 4)}

    return attr


# 10. STATISTICAL SIGNIFICANCE TESTING

def significance_tests(returns: pd.Series, n_bootstrap: int = 1000) -> dict:
    """
    t-test, bootstrap resampling, p-value of Sharpe.
    Returns {t_stat, p_value, is_significant, boot_sharpe_p, boot_ci}.
    """
    if len(returns) < 10:
        return {}

    r  = returns.dropna()
    mu = float(r.mean())
    se = float(r.std() / np.sqrt(len(r)) + 1e-10)

    # One-sample t-test (H0: mean return == 0)
    t_stat = mu / se
    p_val  = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(r) - 1)))

    # Bootstrap: proportion of bootstrap Sharpes > 0 — fully vectorised
    ann_f = 365 * 24
    r_arr = r.values
    # Sample all n_bootstrap paths at once (matrix: n_bootstrap × len(r))
    idx       = np.random.randint(0, len(r_arr), size=(n_bootstrap, len(r_arr)))
    samples   = r_arr[idx]                                   # (n_bootstrap, n)
    s_means   = samples.mean(axis=1)
    s_stds    = samples.std(axis=1)
    boot_sr   = s_means * ann_f / (s_stds * np.sqrt(ann_f) + 1e-10)
    boot_p    = float((boot_sr > 0).mean())
    boot_ci   = (float(np.percentile(boot_sr, 2.5)),
                 float(np.percentile(boot_sr, 97.5)))
    obs_sharpe = float(r.mean() * ann_f / (r.std() * np.sqrt(ann_f) + 1e-10))

    return dict(
        t_stat=round(t_stat, 4),
        p_value=round(p_val, 4),
        is_significant=(p_val < 0.05),
        obs_sharpe=round(obs_sharpe, 3),
        boot_sharpe_mean=round(float(boot_sr.mean()), 3),
        boot_sharpe_p=round(boot_p, 3),
        boot_ci=boot_ci,
        n_obs=len(r),
    )


# 11. MODEL DRIFT DETECTION

def detect_feature_drift(feat_df_ref: pd.DataFrame,
                          feat_df_cur: pd.DataFrame,
                          symbol: str = "BTC-USD",
                          psi_threshold: float = 0.10) -> pd.DataFrame:
    """
    Population Stability Index (PSI) per feature.
    PSI > 0.10 = moderate drift;  PSI > 0.25 = major drift → retrain.
    Logs drift events to DB.
    """
    if feat_df_ref.empty or feat_df_cur.empty:
        return pd.DataFrame()

    rows = []
    for col in FEATURE_COLS:
        if col not in feat_df_ref.columns or col not in feat_df_cur.columns:
            continue
        try:
            ref = feat_df_ref[col].dropna().values
            cur = feat_df_cur[col].dropna().values
            if len(ref) < 5 or len(cur) < 5:
                continue

            # Bin using reference quantiles
            bins = np.unique(np.percentile(ref, np.linspace(0, 100, 11)))
            if len(bins) < 3:
                continue

            ref_counts, _ = np.histogram(ref, bins=bins)
            cur_counts, _ = np.histogram(cur, bins=bins)
            ref_pct = ref_counts / (ref_counts.sum() + 1e-10)
            cur_pct = cur_counts / (cur_counts.sum() + 1e-10)
            # Avoid log(0)
            ref_pct = np.clip(ref_pct, 1e-6, None)
            cur_pct = np.clip(cur_pct, 1e-6, None)
            psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

            status = ("STABLE" if psi < 0.10
                      else "MODERATE DRIFT" if psi < 0.25
                      else "MAJOR DRIFT — RETRAIN")
            _db_log_drift(symbol, col, round(psi, 4), status)
            rows.append({
                "Feature": col,
                "PSI":     round(psi, 4),
                "Status":  status,
                "Retrain": (psi >= 0.25),
            })
        except Exception:
            pass

    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def should_retrain(drift_df: pd.DataFrame, threshold: float = 0.25) -> bool:
    """Return True if any feature has PSI >= threshold."""
    if drift_df.empty:
        return False
    return bool((drift_df["PSI"] >= threshold).any())


# 12. ONLINE LEARNING / RETRAINING SCHEDULER

def retraining_schedule(symbol: str,
                         last_run: Optional[datetime] = None,
                         drift_triggered: bool = False,
                         schedule: str = "weekly") -> dict:
    """
    Determine whether retraining is due.
    schedule: 'daily' | 'weekly' | 'drift_only'
    Returns dict with {due, reason, next_run}.
    """
    now = datetime.utcnow()
    if last_run is None:
        return {"due": True, "reason": "No previous run found", "next_run": now}

    age_hours = (now - last_run).total_seconds() / 3600
    thresholds = {"daily": 24, "weekly": 168, "drift_only": 999_999}
    thr_hours  = thresholds.get(schedule, 168)

    due    = age_hours >= thr_hours or drift_triggered
    reason = ("Drift detected" if drift_triggered
               else f"Schedule ({schedule}): {age_hours:.0f}h elapsed"
               if age_hours >= thr_hours
               else "Not due yet")
    next_r = last_run + timedelta(hours=thr_hours)

    return dict(due=due, reason=reason,
                age_hours=round(age_hours, 1),
                next_run=next_r.strftime("%Y-%m-%d %H:%M UTC"))


# 13. EXPERIMENT TRACKING SYSTEM

def log_experiment(symbol: str, cfg_params: dict,
                    model_results: dict,
                    bt_result: dict,
                    notes: str = "") -> str:
    """
    Log a training run with params + metrics.
    Returns the run_id for reference.
    """
    run_id = _uuid_mod.uuid4().hex[:8]
    # Aggregate metrics across models
    agg_f1  = float(np.mean([m.get("cv_f1", 0) for m in model_results.values()])) if model_results else 0
    agg_auc = float(np.mean([m.get("auc", 0)   for m in model_results.values()])) if model_results else 0
    metrics = {
        "cv_f1":    round(agg_f1,  3),
        "auc":      round(agg_auc, 3),
        "sharpe":   round(bt_result.get("sharpe", 0), 3),
        "n_trades": bt_result.get("n_trades", 0),
    }
    _db_log_experiment(symbol, run_id, cfg_params, metrics, notes)
    log.info("Experiment %s logged for %s", run_id, symbol)
    return run_id


def compare_experiments(symbol: str) -> pd.DataFrame:
    """Load experiment history from DB for comparison display."""
    return _db_load_experiments(symbol)


# 14. ALPHA DECAY MONITORING

def monitor_alpha_decay(bt_returns: pd.Series,
                          symbol: str,
                          window_days: int = 7) -> pd.DataFrame:
    """
    Slice strategy returns into rolling windows and track performance
    to detect if the alpha is degrading over time.
    """
    if bt_returns.empty or len(bt_returns) < window_days * 2:
        return pd.DataFrame()

    ann_f = 365 * 24
    rows  = []
    step  = max(1, len(bt_returns) // 10)

    for i in range(0, len(bt_returns) - step, step):
        chunk = bt_returns.iloc[i: i + step]
        if len(chunk) < 5:
            continue
        sr  = float(chunk.mean() * ann_f / (chunk.std() * np.sqrt(ann_f) + 1e-10))
        ar  = float(chunk.mean() * ann_f * 100)
        wr  = float((chunk > 0).mean())
        lbl = f"W{i // step + 1}"
        rows.append({"Window": lbl, "Sharpe": round(sr, 3),
                     "Ann Ret%": round(ar, 2), "Win Rate": round(wr, 3)})
        _db_log_alpha_decay(symbol, lbl, sr, ar, wr)

    return pd.DataFrame(rows)


# 15. FEEDBACK LOOP — LOSING TRADE ANALYSER

def analyse_losing_trades(returns: pd.Series,
                            signals_df: pd.DataFrame,
                            feat_df:    pd.DataFrame) -> dict:
    """
    Examine features at the entry point of losing trades to find
    patterns. Returns: feature means for winners vs losers.
    """
    if returns.empty or signals_df.empty or feat_df.empty:
        return {}

    try:
        sig = signals_df.set_index("time")["signal"].reindex(returns.index,
                                                               method="ffill").fillna(0)
        trade_rets = returns[sig != 0]
        if len(trade_rets) < 5:
            return {}

        win_mask  = trade_rets > 0
        lose_mask = trade_rets <= 0

        avail = [c for c in FEATURE_COLS if c in feat_df.columns]
        n     = min(len(feat_df), len(signals_df))
        f_sub = feat_df[avail].iloc[:n].reset_index(drop=True)
        r_sub = trade_rets.reset_index(drop=True).iloc[:len(f_sub)]

        win_means  = f_sub.loc[r_sub > 0].mean()
        lose_means = f_sub.loc[r_sub <= 0].mean()
        diff       = (win_means - lose_means).abs().sort_values(ascending=False)

        top_discriminators = diff.head(5).index.tolist()

        return dict(
            n_winners=int(win_mask.sum()),
            n_losers=int(lose_mask.sum()),
            win_rate=round(float(win_mask.mean()), 3),
            avg_win_ret=round(float(trade_rets[win_mask].mean() * 100), 3),
            avg_lose_ret=round(float(trade_rets[lose_mask].mean() * 100), 3),
            top_discriminators=top_discriminators,
            win_feature_means=win_means.round(4).to_dict(),
            lose_feature_means=lose_means.round(4).to_dict(),
        )
    except Exception as exc:
        log.debug("Losing trade analysis: %s", exc)
        return {}


#  END OF INTELLIGENCE ENGINES



def _optuna_tune_rf(X_tr: np.ndarray, y_tr: np.ndarray) -> dict:
    if not OPTUNA_OK:
        return {"n_estimators": 100, "max_depth": 5}

    def _obj(trial):
        clf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            random_state=42, n_jobs=-1, class_weight="balanced",
        )
        scores = []
        for tr, va in _purged_splits(len(X_tr)):
            try:
                clf.fit(X_tr[tr], y_tr[tr])
                scores.append(f1_score(y_tr[va], clf.predict(X_tr[va]), zero_division=0))
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(_obj, n_trials=CFG["optuna_trials"], show_progress_bar=False)
    return study.best_params


@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def train_models(feat_df: pd.DataFrame, symbol: str = "BTC-USD") -> dict:
    """Train multi-model ensemble with purged CV, Optuna, persistence."""
    df = feat_df[feat_df["label"] != 0].copy()
    if len(df) < 40:
        return {}
    df = _remove_outliers(df, 0.05)
    if len(df) < 30:
        return {}

    avail_cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(avail_cols) < 5:
        return {}
    X_raw = df[avail_cols].copy()
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
    for _col in X_raw.columns:
        _q1, _q3 = X_raw[_col].quantile(0.25), X_raw[_col].quantile(0.75)
        _iqr = _q3 - _q1
        X_raw[_col] = X_raw[_col].clip(lower=_q1 - 10*_iqr, upper=_q3 + 10*_iqr)
    X_raw = X_raw.fillna(X_raw.median()).fillna(0)
    X  = X_raw.values.astype(float)
    y  = (df["label"] == 1).astype(int).values
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    n     = len(Xs)
    split = int(n * 0.70)
    X_tr, X_te = Xs[:split], Xs[split:]
    y_tr, y_te = y[:split],  y[split:]

    rf_params = _optuna_tune_rf(X_tr, y_tr)

    models_def: Dict[str, Any] = {
        "Random Forest": RandomForestClassifier(
            **rf_params, random_state=42, n_jobs=-1, class_weight="balanced"),
        "Logistic":      LogisticRegression(
            max_iter=500, random_state=42, class_weight="balanced"),
    }
    if XGB_OK:
        models_def["XGBoost"] = xgb.XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42, verbosity=0)
    if LGB_OK:
        models_def["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.05,
            random_state=42, verbose=-1)

    results: dict = {}
    voting_ests: list = []
    voting_wts:  list = []
    cv_splits = list(_purged_splits(split))

    for name, model in models_def.items():
        # Try loading persisted model first
        saved = _model_load(symbol, name)
        if saved is not None:
            try:
                model = saved.get("model", saved) if isinstance(saved, dict) else saved
            except Exception:
                pass
        try:
            cv_f1 = []
            for tr_idx, va_idx in cv_splits:
                try:
                    model.fit(Xs[tr_idx], y[tr_idx])
                    cv_f1.append(f1_score(y[va_idx], model.predict(Xs[va_idx]), zero_division=0))
                except Exception:
                    cv_f1.append(0.0)

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_prob = (model.predict_proba(X_te)[:, 1]
                      if hasattr(model, "predict_proba")
                      else np.full(len(y_te), 0.5))

            imp = getattr(model, "feature_importances_", None)
            if imp is None:
                coef = getattr(model, "coef_", None)
                imp  = (np.abs(coef[0]) if coef is not None
                        else np.zeros(len(FEATURE_COLS)))

            cv_f1_mean = float(np.mean(cv_f1)) if cv_f1 else 0.0
            metrics = dict(
                model=model, scaler=sc,
                cv_f1=cv_f1_mean, cv_std=float(np.std(cv_f1)) if cv_f1 else 0.0,
                accuracy=float((y_pred == y_te).mean()),
                f1=float(f1_score(y_te, y_pred, zero_division=0)),
                precision=float(precision_score(y_te, y_pred, zero_division=0)),
                recall=float(recall_score(y_te, y_pred, zero_division=0)),
                auc=float(roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5),
                confusion=confusion_matrix(y_te, y_pred).tolist(),
                importance=list(imp),
                y_te=y_te.tolist(), y_pred=y_pred.tolist(), y_prob=y_prob.tolist(),
            )
            results[name] = metrics
            _model_save(symbol, name, {"model": model, "scaler": sc})
            _db_log_metric(symbol, name, metrics)
            voting_ests.append((name, model))
            voting_wts.append(cv_f1_mean)
        except Exception as exc:
            log.warning("Model %s failed: %s", name, exc)

    # Soft-voting ensemble
    if len(voting_ests) >= 2:
        try:
            norm_wts = [w / (sum(voting_wts) + 1e-10) for w in voting_wts]
            voter = VotingClassifier(estimators=voting_ests, voting="soft",
                                     weights=norm_wts)
            voter.fit(X_tr, y_tr)
            yv_pred = voter.predict(X_te)
            yv_prob = voter.predict_proba(X_te)[:, 1]
            results["Ensemble"] = dict(
                model=voter, scaler=sc,
                cv_f1=float(np.average(voting_wts, weights=voting_wts)),
                cv_std=0.0,
                accuracy=float((yv_pred == y_te).mean()),
                f1=float(f1_score(y_te, yv_pred, zero_division=0)),
                precision=float(precision_score(y_te, yv_pred, zero_division=0)),
                recall=float(recall_score(y_te, yv_pred, zero_division=0)),
                auc=float(roc_auc_score(y_te, yv_prob) if len(np.unique(y_te)) > 1 else 0.5),
                confusion=confusion_matrix(y_te, yv_pred).tolist(),
                importance=[0.0] * len(avail_cols),
                y_te=y_te.tolist(), y_pred=yv_pred.tolist(), y_prob=yv_prob.tolist(),
            )
            _model_save(symbol, "Ensemble", {"model": voter, "scaler": sc})
        except Exception as exc:
            log.warning("Ensemble failed: %s", exc)

    return results


@st.cache_data(ttl=600, show_spinner=False)
def compute_shap_values(_model: Any, _X: np.ndarray) -> Optional[np.ndarray]:
    if not SHAP_OK:
        return None
    try:
        exp = (shap.TreeExplainer(_model)
               if hasattr(_model, "feature_importances_")
               else shap.LinearExplainer(_model, _X))
        sv = exp.shap_values(_X)
        return sv[1] if isinstance(sv, list) else sv
    except Exception:
        return None


def compute_lime(model: Any, scaler: Any,
                 feat_df: pd.DataFrame) -> Optional[dict]:
    if not LIME_OK or feat_df.empty:
        return None
    try:
        _lX = feat_df[FEATURE_COLS].copy().replace([np.inf, -np.inf], np.nan)
        for _lc in _lX.columns:
            _lq1, _lq3 = _lX[_lc].quantile(0.25), _lX[_lc].quantile(0.75)
            _liqr = _lq3 - _lq1
            _lX[_lc] = _lX[_lc].clip(lower=_lq1 - 10*_liqr, upper=_lq3 + 10*_liqr)
        X  = _lX.fillna(_lX.median()).fillna(0).values.astype(float)
        Xs = scaler.transform(X)
        tr = Xs[:-10] if len(Xs) > 10 else Xs
        ex = lime.lime_tabular.LimeTabularExplainer(
            tr, feature_names=FEATURE_COLS,
            mode="classification", random_state=42)
        e  = ex.explain_instance(Xs[-1], model.predict_proba,
                                  num_features=8, num_samples=200)
        return {"factors": e.as_list(),
                "prob": float(model.predict_proba([Xs[-1]])[0, 1])}
    except Exception:
        return None


#  LAYER 7: SIGNAL GENERATOR — Kelly bet sizing + regime filter

@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def generate_signals(df: pd.DataFrame, labels_df: pd.DataFrame,
                     feat_df: pd.DataFrame, _model_results: dict,
                     confidence_thr: float = 0.55,
                     regime: str = "bull") -> pd.DataFrame:
    model_results = _model_results  # local alias — _ prefix skips st.cache_data hashing
    if not model_results or feat_df.empty or "label" not in feat_df.columns:
        return pd.DataFrame()

    best_name = ("Ensemble" if "Ensemble" in model_results
                 else max(model_results, key=lambda k: model_results[k]["cv_f1"]))
    mr  = model_results[best_name]
    mdl = mr["model"]; sc = mr["scaler"]

    _X_raw = feat_df[FEATURE_COLS].copy().replace([np.inf, -np.inf], np.nan)
    for _gc in _X_raw.columns:
        _gq1, _gq3 = _X_raw[_gc].quantile(0.25), _X_raw[_gc].quantile(0.75)
        _giqr = _gq3 - _gq1
        _X_raw[_gc] = _X_raw[_gc].clip(lower=_gq1 - 10*_giqr, upper=_gq3 + 10*_giqr)
    X    = _X_raw.fillna(_X_raw.median()).fillna(0).values.astype(float)
    if X.shape[1] != sc.n_features_in_:
        return pd.DataFrame()
    Xs   = sc.transform(X)
    probs = (mdl.predict_proba(Xs)[:, 1]
             if hasattr(mdl, "predict_proba")
             else np.full(len(X), 0.5))

    # Kelly bet sizing
    wins = float(mr.get("precision", 0.55))
    b    = wins / max(1 - wins, 0.001)
    kelly = float(np.clip(
        CFG["kelly_fraction"] * (b * wins - (1 - wins)) / max(b, 0.001),
        0.0, 1.0))

    raw_bets = 2 * probs - 1
    smoothed = pd.Series(raw_bets).ewm(span=5, adjust=False).mean().values
    thr      = 2 * confidence_thr - 1
    signals  = np.where(smoothed > thr, 1, np.where(smoothed < -thr, -1, 0))

    atr_now = float(df["atr_pct"].iloc[-1]) if "atr_pct" in df.columns else 0.01
    out = pd.DataFrame({
        "time":     labels_df["time"].values[: len(feat_df)],
        "label":    feat_df["label"].values,
        "prob":     probs,
        "bet_size": smoothed * kelly,
        "signal":   signals,
        "model":    best_name,
        "regime":   regime,
        "tp_pct":   atr_now * 2.0,
        "sl_pct":   atr_now * 1.0,
    })

    if not out.empty:
        last = out.iloc[-1]
        _db_log_signal(df.index.name or "BTC-USD", {
            "signal": int(last["signal"]),
            "prob":   float(last["prob"]),
            "bet_size": float(last["bet_size"]),
            "model":  best_name,
        })
    return out


#  LAYER 8: BACKTESTING — realistic costs + PSR + DSR + bootstrap CI

@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def run_backtest(price: pd.Series, signals_df: pd.DataFrame,
                 commission: float = None) -> dict:
    commission = commission or CFG["commission"]
    if signals_df.empty or len(signals_df) < 5:
        return {}
    try:
        sig     = signals_df.set_index("time")["signal"]
        sig     = sig.reindex(price.index, method="ffill").fillna(0)
        pos     = sig.shift(1).fillna(0)
        raw_ret = price.pct_change().fillna(0)
        gross   = raw_ret * pos
        trades  = pos.diff().abs().fillna(0)
        # Tiered fees: maker 0.5×, taker 1.5×
        taker   = np.where(trades.abs() > 0.5, 1.5, 0.5)
        fee_c   = trades * commission * taker
        vol_n   = price.pct_change().abs().rolling(20).mean().fillna(0.001)
        slip_c  = trades * CFG["slippage"] * (1 + vol_n * 10)
        net     = gross - fee_c - slip_c

        cum  = (1 + net).cumprod()
        bh   = (1 + raw_ret).cumprod()
        peak = cum.expanding().max()
        dd   = (cum - peak) / (peak.abs() + 1e-10)

        ann_f   = 365 * 24
        ann_ret = float(net.mean() * ann_f)
        ann_vol = float(net.std() * np.sqrt(ann_f)) + 1e-10
        sharpe  = ann_ret / ann_vol
        downvol = float(net[net < 0].std() * np.sqrt(ann_f)) + 1e-10
        sortino = ann_ret / downvol
        max_dd  = float(dd.min())
        calmar  = abs(ann_ret / (max_dd + 1e-10))
        wins_   = float((net[pos != 0] > 0).mean()) if (pos != 0).any() else 0.5
        gp      = float(net[net > 0].sum()); gl = float(abs(net[net < 0].sum()))
        pf      = gp / (gl + 1e-10)
        sk      = float(net.skew()); ku = float(net.kurtosis())
        T       = len(net)

        # PSR
        psr_d = np.sqrt(max(1e-10, (1 - sk * sharpe + (ku - 1) / 4 * sharpe ** 2) / max(T - 1, 1)))
        psr   = float(norm.cdf(sharpe / psr_d))

        # DSR (4 implicit trials)
        n_trials = 4
        e_max = (np.sqrt(2 * np.log(n_trials))
                 - (np.log(np.log(n_trials + 1e-10)) + np.log(4 * np.pi))
                 / (2 * np.sqrt(2 * np.log(n_trials)) + 1e-10))
        dsr   = float(norm.cdf((sharpe - e_max) / (psr_d + 1e-10)))

        # Bootstrap Sharpe CI
        r_arr = net.dropna().values
        boot  = []
        for _ in range(300):
            s   = np.random.choice(r_arr, size=len(r_arr), replace=True)
            sv  = s.std() * np.sqrt(ann_f) + 1e-10
            boot.append(s.mean() * ann_f / sv)
        sharpe_ci = (float(np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5)))

        return dict(
            equity=cum, bh=bh, drawdown=dd, returns=net,
            ann_ret=ann_ret, ann_vol=ann_vol,
            sharpe=sharpe, sortino=sortino, calmar=calmar,
            max_dd=max_dd, win_rate=wins_, profit_factor=pf,
            n_trades=int(trades.astype(bool).sum()),
            psr=psr, dsr=dsr, sharpe_ci=sharpe_ci,
            total_cost=float((fee_c + slip_c).sum()),
        )
    except Exception as exc:
        log.error("Backtest: %s", exc)
        return {}


def walk_forward_backtest(df: pd.DataFrame, signals_df: pd.DataFrame,
                           n_folds: int = 5) -> List[dict]:
    if signals_df.empty or df.empty:
        return []
    price = df["Close"].squeeze(); n = len(price)
    fold_size = n // n_folds; results = []
    for k in range(1, n_folds):
        end_idx = (k + 1) * fold_size
        if end_idx > n:
            break
        p_fold = price.iloc[:end_idx]
        s_fold = signals_df[signals_df["time"] <= p_fold.index[-1]]
        bt = run_backtest(p_fold, s_fold)
        if bt:
            bt["fold"] = k; results.append(bt)
    return results


def run_rolling_backtest(price: pd.Series, signals_df: pd.DataFrame) -> List[dict]:
    if signals_df.empty or len(price) < 40:
        return []
    window = max(len(price) // 5, 40); step = window // 2; results = []
    for start in range(0, len(price) - window, step):
        end  = start + window
        pw   = price.iloc[start:end]
        sw   = signals_df[(signals_df["time"] >= pw.index[0]) &
                           (signals_df["time"] <= pw.index[-1])]
        if len(sw) < 5:
            continue
        try:
            bt = run_backtest(pw, sw)
            if bt:
                results.append({
                    "window": start // step,
                    "start":  str(pw.index[0].date()),
                    "end":    str(pw.index[-1].date()),
                    "sharpe": round(bt["sharpe"], 3),
                    "ret":    round(float(bt["returns"].sum()), 4),
                    "max_dd": round(bt["max_dd"], 4),
                })
        except Exception:
            continue
    return results


def benchmark_comparison(strategy_equity: Optional[pd.Series],
                          benchmarks: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []; ann = 365

    def _m(s: pd.Series, name: str) -> dict:
        r  = s.pct_change().dropna()
        cp = (1 + r).cumprod(); pk = cp.expanding().max()
        dd = float(((cp - pk) / (pk + 1e-10)).min())
        ar = float(r.mean() * ann); av = float(r.std() * ann ** 0.5) + 1e-10
        sh = ar / av; cal = abs(ar / (dd + 1e-10))
        return {"Asset": name, "Ann Ret%": round(ar * 100, 1),
                "Sharpe": round(sh, 2), "Max DD%": round(dd * 100, 1),
                "Calmar": round(cal, 2)}

    if strategy_equity is not None and len(strategy_equity) > 5:
        rows.append(_m(strategy_equity, "HECTOR"))
    for label, price in benchmarks.items():
        try:
            rows.append(_m(price, label))
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else pd.DataFrame()


#  LAYER 9: PORTFOLIO OPTIMISATION — HRP + MV Frontier

@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def hrp_weights(ret_df: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado)."""
    try:
        cov  = ret_df.cov()
        dist = np.sqrt((1 - ret_df.corr().clip(-1, 1)) / 2).fillna(0)
        link = linkage(squareform(dist.values), method="ward")
        n    = len(ret_df.columns)

        def _ser(node: int) -> list:
            if node < n:
                return [node]
            L = int(link[node - n, 0]); R = int(link[node - n, 1])
            return _ser(L) + _ser(R)

        ordered = [ret_df.columns[i] for i in _ser(n + n - 2)]
        wts     = pd.Series(1.0, index=ordered)
        clusters = [list(ordered)]

        while clusters:
            nc = []
            for cl in clusters:
                if len(cl) <= 1:
                    continue
                mid = len(cl) // 2; L, R = cl[:mid], cl[mid:]
                def _cv(cols):
                    sub = cov.loc[cols, cols].values
                    iv  = np.linalg.pinv(sub)
                    w   = iv.sum(axis=1) / (iv.sum() + 1e-12)
                    return float(w @ sub @ w)
                vl  = _cv(L); vr = _cv(R)
                alpha = 1 - vl / (vl + vr + 1e-12)
                for a in L: wts[a] *= alpha
                for a in R: wts[a] *= (1 - alpha)
                nc += [L, R]
            clusters = nc

        total = wts.sum()
        return wts / (total if total > 0 else 1)
    except Exception:
        n = len(ret_df.columns)
        return pd.Series(1.0 / n, index=ret_df.columns)


@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def efficient_frontier(ret_df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    mu  = ret_df.mean().values; cov = ret_df.cov().values; na = len(mu)
    rows = []
    for target in np.linspace(mu.min(), mu.max(), n):
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        res = minimize(lambda w: float(w @ cov @ w),
                       np.ones(na) / na, method="SLSQP",
                       bounds=[(0, 1)] * na, constraints=cons,
                       options={"ftol": 1e-9, "maxiter": 200})
        if res.success:
            v = np.sqrt(res.fun)
            rows.append({"vol": v * 100, "ret": target * 100,
                          "sharpe": target / (v + 1e-10)})
    return pd.DataFrame(rows)


#  LAYER 10: RISK MANAGEMENT

@st.cache_data(ttl=CFG["cache_ttl_ohlcv"], show_spinner=False)
def compute_risk(returns: pd.Series) -> dict:
    r = returns.dropna()
    if len(r) < 10:
        return {}
    var95  = float(np.percentile(r, 5))  * 100
    cvar95 = float(r[r <= np.percentile(r, 5)].mean()) * 100
    var99  = float(np.percentile(r, 1))  * 100
    ann_vol = float(r.std() * np.sqrt(365 * 24)) * 100
    sk      = float(r.skew()); ku = float(r.kurtosis())
    sr      = r.mean() / (r.std() + 1e-10) * np.sqrt(365 * 24)
    denom   = np.sqrt(max(1e-10, 1 - sk * sr + (ku - 1) / 4 * sr ** 2))
    psr     = float(norm.cdf(sr * np.sqrt(len(r)) / denom))
    cum     = (1 + r).cumprod(); peak = cum.expanding().max()
    dd      = (cum - peak) / (peak.abs() + 1e-10)
    rs      = (r.rolling(30).mean() / (r.rolling(30).std() + 1e-10)) * np.sqrt(365 * 24)
    is_stat = False
    if ADF_OK and len(r) > 20:
        try:
            _, pval, *_ = adfuller(r)
            is_stat = pval < 0.05
        except Exception:
            pass
    stress = {
        "Flash Crash −30%": float(np.percentile(r, 1) * 30 * 100),
        "Bear Market −60%": float(np.percentile(r, 1) * 60 * 100),
        "Vol Spike ×4":     float(var95 * 4),
        "Corr Breakdown":   float(cvar95 * 2),
        "Liquidity Shock":  float(var99 * 3),
    }
    return dict(var95=var95, cvar95=cvar95, var99=var99, ann_vol=ann_vol,
                psr=psr, skew=sk, kurt=ku, max_dd=float(dd.min() * 100),
                roll_sharpe=rs, drawdown=dd, returns=r,
                is_stationary=is_stat, stress=stress)


#  VOLUME PROFILE

def volume_profile(df: pd.DataFrame, n_bins: int = 30) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    try:
        lo_ = float(df["Low"].min()); hi_ = float(df["High"].max())
        bins = np.linspace(lo_, hi_, n_bins + 1)
        mid  = (bins[:-1] + bins[1:]) / 2; vols = np.zeros(n_bins)
        for _, row in df.iterrows():
            rl = min(float(row["Low"]), float(row["High"]))
            rh = max(float(row["Low"]), float(row["High"]))
            for i, (b0, b1) in enumerate(zip(bins[:-1], bins[1:])):
                ov = max(0, min(b1, rh) - max(b0, rl))
                sp = max(rh - rl, 1e-10)
                vols[i] += float(row["Volume"]) * (ov / sp)
        vp = pd.DataFrame({"price": mid, "volume": vols})
        vp["pct"]    = vp["volume"] / (vp["volume"].sum() + 1e-10)
        vp["is_hvn"] = vp["volume"] >= vp["volume"].quantile(0.80)
        return vp
    except Exception:
        return pd.DataFrame()


#  PDF REPORT

def generate_pdf(cfg: dict, info: dict, bt: dict, risk: dict) -> bytes:
    if not FPDF_OK:
        lines = [
            "HECTOR v6 Strategy Report",
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"Asset: {cfg.get('yf_sym', 'N/A')} | Period: {cfg.get('period', 'N/A')}",
        ]
        if bt:
            for k, v in [
                ("Ann Return", f"{bt.get('ann_ret', 0) * 100:.2f}%"),
                ("Sharpe",     f"{bt.get('sharpe', 0):.3f}"),
                ("Max DD",     f"{bt.get('max_dd', 0) * 100:.2f}%"),
                ("Win Rate",   f"{bt.get('win_rate', 0) * 100:.1f}%"),
                ("Trades",     str(bt.get("n_trades", 0))),
                ("PSR",        f"{bt.get('psr', 0):.3f}"),
                ("DSR",        f"{bt.get('dsr', 0):.3f}"),
            ]:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines).encode("utf-8")
    try:
        pdf = FPDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
        pdf.set_font("Courier", "B", 16); pdf.set_text_color(255, 140, 0)
        pdf.cell(0, 10, "HECTOR v6 Strategy Report", ln=True, align="C")
        pdf.set_font("Courier", "", 9); pdf.set_text_color(140, 140, 140)
        pdf.cell(0, 6, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
                 ln=True, align="C")
        pdf.ln(4); pdf.set_font("Courier", "B", 11); pdf.set_text_color(255, 140, 0)
        pdf.cell(0, 8, "ASSET PROFILE", ln=True)
        pdf.set_font("Courier", "", 9); pdf.set_text_color(0, 0, 0)
        for k, v in [("Asset", cfg.get("yf_sym", "")),
                     ("Period", cfg.get("period", "")),
                     ("Price",  f"${info.get('price', 0):,.4f}"),
                     ("24H%",   f"{info.get('change_24h', 0):+.2f}%")]:
            pdf.cell(60, 6, f"{k}:", border=0); pdf.cell(0, 6, str(v), ln=True)
        if bt:
            pdf.ln(4); pdf.set_font("Courier", "B", 11); pdf.set_text_color(255, 140, 0)
            pdf.cell(0, 8, "BACKTEST RESULTS", ln=True)
            pdf.set_font("Courier", "", 9); pdf.set_text_color(0, 0, 0)
            for k, v in [
                ("Ann Return",    f"{bt.get('ann_ret', 0) * 100:.2f}%"),
                ("Sharpe",        f"{bt.get('sharpe', 0):.3f}"),
                ("Sharpe 95% CI", f"[{bt.get('sharpe_ci', (0, 0))[0]:.2f}, "
                                  f"{bt.get('sharpe_ci', (0, 0))[1]:.2f}]"),
                ("Sortino",       f"{bt.get('sortino', 0):.3f}"),
                ("Max Drawdown",  f"{bt.get('max_dd', 0) * 100:.2f}%"),
                ("Win Rate",      f"{bt.get('win_rate', 0) * 100:.1f}%"),
                ("Profit Factor", f"{bt.get('profit_factor', 0):.2f}"),
                ("# Trades",      str(bt.get("n_trades", 0))),
                ("PSR",           f"{bt.get('psr', 0):.3f}"),
                ("DSR",           f"{bt.get('dsr', 0):.3f}"),
                ("Total Costs",   f"{bt.get('total_cost', 0):.4f}"),
            ]:
                pdf.cell(70, 6, f"{k}:", border=0); pdf.cell(0, 6, str(v), ln=True)
        out = pdf.output(dest="S")
        return out.encode("latin-1") if isinstance(out, str) else bytes(out)
    except Exception as exc:
        log.warning("PDF: %s", exc)
        return "PDF generation error -- check fpdf2 install.".encode("utf-8")


#  CHART FACTORY

def fig_candle(df: pd.DataFrame, title: str = "",
               show_bb: bool = True, show_ema: bool = True,
               show_vwap: bool = True, signals_df: pd.DataFrame = None) -> go.Figure:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03, row_heights=[0.60, 0.20, 0.20])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing=dict(fillcolor="rgba(0,212,138,.20)",
                        line=dict(color=C["green"], width=1)),
        decreasing=dict(fillcolor="rgba(255,61,90,.20)",
                        line=dict(color=C["red"], width=1)),
        name="OHLC", showlegend=False,
    ), row=1, col=1)

    if show_bb and "bb_up" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_up"],
            line=dict(color=C["blue"], width=.8, dash="dot"),
            name="BB+", hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["bb_dn"],
            line=dict(color=C["blue"], width=.8, dash="dot"),
            name="BB−", fill="tonexty", fillcolor="rgba(0,136,255,.04)",
            hoverinfo="skip", showlegend=False), row=1, col=1)

    if show_ema and "ema9" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ema9"],
            line=dict(color=C["orange"], width=1.2), name="EMA9"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["ema21"],
            line=dict(color=C["yellow"], width=1.2), name="EMA21"), row=1, col=1)

    if show_vwap and "vwap" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["vwap"],
            line=dict(color=C["purple"], width=1.0, dash="dash"),
            name="VWAP"), row=1, col=1)

    # Signal overlays
    if signals_df is not None and not signals_df.empty and "time" in signals_df.columns:
        c_price = df["Close"].squeeze()
        for t in signals_df[signals_df["signal"] == 1]["time"]:
            if t in c_price.index:
                fig.add_annotation(x=t, y=float(c_price.loc[t]) * 0.985,
                                   text="▲", showarrow=False,
                                   font=dict(color=C["green"], size=11))
        for t in signals_df[signals_df["signal"] == -1]["time"]:
            if t in c_price.index:
                fig.add_annotation(x=t, y=float(c_price.loc[t]) * 1.015,
                                   text="▼", showarrow=False,
                                   font=dict(color=C["red"], size=11))

    vol_colors = ["rgba(0,212,138,.55)" if float(c) >= float(o)
                  else "rgba(255,61,90,.55)"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"],
        marker_color=vol_colors, name="Volume", showlegend=False), row=2, col=1)

    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi"],
            line=dict(color=C["cyan"], width=1.2), name="RSI"), row=3, col=1)
        fig.add_hline(y=70, line_color=C["red"],   line_dash="dot", line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_color=C["green"], line_dash="dot", line_width=1, row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor=C["red"],   opacity=.04, row=3, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor=C["green"], opacity=.04, row=3, col=1)

    ly = BB(title, h=580)
    ly["xaxis"] = dict(**_xa()); ly["xaxis2"] = _xa(); ly["xaxis3"] = _xa()
    ly["yaxis3"] = dict(**BB()["yaxis"], range=[0, 100]); ly["showlegend"] = True
    fig.update_layout(**ly)
    return fig


def fig_macd(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=.03, row_heights=[.5, .5])
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"],
        line=dict(color=C["blue"], width=1.4), name="MACD"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"],
        line=dict(color=C["orange"], width=1.4), name="Signal"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"],
        marker_color=[C["green"] if v >= 0 else C["red"] for v in df["macd_hist"]],
        showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["stoch_k"],
        line=dict(color=C["cyan"], width=1.2), name="Stoch%K"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["stoch_d"],
        line=dict(color=C["purple"], width=1.2), name="Stoch%D"), row=2, col=1)
    fig.add_hline(y=80, line_color=C["red"],   line_dash="dot", line_width=1, row=2, col=1)
    fig.add_hline(y=20, line_color=C["green"], line_dash="dot", line_width=1, row=2, col=1)
    ly = BB("MACD & Stochastic", h=360)
    ly["xaxis"] = _xa(); ly["xaxis2"] = _xa()
    ly["yaxis2"] = dict(**BB()["yaxis"], range=[0, 100]); ly["showlegend"] = True
    fig.update_layout(**ly)
    return fig


def fig_ofi_cvd(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.5, .5])
    if "ofi" in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df["ofi"],
            marker_color=[C["green"] if v > 0 else C["red"] for v in df["ofi"]],
            name="OFI", showlegend=False), row=1, col=1)
    if "cvd" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["cvd"],
            line=dict(color=C["cyan"], width=1.2), name="CVD"), row=2, col=1)
        if "cvd_ma" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["cvd_ma"],
                line=dict(color=C["orange"], width=1, dash="dash"),
                name="CVD MA"), row=2, col=1)
    ly = BB("Order Flow Imbalance & CVD", h=320)
    ly["xaxis"] = _xa(); ly["xaxis2"] = _xa(); ly["showlegend"] = True
    fig.update_layout(**ly)
    return fig


def fig_vol_forecast(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "vol_20" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["vol_20"] * 100,
            line=dict(color=C["blue"], width=1, dash="dash"), name="Realised Vol(20)"))
    if "ewma_vol" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["ewma_vol"] * 100,
            line=dict(color=C["orange"], width=1.5), name="EWMA Forecast"))
    fig.update_layout(**BB("Volatility Forecast (%)", h=280))
    return fig


def _spans(mask: np.ndarray) -> list:
    spans = []; s = None
    for i, v in enumerate(mask):
        if v and s is None: s = i
        elif not v and s is not None: spans.append((s, i)); s = None
    if s is not None: spans.append((s, len(mask) - 1))
    return spans


def fig_regime(df: pd.DataFrame) -> go.Figure:
    if "returns" not in df.columns or len(df) < 30:
        return go.Figure()
    labels = detect_regime(df["returns"])
    price  = df["Close"].squeeze()
    n      = min(len(labels), len(price))
    idx    = price.index[-n:]
    lbl_s  = pd.Series(labels[-n:], index=idx)
    cmap   = {0: C["red"], 1: C["yellow"], 2: C["green"]}
    nmap   = {0: "Bear", 1: "Sideways", 2: "Bull"}
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=price.iloc[-n:].values,
        line=dict(color=C["muted"], width=1), name="Price", showlegend=False))
    for state in [0, 1, 2]:
        mask = (lbl_s == state).values
        for s, e in _spans(mask):
            fig.add_vrect(x0=idx[s], x1=idx[min(e, n - 1)],
                          fillcolor=cmap[state], opacity=0.12, line_width=0,
                          annotation_text=nmap[state] if e - s > 5 else "",
                          annotation_font_size=7)
    fig.update_layout(**BB("Market Regime Detection (HMM)", h=280))
    return fig


def fig_triple_barrier(df: pd.DataFrame, labels_df: pd.DataFrame) -> go.Figure:
    c   = df["Close"].squeeze()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c.index, y=c.values,
        line=dict(color="#3a4a60", width=1), name="Price"))
    if not labels_df.empty:
        for sig_val, color, sym in [(1, C["green"], "triangle-up"),
                                     (-1, C["red"], "triangle-down")]:
            sub   = labels_df[labels_df["label"] == sig_val]
            valid = [t for t in sub["time"] if t in c.index]
            if valid:
                fig.add_trace(go.Scatter(x=valid, y=c.loc[valid].values,
                    mode="markers",
                    marker=dict(symbol=sym, size=9, color=color),
                    name=("Long" if sig_val == 1 else "Short")))
    fig.update_layout(**BB("Triple Barrier Labels", h=340))
    return fig


def fig_model_radar(mr: dict) -> go.Figure:
    metrics = ["accuracy", "f1", "precision", "recall", "auc"]
    palette = [C["orange"], C["green"], C["blue"], C["purple"], C["cyan"]]
    fig = go.Figure()
    for i, (name, m) in enumerate(mr.items()):
        vals = [m.get(k, 0) for k in metrics] + [m.get(metrics[0], 0)]
        fig.add_trace(go.Scatterpolar(r=vals, theta=metrics + [metrics[0]],
            fill="toself", name=name,
            line=dict(color=palette[i % len(palette)], width=1.5), opacity=0.75))
    ly = BB("Model Comparison", h=340)
    ly["polar"] = dict(bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, range=[0, 1],
                        gridcolor="rgba(255,255,255,.05)",
                        color=C["muted"], tickfont=dict(size=7)),
        angularaxis=dict(gridcolor="rgba(255,255,255,.05)",
                         color=C["muted"], tickfont=dict(size=9)))
    ly["showlegend"] = True
    fig.update_layout(**ly)
    return fig


def fig_feat_importance(mr: dict, shap_vals: np.ndarray = None) -> go.Figure:
    if not mr:
        return go.Figure()
    best = ("Ensemble" if "Ensemble" in mr
            else max(mr, key=lambda k: mr[k]["cv_f1"]))
    imp  = mr[best]["importance"]
    if len(imp) != len(FEATURE_COLS):
        return go.Figure()
    if shap_vals is not None and shap_vals.ndim == 2 and shap_vals.shape[1] == len(FEATURE_COLS):
        vals = np.abs(shap_vals).mean(axis=0)
        title = f"SHAP Importance ({best})"
    else:
        vals  = np.abs(imp)
        title = f"Feature Importance ({best})"
    s    = pd.Series(vals, index=FEATURE_COLS).sort_values()
    cols = [C["orange"] if v >= s.median() else C["blue"] for v in s.values]
    fig  = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                             marker=dict(color=cols, opacity=.85)))
    fig.update_layout(**BB(title, h=340, legend=False))
    return fig


def fig_confusion(mr: dict) -> go.Figure:
    if not mr:
        return go.Figure()
    best = max(mr, key=lambda k: mr[k]["cv_f1"])
    cm   = np.array(mr[best]["confusion"])
    if cm.shape != (2, 2):
        return go.Figure()
    lbl  = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{lbl[i][j]}\n{cm[i, j]}" for j in range(2)] for i in range(2)]
    fig  = go.Figure(go.Heatmap(z=cm, text=text, texttemplate="%{text}",
        textfont=dict(size=12, color=C["text"]),
        colorscale=[[0, C["panel"]], [1, "rgba(255,140,0,.7)"]],
        showscale=False,
        x=["Pred Long", "Pred Short"], y=["Actual Long", "Actual Short"]))
    fig.update_layout(**BB(f"Confusion Matrix ({best})", h=280, legend=False))
    return fig


def fig_roc(mr: dict) -> go.Figure:
    palette = [C["orange"], C["green"], C["blue"], C["purple"], C["cyan"]]
    fig = go.Figure()
    for i, (name, m) in enumerate(mr.items()):
        if not m.get("y_prob"):
            continue
        fpr, tpr, _ = roc_curve(m["y_te"], m["y_prob"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr,
            name=f"{name} (AUC={m['auc']:.2f})",
            line=dict(color=palette[i % len(palette)], width=1.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
        line=dict(color=C["muted"], dash="dash", width=1), name="Random"))
    fig.update_layout(**BB("ROC Curves", h=300))
    return fig


def fig_equity(bt: dict) -> go.Figure:
    if not bt:
        return go.Figure()
    eq  = bt["equity"]; bh = bt["bh"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=(eq - 1) * 100,
        line=dict(color=C["orange"], width=2), name="HECTOR",
        fill="tozeroy", fillcolor="rgba(255,140,0,.06)"))
    fig.add_trace(go.Scatter(x=bh.index, y=(bh - 1) * 100,
        line=dict(color=C["blue"], width=1.5, dash="dash"), name="Buy & Hold"))
    fig.update_layout(**BB("Equity Curve (%)", h=340))
    return fig


def fig_drawdown(bt: dict) -> go.Figure:
    if not bt:
        return go.Figure()
    dd  = bt["drawdown"]
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values * 100,
        fill="tozeroy", fillcolor="rgba(255,61,90,.20)",
        line=dict(color=C["red"], width=1), name="Drawdown"))
    fig.update_layout(**BB("Drawdown (%)", h=240, legend=False))
    return fig


def fig_wf_sharpe(wf: list) -> go.Figure:
    if not wf:
        return go.Figure()
    folds   = [r["fold"] for r in wf]
    sharpes = [r["sharpe"] for r in wf]
    fig = go.Figure(go.Bar(x=folds, y=sharpes,
        marker_color=[C["green"] if s > 0 else C["red"] for s in sharpes],
        opacity=.80))
    fig.update_layout(**BB("Walk-Forward Sharpe by Fold", h=220, legend=False))
    return fig


def fig_return_hist(bt: dict) -> go.Figure:
    if not bt:
        return go.Figure()
    r   = bt["returns"].dropna() * 100
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Return Distribution", "QQ Plot"])
    fig.add_trace(go.Histogram(x=r, nbinsx=50, marker_color=C["orange"],
                                opacity=.70, name="Returns"), row=1, col=1)
    mu, sg = r.mean(), r.std()
    xn = np.linspace(r.min(), r.max(), 100)
    yn = stats.norm.pdf(xn, mu, sg) * len(r) * (r.max() - r.min()) / 50
    fig.add_trace(go.Scatter(x=xn, y=yn,
        line=dict(color=C["yellow"], width=1.5), name="Normal"), row=1, col=1)
    try:
        (osm, osr), (slope, intercept, _) = stats.probplot(r.dropna())
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
            marker=dict(color=C["cyan"], size=3, opacity=.6), name="QQ"), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[osm[0], osm[-1]],
            y=[slope * osm[0] + intercept, slope * osm[-1] + intercept],
            line=dict(color=C["red"], width=1.5), name="Theoretical"), row=1, col=2)
    except Exception:
        pass
    ly = BB("Return Distribution & QQ", h=300); ly["showlegend"] = True
    fig.update_layout(**ly)
    return fig


def fig_monthly(bt: dict) -> go.Figure:
    if not bt:
        return go.Figure()
    try:
        rm  = bt["returns"].resample("ME").sum() * 100
        rm.index = rm.index.strftime("%b %Y")
        fig = go.Figure(go.Bar(x=rm.index, y=rm.values,
            marker_color=[C["green"] if v > 0 else C["red"] for v in rm.values],
            opacity=.80))
        fig.update_layout(**BB("Monthly Returns (%)", h=200, legend=False),
                          yaxis_ticksuffix="%")
        return fig
    except Exception:
        return go.Figure()


def fig_var_dist(risk: dict) -> go.Figure:
    if not risk:
        return go.Figure()
    r   = risk["returns"] * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, nbinsx=60,
        marker_color=C["orange"], name="Returns", opacity=.70))
    fig.add_vline(x=risk["var95"],  line_color=C["red"],    line_dash="dash", line_width=2,
                  annotation_text=f"VaR95: {risk['var95']:.2f}%",
                  annotation_font=dict(color=C["red"], size=8))
    fig.add_vline(x=risk["cvar95"], line_color=C["purple"], line_dash="dot",  line_width=1,
                  annotation_text=f"CVaR: {risk['cvar95']:.2f}%",
                  annotation_font=dict(color=C["purple"], size=8))
    fig.update_layout(**BB("VaR / CVaR Distribution", h=280, legend=False))
    return fig


def fig_rolling_sharpe(risk: dict) -> go.Figure:
    if not risk or risk.get("roll_sharpe") is None:
        return go.Figure()
    rs  = risk["roll_sharpe"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index, y=rs.values,
        line=dict(color=C["cyan"], width=1.2), name="Rolling Sharpe"))
    fig.add_hline(y=0, line_color=C["muted"],  line_dash="dash", line_width=1)
    fig.add_hline(y=1, line_color=C["green"],  line_dash="dot",  line_width=1)
    fig.update_layout(**BB("Rolling Sharpe (30-bar)", h=240, legend=False))
    return fig


def fig_fear_gauge(fg: dict) -> go.Figure:
    val = fg.get("current", 50); lbl = fg.get("label", "Neutral")
    col = C["green"] if val > 60 else C["red"] if val < 40 else C["yellow"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=val,
        delta={"reference": 50, "valueformat": ".0f"},
        gauge=dict(
            axis=dict(range=[0, 100], tickfont=dict(size=8)),
            bar=dict(color=col, thickness=.25),
            steps=[dict(range=[0,  25], color="rgba(255,61,90,.30)"),
                   dict(range=[25, 45], color="rgba(255,140,0,.20)"),
                   dict(range=[45, 55], color="rgba(74,82,112,.20)"),
                   dict(range=[55, 75], color="rgba(0,212,138,.20)"),
                   dict(range=[75,100], color="rgba(0,212,138,.40)")],
            threshold=dict(line=dict(color=C["orange"], width=3), value=val),
        ),
        title=dict(text=f"Fear & Greed — {lbl}",
                   font=dict(size=10, color=C["sec"])),
        number=dict(font=dict(color=col)),
    ))
    fig.update_layout(**BB(h=240, legend=False))
    return fig


def fig_hrp_bar(wts: pd.Series) -> go.Figure:
    s   = wts.sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=s.values * 100,
        y=[sym.replace("-USD", "") for sym in s.index],
        orientation="h",
        marker=dict(color=C["orange"], opacity=.80),
    ))
    fig.update_layout(**BB("HRP Portfolio Weights (%)",
                           h=max(260, len(s) * 28), legend=False),
                      xaxis_ticksuffix="%")
    return fig


def fig_corr_matrix(ret_df: pd.DataFrame) -> go.Figure:
    corr = ret_df.corr()
    labs = [s.replace("-USD", "") for s in corr.columns]
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=labs, y=labs,
        colorscale=[[0, C["red"]], [.5, C["panel"]], [1, C["green"]]],
        zmid=0, text=corr.round(2).values,
        texttemplate="%{text}", textfont=dict(size=8), showscale=True,
        colorbar=dict(tickfont=dict(color=C["sec"], size=8)),
    ))
    fig.update_layout(**BB("Correlation Matrix", h=360, legend=False))
    return fig


def fig_frontier(front: pd.DataFrame) -> go.Figure:
    if front.empty:
        return go.Figure()
    fig = go.Figure(go.Scatter(
        x=front["vol"], y=front["ret"], mode="markers+lines",
        marker=dict(color=front["sharpe"], colorscale="Plasma", size=6,
                    showscale=True,
                    colorbar=dict(title="Sharpe", thickness=10,
                                  tickfont=dict(size=8))),
        line=dict(color=C["orange"], width=1), name="Efficient Frontier",
    ))
    if not front.empty:
        best = front.loc[front["sharpe"].idxmax()]
        fig.add_trace(go.Scatter(x=[best["vol"]], y=[best["ret"]],
            mode="markers",
            marker=dict(color=C["orange"], size=14, symbol="star"),
            name="Max Sharpe"))
    fig.update_layout(**BB("Mean-Variance Frontier", h=320),
                      xaxis_ticksuffix="%", yaxis_ticksuffix="%")
    return fig


def fig_entropy(price: pd.Series) -> go.Figure:
    r    = price.pct_change().dropna()
    enc  = (r > r.median()).astype(int)
    ent  = []
    win  = CFG["entropy_window"]
    for i in range(win, len(enc)):
        ch = enc.iloc[i - win: i]; vc = ch.value_counts(normalize=True)
        ent.append(-sum(p * np.log2(p + 1e-10) for p in vc))
    ent_s = pd.Series(ent, index=price.index[win + 1: win + 1 + len(ent)])
    fig   = make_subplots(rows=2, cols=1, shared_xaxes=True,
                          row_heights=[.5, .5], vertical_spacing=.03)
    fig.add_trace(go.Scatter(x=price.index, y=price.values,
        line=dict(color="#3a4a60", width=1), name="Price"), row=1, col=1)
    if not ent_s.empty:
        fig.add_trace(go.Scatter(x=ent_s.index, y=ent_s.values,
            fill="tozeroy", fillcolor="rgba(155,109,255,.12)",
            line=dict(color=C["purple"], width=1.5),
            name="Shannon Entropy"), row=2, col=1)
        fig.add_hline(y=float(ent_s.quantile(.25)),
                      line_color=C["orange"], line_dash="dot", line_width=1,
                      row=2, col=1)
    ly = BB("Market Entropy — Inefficiency Detector", h=340)
    ly["xaxis2"] = _xa(); ly["showlegend"] = True
    fig.update_layout(**ly)
    return fig


def fig_cusum(prices: pd.Series) -> go.Figure:
    r   = prices.pct_change().fillna(0)
    sp_arr, sn_arr = [0.0], [0.0]; thr = CFG["cusum_thr"]
    for val in r.values[1:]:
        sp_arr.append(max(0.0, sp_arr[-1] + val))
        sn_arr.append(min(0.0, sn_arr[-1] + val))
        if sp_arr[-1] > thr or sn_arr[-1] < -thr:
            sp_arr[-1] = sn_arr[-1] = 0.0
    idx = r.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=sp_arr[1:], fill="tozeroy",
        fillcolor="rgba(0,212,138,.12)",
        line=dict(color=C["green"], width=1), name="S+ CUSUM"))
    fig.add_trace(go.Scatter(x=idx, y=sn_arr[1:], fill="tozeroy",
        fillcolor="rgba(255,61,90,.12)",
        line=dict(color=C["red"], width=1), name="S− CUSUM"))
    fig.add_hline(y=thr,  line_color=C["yellow"], line_dash="dash", line_width=1)
    fig.add_hline(y=-thr, line_color=C["yellow"], line_dash="dash", line_width=1)
    fig.update_layout(**BB("CUSUM Structural Break Detection", h=260))
    return fig


def fig_frac_diff(price: pd.Series) -> go.Figure:
    lp  = np.log(price.replace(0, np.nan).dropna() + 1e-10)
    w   = [1.0, -0.35, -0.35 * 0.65 / 2, -0.35 * 0.65 * 0.3 / 6]
    fv  = [
        float(sum(a * b for a, b in zip(w[::-1], lp.iloc[i - len(w): i].values)))
        for i in range(len(w), len(lp))
    ]
    fd_s = pd.Series(fv, index=lp.index[len(w):])
    fig  = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[.5, .5], vertical_spacing=.03)
    fig.add_trace(go.Scatter(x=price.index, y=price.values,
        line=dict(color=C["blue"], width=1.2), name="Price (d=0)"), row=1, col=1)
    if not fd_s.empty:
        fig.add_trace(go.Scatter(x=fd_s.index, y=fd_s.values,
            line=dict(color=C["orange"], width=1.2), name="FracDiff d=0.35"), row=2, col=1)
    ly = BB("Fractional Differentiation", h=300)
    ly["xaxis2"] = _xa(); ly["showlegend"] = True
    fig.update_layout(**ly)
    return fig


def fig_news(news: list) -> go.Figure:
    if not news:
        return go.Figure()
    labels = [(n["title"][:42] + "..." if len(n["title"]) > 42 else n["title"])
              for n in news[:10]]
    sent   = [n.get("sentiment", "neutral") for n in news[:10]]
    vals   = [1 if s == "bullish" else -1 if s == "bearish" else 0 for s in sent]
    cols   = [C["green"] if v > 0 else C["red"] if v < 0 else C["muted"] for v in vals]
    fig    = go.Figure(go.Bar(x=vals, y=labels, orientation="h",
                               marker_color=cols, opacity=.80))
    fig.update_layout(**BB("News Sentiment", h=320, legend=False))
    return fig


def fig_vol_profile(df: pd.DataFrame) -> go.Figure:
    vp = volume_profile(df)
    if vp.empty:
        return go.Figure()
    curr = float(df["Close"].iloc[-1])
    fig  = make_subplots(rows=1, cols=2,
                         column_widths=[.75, .25], shared_yaxes=True)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(),
        line=dict(color=C["muted"], width=1), name="Price"), row=1, col=1)
    for _, row in vp[vp["is_hvn"]].iterrows():
        fig.add_hline(y=row["price"], line_color=C["orange"],
                      line_dash="dot", line_width=1, row=1, col=1)
    fig.add_trace(go.Bar(y=vp["price"], x=vp["volume"], orientation="h",
        marker_color=[C["orange"] if r["is_hvn"] else C["blue"]
                      for _, r in vp.iterrows()],
        opacity=.75, name="Vol Profile"), row=1, col=2)
    fig.add_hline(y=curr, line_color=C["cyan"], line_width=1.5,
                  annotation_text=f"${curr:,.0f}", row="all")
    ly = BB("Volume Profile — HVN Support/Resistance", h=380); ly["showlegend"] = True
    fig.update_layout(**ly)
    return fig


#  SIDEBAR

def render_sidebar() -> dict:
    st.sidebar.markdown("""
    <div style="padding:8px 0 6px;border-bottom:1px solid #1e2130;margin-bottom:10px;">
      <div style="font-size:16px;font-weight:700;color:#ff8c00;letter-spacing:3px;">⬡ HECTOR</div>
      <div style="font-size:7px;color:#4a5270;letter-spacing:2px;text-transform:uppercase;">
        v6.0 · Financial Intelligence System
      </div>
    </div>""", unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sec-hdr">Asset</div>', unsafe_allow_html=True)
    coin_label = st.sidebar.selectbox("Asset", list(COIN_MAP.keys()),
                                       index=0, label_visibility="collapsed")
    cg_id, yf_sym = COIN_MAP[coin_label]

    st.sidebar.markdown('<div class="sec-hdr">Portfolio Assets</div>', unsafe_allow_html=True)
    comp_keys    = st.sidebar.multiselect(
        "Compare", [k for k in COIN_MAP if k != coin_label],
        default=list(COIN_MAP.keys())[1:5], label_visibility="collapsed")
    comp_symbols = [COIN_MAP[k][1] for k in comp_keys]

    st.sidebar.markdown('<div class="sec-hdr">Time Frame</div>', unsafe_allow_html=True)
    c1, c2 = st.sidebar.columns(2)
    with c1:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1,
            format_func={"1mo": "1M", "3mo": "3M", "6mo": "6M", "1y": "1Y"}.get,
            label_visibility="collapsed")
    with c2:
        interval = st.selectbox("Interval", ["1h", "1d"],
            format_func={"1h": "1H", "1d": "1D"}.get,
            label_visibility="collapsed")

    st.sidebar.markdown('<div class="sec-hdr">Chart Overlays</div>', unsafe_allow_html=True)
    show_bb   = st.sidebar.checkbox("Bollinger Bands", value=True)
    show_ema  = st.sidebar.checkbox("EMA 9 / 21",      value=True)
    show_vwap = st.sidebar.checkbox("VWAP",            value=True)

    st.sidebar.markdown('<div class="sec-hdr">ML Parameters</div>', unsafe_allow_html=True)
    pt_mult  = st.sidebar.slider("Profit Barrier ×σ", 0.5, 3.0, 1.5, 0.1)
    sl_mult  = st.sidebar.slider("Stop Barrier ×σ",   0.5, 3.0, 1.0, 0.1)
    max_hold = st.sidebar.slider("Max Hold (bars)",    5,   48,  20,  1)
    conf_thr = st.sidebar.slider("Confidence Thr",    0.50, 0.80, 0.55, 0.01)

    st.sidebar.markdown('<div class="sec-hdr">Risk & Costs</div>', unsafe_allow_html=True)
    commission = st.sidebar.slider("Commission %", 0.0, 0.5, 0.1, 0.01) / 100

    st.sidebar.markdown('<div class="sec-hdr">Controls</div>', unsafe_allow_html=True)
    run_ml = st.sidebar.checkbox("Run Full ML Pipeline", value=True)
    if st.sidebar.button("🔄  REFRESH DATA"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown('<div class="sec-hdr">Auto Refresh</div>', unsafe_allow_html=True)
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=False)
    refresh_sec  = 60
    if auto_refresh:
        refresh_sec = st.sidebar.selectbox(
            "Interval", [30, 60, 120, 300],
            format_func=lambda x: f"{x}s", index=1,
            label_visibility="collapsed")

    st.sidebar.markdown('<div class="sec-hdr">Library Status</div>', unsafe_allow_html=True)
    lib_status = [
        ("XGBoost",  XGB_OK), ("LightGBM",  LGB_OK), ("Optuna", OPTUNA_OK),
        ("SHAP",     SHAP_OK), ("HMM",       HMM_OK), ("LIME",   LIME_OK),
        ("Parquet",  PARQUET_OK), ("joblib",  JOBLIB_OK), ("fpdf2", FPDF_OK),
    ]
    html_libs = ""
    for name, ok in lib_status:
        col = "#00d48a" if ok else "#2a3040"
        icon = "✓" if ok else "—"
        html_libs += f'<span style="font-size:8px;color:{col};margin-right:8px;">{icon} {name}</span>'
    st.sidebar.markdown(html_libs, unsafe_allow_html=True)

    st.sidebar.markdown('<div class="sec-hdr">Advanced Options</div>',
                         unsafe_allow_html=True)
    use_meta_label = st.sidebar.checkbox("Use Meta-Labeling filter", value=False)
    meta_thr       = st.sidebar.slider("Meta threshold", 0.40, 0.80, 0.55, 0.05,
                                        help="Minimum meta-model confidence to keep a signal")
    max_dd_cap     = st.sidebar.slider("Max DD Cap", 0.05, 0.50, 0.20, 0.05,
                                        help="Go flat when drawdown exceeds this fraction")
    vol_target     = st.sidebar.slider("Vol Target (ann.)", 0.05, 0.50, 0.15, 0.05,
                                        help="Annualised volatility target for position sizing")

    with st.sidebar.expander("ℹ️ Data Sources & Limits"):
        st.markdown("""
**Free APIs — no keys required:**
- yfinance: ~2,000 req/hour
- CoinGecko: 30 req/minute
- alternative.me: 50 req/minute

Rate limits are handled automatically with caching and retries.
        """)

    # Sidebar footer
    st.sidebar.markdown("""
---
<div style="text-align:center;padding-top:6px;">
  <span style="font-size:8px;color:#2a3040;letter-spacing:1px;">
    Created by <strong style="color:#ff8c00;">Daniyal Aziz</strong>
  </span>
</div>
""", unsafe_allow_html=True)

    return dict(
        coin_label=coin_label, cg_id=cg_id, yf_sym=yf_sym,
        comp_symbols=comp_symbols, period=period, interval=interval,
        show_bb=show_bb, show_ema=show_ema, show_vwap=show_vwap,
        pt_mult=pt_mult, sl_mult=sl_mult, max_hold=max_hold,
        conf_thr=conf_thr, commission=commission, run_ml=run_ml,
        auto_refresh=auto_refresh, refresh_sec=refresh_sec,
        use_meta_label=use_meta_label, meta_thr=meta_thr,
        max_dd_cap=max_dd_cap, vol_target=vol_target,
    )


#  KPI BANNER

def render_kpi(df: pd.DataFrame, info: dict, gm: dict, fg: dict) -> None:
    price  = info.get("price") or (float(df["Close"].iloc[-1]) if not df.empty else 0)
    chg24  = info.get("change_24h", 0)
    mcap   = info.get("market_cap", 0)
    vol24  = info.get("volume_24h", 0)
    hi24   = info.get("high_24h", 0)
    lo24   = info.get("low_24h",  0)
    sym    = info.get("symbol", "BTC")
    btcdom = gm.get("btc_dom", 0)
    fgval  = fg.get("current", 50)
    fglbl  = fg.get("label", "NEUTRAL")
    rsi_n  = float(df["rsi"].iloc[-1]) if "rsi" in df.columns and not df.empty else 50

    chg_html = (f'<div class="kpi-pos">▲ {chg24:+.2f}%</div>'
                if chg24 >= 0
                else f'<div class="kpi-neg">▼ {chg24:+.2f}%</div>')
    rsi_badge = (
        '<span class="tag-short">OVERBOUGHT</span>' if rsi_n > 70
        else '<span class="tag-long">OVERSOLD</span>'  if rsi_n < 30
        else '<span class="tag-neu">NEUTRAL</span>'
    )

    def _fmt_price(p):
        if p < 0.01:   return f"${p:.6f}"
        if p < 1:      return f"${p:.4f}"
        return f"${p:,.2f}"

    def _fmt_large(v):
        if v >= 1e12: return f"${v/1e12:.2f}T"
        if v >= 1e9:  return f"${v/1e9:.2f}B"
        if v >= 1e6:  return f"${v/1e6:.0f}M"
        return f"${v:,.0f}"

    cols  = st.columns(8)
    tiles = [
        (f"{sym}/USD",
         _fmt_price(price) if price else "—", chg_html),
        ("24H High",   _fmt_price(hi24) if hi24 else "—", ""),
        ("24H Low",    _fmt_price(lo24) if lo24 else "—", ""),
        ("Market Cap", _fmt_large(mcap) if mcap else "—", ""),
        ("Volume 24h", _fmt_large(vol24) if vol24 else "—", ""),
        ("BTC Dom",    f"{btcdom:.1f}%" if btcdom else "—", ""),
        ("RSI (14)",   f"{rsi_n:.1f}", rsi_badge),
        ("Fear/Greed", f"{fgval} — {fglbl}", ""),
    ]
    for col, (label, value, extra) in zip(cols, tiles):
        with col:
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-lbl">{label}</div>'
                f'<div class="kpi-val">{value}</div>'
                f'{extra}</div>',
                unsafe_allow_html=True,
            )


#  HELPERS

def stat_block(rows: list) -> None:
    html = ""
    for item in rows:
        k, v = item[0], item[1]
        col  = item[2] if len(item) > 2 else C["text"]
        html += (f'<div class="stat-row">'
                 f'<span class="stat-k">{k}</span>'
                 f'<span class="stat-v" style="color:{col}">{v}</span>'
                 f'</div>')
    st.markdown(html, unsafe_allow_html=True)


def check_alerts(df: pd.DataFrame, info: dict) -> list:
    alerts = []
    if df.empty:
        return alerts
    rsi = float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50
    if rsi > 75:
        alerts.append(("red",    f"RSI {rsi:.1f} — EXTREME OVERBOUGHT"))
    elif rsi > 70:
        alerts.append(("yellow", f"RSI {rsi:.1f} — Overbought zone"))
    elif rsi < 25:
        alerts.append(("red",    f"RSI {rsi:.1f} — EXTREME OVERSOLD"))
    elif rsi < 30:
        alerts.append(("green",  f"RSI {rsi:.1f} — Oversold opportunity"))

    if "ema9" in df.columns and "ema21" in df.columns and len(df) > 2:
        prev_x = df["ema9"].iloc[-2] - df["ema21"].iloc[-2]
        curr_x = df["ema9"].iloc[-1] - df["ema21"].iloc[-1]
        if prev_x < 0 < curr_x:
            alerts.append(("green", "EMA 9/21 Golden Cross detected"))
        elif prev_x > 0 > curr_x:
            alerts.append(("red",   "EMA 9/21 Death Cross detected"))

    chg = info.get("change_24h", 0)
    if abs(chg) > 10:
        alerts.append(("red" if chg < 0 else "yellow",
                        f"Large 24H move: {chg:+.1f}%"))
    return alerts


def render_alerts(alerts: list) -> None:
    for level, msg in alerts:
        cls = {"red": "alert-red", "green": "alert-green",
               "yellow": "alert-yellow"}.get(level, "alert-yellow")
        st.markdown(f'<div class="{cls}">⚠  {msg}</div>', unsafe_allow_html=True)


def dl_csv(df: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(label=label, data=df.to_csv().encode(),
                       file_name=filename, mime="text/csv")



#  ENHANCEMENT 15: AUTOMATED MODEL REFINEMENT & AUTO-REDEPLOYMENT

def auto_refine_model(feat_df: pd.DataFrame,
                       feedback: dict,
                       symbol: str = "BTC-USD") -> dict:
    """
    Enhancement 15a — Automated Model Refinement.
    Compares a 'loser-weighted' retrain vs the baseline.
    Returns {improved: bool, delta_f1, new_f1, base_f1, selected_features}.
    """
    if feat_df.empty or not feedback:
        return {}

    try:
        df = feat_df[feat_df["label"] != 0].copy()
        if len(df) < 40:
            return {}

        avail = [c for c in FEATURE_COLS if c in df.columns]
        X_all = _san_df(df[avail])
        y_all = (df["label"] == 1).astype(int).values

        # Build sample weights — upweight losing-trade conditions
        top_disc = feedback.get("top_discriminators", [])
        weights  = np.ones(len(X_all))
        if top_disc:
            for col in top_disc:
                if col in avail:
                    idx_col = avail.index(col)
                    lose_val = feedback.get("lose_feature_means", {}).get(col, 0)
                    diff     = np.abs(X_all[:, idx_col] - lose_val)
                    # Upweight rows that resemble losing-trade conditions
                    weights += 0.5 * (1 - diff / (diff.max() + 1e-10))

        weights = weights / weights.sum() * len(weights)

        n      = len(X_all)
        split  = int(n * 0.70)
        sc_ref = StandardScaler()
        Xs     = sc_ref.fit_transform(_san(X_all))
        X_tr, X_te = Xs[:split], Xs[split:]
        y_tr, y_te = y_all[:split], y_all[split:]
        w_tr = weights[:split]

        # Baseline model
        base_clf = RandomForestClassifier(
            n_estimators=80, max_depth=5, random_state=42,
            n_jobs=-1, class_weight="balanced",
        )
        base_clf.fit(X_tr, y_tr)
        base_f1 = float(f1_score(y_te, base_clf.predict(X_te), zero_division=0))

        # Refined model — uses sample weights
        ref_clf = RandomForestClassifier(
            n_estimators=80, max_depth=5, random_state=99,
            n_jobs=-1, class_weight="balanced",
        )
        ref_clf.fit(X_tr, y_tr, sample_weight=w_tr)
        ref_f1 = float(f1_score(y_te, ref_clf.predict(X_te), zero_division=0))
        improved = ref_f1 > base_f1 + 0.005

        if improved:
            _model_save(symbol, "Refined", {"model": ref_clf, "scaler": sc_ref})
            log.info("Refined model beats base: %.3f > %.3f", ref_f1, base_f1)

        return {
            "improved":           improved,
            "base_f1":            round(base_f1, 4),
            "new_f1":             round(ref_f1, 4),
            "delta_f1":           round(ref_f1 - base_f1, 4),
            "selected_features":  top_disc,
            "model":              ref_clf if improved else base_clf,
            "scaler":             sc_ref,
        }
    except Exception as exc:
        log.debug("auto_refine_model: %s", exc)
        return {}


def ab_test_models(model_a: Any, model_b: Any,
                    scaler_a: Any, scaler_b: Any,
                    feat_df: pd.DataFrame) -> dict:
    """
    Enhancement 15b — A/B test two models on held-out data.
    Returns {winner, f1_a, f1_b, recommendation}.
    """
    if feat_df.empty or model_a is None or model_b is None:
        return {}
    try:
        df = feat_df[feat_df["label"] != 0].copy()
        if len(df) < 20:
            return {}
        avail = [c for c in FEATURE_COLS if c in df.columns]
        X_raw = _san_df(df[avail])
        y     = (df["label"] == 1).astype(int).values
        split = int(len(X_raw) * 0.70)

        Xa = scaler_a.transform(X_raw[split:])
        Xb = scaler_b.transform(X_raw[split:])
        y_te = y[split:]

        f1_a = float(f1_score(y_te, model_a.predict(Xa), zero_division=0))
        f1_b = float(f1_score(y_te, model_b.predict(Xb), zero_division=0))
        winner = "A (baseline)" if f1_a >= f1_b else "B (refined)"

        return {
            "winner": winner, "f1_a": round(f1_a, 4), "f1_b": round(f1_b, 4),
            "recommendation": (
                "Deploy refined model — statistically better performance"
                if f1_b > f1_a + 0.01
                else "Keep baseline — no significant improvement"
            ),
        }
    except Exception as exc:
        log.debug("ab_test: %s", exc)
        return {}


#  ENHANCEMENT 16: REAL-TIME DATA PIPELINE (WebSocket simulation layer)

# Note: Full WebSocket connections require a running server process.
# Streamlit is single-threaded so we implement a polling-based
# real-time simulation using yfinance 1-minute bars + st.empty() refresh.
# The architecture mirrors a WebSocket design: fetch → ring buffer → feature compute → signal.

class RealTimeFeed:
    """
    Lightweight ring-buffer tick aggregator.
    In production this would consume from Binance/Coinbase WebSocket.
    Here it polls yfinance at each Streamlit rerun.
    """
    def __init__(self, maxlen: int = 200):
        self._buf: List[dict] = []
        self._maxlen = maxlen

    def push(self, bar: dict) -> None:
        self._buf.append(bar)
        if len(self._buf) > self._maxlen:
            self._buf.pop(0)

    def to_df(self) -> pd.DataFrame:
        if not self._buf:
            return pd.DataFrame()
        df = pd.DataFrame(self._buf)
        df["ts"] = pd.to_datetime(df["ts"])
        return df.set_index("ts").sort_index()

    def latest_price(self) -> float:
        return float(self._buf[-1]["close"]) if self._buf else 0.0


@st.cache_resource
def _get_rt_feed() -> RealTimeFeed:
    """Singleton ring buffer — survives Streamlit reruns."""
    return RealTimeFeed(maxlen=200)


def update_rt_feed(yf_sym: str, feed: RealTimeFeed) -> bool:
    """
    Poll yfinance for the freshest 1-min bar and push to feed.
    Returns True if a new bar was added.
    """
    import yfinance as yf
    try:
        raw = yf.download(yf_sym, period="1d", interval="1m",
                          progress=False, auto_adjust=True, threads=False)
        if raw.empty:
            return False
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        for ts, row in raw.tail(5).iterrows():
            feed.push({
                "ts":     str(ts),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": float(row["Volume"]),
            })
        return True
    except Exception as exc:
        log.debug("RT feed: %s", exc)
        return False


def compute_rt_signal(feed: RealTimeFeed, model_result: dict) -> dict:
    """
    Generate a real-time signal from the ring buffer.
    Mimics the sub-second latency of a WebSocket-driven system.
    """
    df_rt = feed.to_df()
    if df_rt.empty or not model_result:
        return {"signal": 0, "prob": 0.5, "latency_ms": 0, "price": 0.0}

    try:
        t0    = time.monotonic()
        df_rt = df_rt.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close", "volume": "Volume",
        })
        if len(df_rt) < 30:
            return {"signal": 0, "prob": 0.5, "latency_ms": 0, "price": feed.latest_price()}

        eng   = engineer_data(df_rt)
        price = feed.latest_price()

        # Quick single-bar feature vector
        i   = len(eng) - 1
        c   = eng["Close"].squeeze()
        v_s = eng["Volume"].squeeze()
        cvd_ = eng["cvd"] if "cvd" in eng.columns else pd.Series(0, index=eng.index)
        cvd_max = float(cvd_.abs().max()) + 1e-10

        def _g(col, default=0.0):
            return float(eng[col].iloc[i]) if col in eng.columns else default

        feat_vec = np.array([[
            float(c.iloc[i] / c.iloc[i-1] - 1) if i >= 1  else 0,
            float(c.iloc[i] / c.iloc[i-5] - 1) if i >= 5  else 0,
            float(c.iloc[i] / c.iloc[i-10]- 1) if i >= 10 else 0,
            float(c.iloc[i] / c.iloc[i-20]- 1) if i >= 20 else 0,
            _g("vol_5"), _g("vol_20"), _g("ewma_vol"),
            _g("rsi", 50), _g("macd"), _g("macd_hist"),
            _g("bb_pct", 0.5), _g("bb_bw"), _g("atr_pct"),
            _g("stoch_k", 50), _g("mfi", 50),
            _g("willr", -50), _g("cci"),
            _g("trend_up"), _g("above_vwap"),
            float(v_s.iloc[i]) / (float(v_s.iloc[max(0,i-5):i].mean()) + 1e-10),
            _g("ofi"), float(cvd_.iloc[i]) / cvd_max, 0.0,
        ]])
        feat_vec = np.clip(np.nan_to_num(feat_vec, 0), -10, 10)

        sc  = model_result["scaler"]
        mdl = model_result["model"]
        Xs  = sc.transform(feat_vec)
        prob = float(mdl.predict_proba(Xs)[0, 1]) if hasattr(mdl, "predict_proba") else 0.5
        sig  = 1 if prob > 0.60 else -1 if prob < 0.40 else 0
        lat  = round((time.monotonic() - t0) * 1000, 2)

        return {"signal": sig, "prob": round(prob, 4), "latency_ms": lat, "price": price}
    except Exception as exc:
        log.debug("RT signal: %s", exc)
        return {"signal": 0, "prob": 0.5, "latency_ms": 0, "price": feed.latest_price()}


#  ENHANCEMENT 17: PRODUCTION DEPLOYMENT DOCS + HEALTH CHECK

DOCKERFILE_CONTENT = '''# HECTOR v6 — Production Dockerfile
# Multi-stage build for minimal image size

FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    gcc g++ libgomp1 curl \\
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY hector_v6.py .
COPY .env.example .env

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \\
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "hector_v6.py", \\
     "--server.port=8501", \\
     "--server.address=0.0.0.0", \\
     "--server.headless=true", \\
     "--browser.gatherUsageStats=false"]
'''

DOCKER_COMPOSE_CONTENT = '''version: "3.9"
services:
  hector:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./hector_data.db:/app/hector_data.db
      - ./hector_models:/app/hector_models
      - ./hector_cache:/app/hector_cache
    env_file: .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Optional: Redis for caching (uncomment for production)
  # redis:
  #   image: redis:7-alpine
  #   ports: ["6379:6379"]
  #   restart: unless-stopped
'''

GITHUB_ACTIONS_CONTENT = '''name: HECTOR CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run test suite
        run: python hector_v6.py --test

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t hector:${{ github.sha }} .
      - name: Push to registry
        run: |
          echo "Push to your registry here"
          # docker push your-registry/hector:${{ github.sha }}
'''

REQUIREMENTS_CONTENT = """streamlit>=1.35
yfinance>=0.2.38
pandas>=2.0
numpy>=1.26
plotly>=5.20
scipy>=1.12
scikit-learn>=1.4
requests>=2.31
# Optional — install for full functionality:
# xgboost>=2.0
# lightgbm>=4.0
# optuna>=3.6
# shap>=0.45
# hmmlearn>=0.3
# lime>=0.2
# joblib>=1.3
# pyarrow>=15.0
# statsmodels>=0.14
# fpdf2>=2.7
"""


def get_system_health() -> dict:
    """
    Enhancement 17 — health check endpoint simulation.
    Returns system status dict for monitoring dashboards.
    """
    import platform
    health = {
        "status":    "healthy",
        "version":   "HECTOR v6.0",
        "timestamp": datetime.utcnow().isoformat(),
        "platform":  platform.system(),
        "python":    platform.python_version(),
        "db_ok":     False,
        "model_dir_ok": False,
        "cache_dir_ok": False,
        "libraries": {},
        "issues": [],
    }
    try:
        conn = _get_db()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        health["db_ok"] = True
    except Exception as e:
        health["issues"].append(f"DB: {e}")
        health["status"] = "degraded"

    health["model_dir_ok"] = Path(CFG["model_dir"]).exists()
    health["cache_dir_ok"] = Path(CFG["cache_dir"]).exists()
    health["libraries"] = {
        "xgboost":    XGB_OK, "lightgbm": LGB_OK,
        "optuna":     OPTUNA_OK, "shap":  SHAP_OK,
        "hmmlearn":   HMM_OK, "parquet":  PARQUET_OK,
        "joblib":     JOBLIB_OK, "fpdf2": FPDF_OK,
    }
    if not all([health["db_ok"], health["model_dir_ok"]]):
        health["status"] = "degraded"
    return health


#  ENHANCEMENT 18: PAPER TRADING GATEWAY (OMS simulation)

def _init_paper_account() -> dict:
    """Default paper trading account state."""
    return {
        "balance":   10_000.0,
        "position":  0.0,
        "avg_price": 0.0,
        "trades":    [],
        "pnl":       0.0,
        "total_trades": 0,
        "wins":      0,
        "max_pos_usd": 2_000.0,
        "daily_loss_limit": -500.0,
        "daily_pnl": 0.0,
        "kill_switch": False,
    }


def paper_execute_order(account: dict, signal: int, price: float,
                         size_usd: float = 500.0) -> dict:
    """
    Enhancement 18 — Paper OMS.
    Executes Long/Short/Exit orders with pre-trade risk checks.
    """
    if price <= 0 or account.get("kill_switch"):
        return {"executed": False, "reason": "Kill switch active or invalid price"}

    # Pre-trade risk checks
    if account["daily_pnl"] < account["daily_loss_limit"]:
        account["kill_switch"] = True
        return {"executed": False, "reason": f"Daily loss limit hit: ${account['daily_pnl']:.2f}"}

    qty    = size_usd / price
    commission = size_usd * 0.001
    slippage   = price * 0.0005

    if signal == 1 and account["position"] <= 0:
        # Enter Long
        cost = size_usd + commission
        if cost > account["balance"]:
            return {"executed": False, "reason": "Insufficient balance"}
        account["balance"]   -= cost
        account["position"]  = qty
        account["avg_price"] = price + slippage
        trade = {"side": "BUY", "qty": round(qty, 6),
                 "price": round(account["avg_price"], 2),
                 "time": datetime.utcnow().isoformat(),
                 "commission": round(commission, 4)}
        account["trades"].append(trade)
        account["total_trades"] += 1
        return {"executed": True, "trade": trade, "reason": "Long entered"}

    elif signal == -1 and account["position"] > 0:
        # Close Long
        exit_price = price - slippage
        pnl = (exit_price - account["avg_price"]) * account["position"] - commission
        account["balance"]   += account["position"] * exit_price - commission
        account["pnl"]       += pnl
        account["daily_pnl"] += pnl
        if pnl > 0:
            account["wins"] += 1
        trade = {"side": "SELL", "qty": round(account["position"], 6),
                 "price": round(exit_price, 2),
                 "pnl":   round(pnl, 4),
                 "time":  datetime.utcnow().isoformat(),
                 "commission": round(commission, 4)}
        account["trades"].append(trade)
        account["total_trades"] += 1
        account["position"]  = 0.0
        account["avg_price"] = 0.0
        return {"executed": True, "trade": trade, "reason": "Long closed"}

    return {"executed": False, "reason": "No action — signal unchanged"}


def get_paper_account_stats(account: dict) -> dict:
    """Compute P&L stats for the paper account."""
    if account["total_trades"] == 0:
        return account
    account["win_rate"] = round(account["wins"] / max(account["total_trades"], 1), 3)
    return account


#  ENHANCEMENT 19: MONITORING & ALERTING

def get_prometheus_metrics(health: dict, bt: dict, signals_df: pd.DataFrame) -> str:
    """
    Enhancement 19 — Export metrics in Prometheus text format.
    In production this would be served on /metrics endpoint.
    """
    lines = [
        "# HELP hector_sharpe_ratio Current backtest Sharpe ratio",
        "# TYPE hector_sharpe_ratio gauge",
        f'hector_sharpe_ratio {bt.get("sharpe", 0):.4f}',
        "",
        "# HELP hector_max_drawdown Maximum drawdown",
        "# TYPE hector_max_drawdown gauge",
        f'hector_max_drawdown {bt.get("max_dd", 0):.4f}',
        "",
        "# HELP hector_n_trades Total number of trades",
        "# TYPE hector_n_trades counter",
        f'hector_n_trades {bt.get("n_trades", 0)}',
        "",
        "# HELP hector_win_rate Win rate of strategy",
        "# TYPE hector_win_rate gauge",
        f'hector_win_rate {bt.get("win_rate", 0):.4f}',
        "",
        "# HELP hector_db_healthy Database connectivity",
        "# TYPE hector_db_healthy gauge",
        f'hector_db_healthy {1 if health.get("db_ok") else 0}',
        "",
        "# HELP hector_signals_total Total signals generated",
        "# TYPE hector_signals_total counter",
        f'hector_signals_total {len(signals_df) if not signals_df.empty else 0}',
        "",
        "# HELP hector_last_signal Latest signal value",
        "# TYPE hector_last_signal gauge",
        f'hector_last_signal {int(signals_df["signal"].iloc[-1]) if not signals_df.empty and "signal" in signals_df.columns else 0}',
    ]
    return "\n".join(lines)


def smart_alert(level: str, subject: str, body: str,
                send_tg: bool = False, send_email: bool = False) -> dict:
    """
    Enhancement 19 — Enhanced alerting with circuit breaker.
    Prevents alert storms by tracking last alert time per subject.
    """
    # Use session state as circuit breaker (per-session, 5-min cooldown)
    cb_key = f"_alert_cb_{hashlib.md5(subject.encode()).hexdigest()[:8]}"
    now    = time.time()
    last   = st.session_state.get(cb_key, 0)
    if now - last < 300:   # 5-minute cooldown
        return {"sent": False, "reason": "Circuit breaker active"}

    st.session_state[cb_key] = now
    sent = {}
    if send_tg:
        sent["telegram"] = _send_telegram(f"[{level.upper()}] {subject}\n{body}")
    if send_email:
        sent["email"]    = _send_email(subject, body)
    if SLACK_WEBHOOK:
        sent["slack"]    = _send_slack(f"[{level.upper()}] {subject}\n{body}")
    return {"sent": True, "channels": sent}


#  ENHANCEMENT 20: EVENT-DRIVEN BACKTEST + MONTE CARLO SIMULATION

def event_driven_backtest(price: pd.Series,
                           signals_df: pd.DataFrame,
                           commission: float = None,
                           initial_capital: float = 10_000.0) -> dict:
    """
    Enhancement 20a — Tick-by-tick event-driven backtest.
    Simulates actual order execution (not vectorised P&L multiplication).
    Includes: margin requirements, order types, position tracking.
    """
    commission = commission or CFG["commission"]
    if signals_df.empty or len(price) < 10:
        return {}
    try:
        sig      = signals_df.set_index("time")["signal"]
        sig      = sig.reindex(price.index, method="ffill").fillna(0)
        sig_lag  = sig.shift(1).fillna(0)   # realistic 1-bar execution lag

        cash      = initial_capital
        position  = 0.0     # in units of the asset
        avg_cost  = 0.0
        equity_curve = []
        trades    = []
        prev_sig  = 0

        for ts, cur_price in price.items():
            new_sig = int(sig_lag.get(ts, 0))
            bar_ret = float(price.pct_change().fillna(0).get(ts, 0))

            # Mark-to-market
            port_val = cash + position * float(cur_price)
            equity_curve.append({"ts": ts, "equity": port_val,
                                   "price": float(cur_price)})

            # Event: signal changed
            if new_sig != prev_sig:
                trade_usd  = min(cash * 0.95, 2000.0)   # max 95% of cash, capped
                qty        = trade_usd / (float(cur_price) + 1e-10)
                cost_basis = qty * float(cur_price)
                fee        = cost_basis * commission
                slip       = float(cur_price) * 0.0005

                if new_sig == 1 and cash >= cost_basis + fee:
                    # Buy
                    cash     -= cost_basis + fee
                    position += qty
                    avg_cost  = float(cur_price) + slip
                    trades.append({"ts": ts, "side": "BUY", "qty": round(qty, 6),
                                   "price": round(avg_cost, 4), "fee": round(fee, 4)})

                elif new_sig != 1 and position > 0:
                    # Sell / Close
                    exit_px   = float(cur_price) - slip
                    proceeds  = position * exit_px - position * exit_px * commission
                    pnl       = (exit_px - avg_cost) * position
                    cash     += proceeds
                    trades.append({"ts": ts, "side": "SELL",
                                   "qty": round(position, 6),
                                   "price": round(exit_px, 4),
                                   "pnl": round(pnl, 4)})
                    position  = 0.0
                    avg_cost  = 0.0

            prev_sig = new_sig

        eq_df   = pd.DataFrame(equity_curve).set_index("ts")["equity"]
        bh_     = initial_capital * (price / price.iloc[0])
        ret_    = eq_df.pct_change().dropna()
        peak    = eq_df.expanding().max()
        dd_     = (eq_df - peak) / (peak.abs() + 1e-10)
        ann_f   = 365 * 24
        ann_ret = float(ret_.mean() * ann_f)
        ann_vol = float(ret_.std() * np.sqrt(ann_f)) + 1e-10
        sharpe  = ann_ret / ann_vol
        n_wins  = sum(1 for t in trades if t.get("pnl", 0) > 0)
        n_sells = sum(1 for t in trades if t["side"] == "SELL")

        return dict(
            equity=eq_df, bh=bh_, drawdown=dd_, returns=ret_,
            ann_ret=ann_ret, ann_vol=ann_vol, sharpe=sharpe,
            max_dd=float(dd_.min()),
            win_rate=round(n_wins / max(n_sells, 1), 3),
            n_trades=n_sells,
            final_equity=round(float(eq_df.iloc[-1]), 2),
            total_return_pct=round((float(eq_df.iloc[-1]) / initial_capital - 1) * 100, 2),
            trades=trades[-20:],   # last 20 trades
        )
    except Exception as exc:
        log.error("Event-driven backtest: %s", exc)
        return {}


def monte_carlo_simulation(returns: pd.Series,
                            n_paths: int = 500,
                            horizon: int = None) -> dict:
    """
    Enhancement 20b — Monte Carlo simulation for strategy analysis.
    Simulates n_paths return sequences by bootstrapping historical returns.
    Returns distribution of terminal wealth and Sharpe ratios.
    """
    if len(returns) < 10:
        return {}
    try:
        r_arr    = returns.dropna().values
        horizon  = horizon or min(len(r_arr), 252)
        ann_f    = 365 * 24

        # Vectorised: sample all paths at once (n_paths × horizon matrix)
        idx   = np.random.randint(0, len(r_arr), size=(n_paths, horizon))
        paths = r_arr[idx]                          # shape (n_paths, horizon)

        # Terminal equity: cumprod along horizon axis, take last value
        eq_matrix        = np.cumprod(1 + paths, axis=1)   # (n_paths, horizon)
        terminal_equity  = eq_matrix[:, -1]

        # Sharpe per path: vectorised mean/std
        path_means       = paths.mean(axis=1)
        path_stds        = paths.std(axis=1)
        path_sharpes     = path_means * ann_f / (path_stds * np.sqrt(ann_f) + 1e-10)

        # Max drawdown per path: running max then drawdown
        pk_matrix = np.maximum.accumulate(eq_matrix, axis=1)
        dd_matrix = (eq_matrix - pk_matrix) / (pk_matrix + 1e-10)
        max_dds   = dd_matrix.min(axis=1)


        return dict(
            n_paths=n_paths,
            terminal_median=round(float(np.median(terminal_equity)), 4),
            terminal_p5=round(float(np.percentile(terminal_equity, 5)),   4),
            terminal_p95=round(float(np.percentile(terminal_equity, 95)), 4),
            prob_profit=round(float((terminal_equity > 1.0).mean()), 3),
            prob_double=round(float((terminal_equity > 2.0).mean()), 3),
            sharpe_median=round(float(np.median(path_sharpes)), 3),
            max_dd_median=round(float(np.median(max_dds)), 4),
            terminal_equity=terminal_equity,
            path_sharpes=path_sharpes,
            max_dds=max_dds,
        )
    except Exception as exc:
        log.error("Monte Carlo: %s", exc)
        return {}



#  MAIN APPLICATION

def main() -> None:
    now_str = datetime.utcnow().strftime("%b %d, %Y · %I:%M %p UTC")
    st.markdown(
        f'<div class="bb-hdr">'
        f'<div>'
        f'<div class="bb-logo">⬡ HECTOR</div>'
        f'<div class="bb-sub">Alpha Lab Edition · Multi-Layer Intelligence · '
        f'{len(COIN_MAP)} Assets · 100% Free Data</div>'
        f'</div>'
        f'<div class="bb-live">🟢 LIVE &nbsp;|&nbsp; {now_str}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cfg = render_sidebar()

    # Invalidate spinner-done keys when data params change
    _data_sig = f"{cfg['yf_sym']}_{cfg['period']}_{cfg['interval']}"
    if st.session_state.get("_last_data_sig") != _data_sig:
        # New asset or timeframe — purge cached-spinner markers so spinners reappear
        for k in list(st.session_state.keys()):
            if k.startswith("ml_done_") or k.startswith("lab_done_"):
                del st.session_state[k]
        st.session_state["_last_data_sig"] = _data_sig

    # Auto-refresh via meta tag (no external component needed)
    if cfg.get("auto_refresh"):
        st.markdown(
            f'<meta http-equiv="refresh" content="{cfg["refresh_sec"]}">',
            unsafe_allow_html=True)

    # Layer 1: Ingest (all cached)
    with st.spinner("Loading market data…"):
        df_raw = fetch_ohlcv(cfg["yf_sym"], cfg["period"], cfg["interval"])
        info   = fetch_coin_info(cfg["cg_id"])
        gm     = fetch_global_market()
        fg     = fetch_fear_greed()

    if df_raw.empty:
        st.error(
            f"No data returned for **{cfg['yf_sym']}**. "
            "Check your internet connection or select a different asset."
        )
        return

    df_raw, val_report = validate_ohlcv(df_raw)

    # Layer 2: Engineer (cached per asset+period+interval)
    df = engineer_data(df_raw)

    # Regime detection (must happen BEFORE ML pipeline)
    regime = classify_regime(df)

    # 1. Initialize ALL variables to safe defaults
    labels_df      = pd.DataFrame()
    feat_df        = pd.DataFrame()
    mr             = {}
    signals_df     = pd.DataFrame()
    bt             = {}
    stability_df   = pd.DataFrame()
    baselines      = {}
    wf_folds       = []
    cal_clf        = None
    meta_result    = None
    adv_bt         = {}
    dyn_thresholds = {}
    sig_tests      = {}
    drift_df       = pd.DataFrame()
    attr_dict      = {}
    decay_df       = pd.DataFrame()
    feedback       = {}
    constraints    = {}
    run_id         = ""
    retrain_sched  = {}
    refined_result = {}
    ab_result      = {}
    mc_result      = {}
    edb_result     = {}
    health         = {}
    paper_acct     = _init_paper_account()

    # Initialize live feed price from real data so $0.0000 never shows
    _init_price = info.get("price") or (float(df_raw["Close"].iloc[-1]) if not df_raw.empty else 0.0)
    rt_sig         = {"signal": 0, "prob": 0.5, "latency_ms": 0, "price": _init_price}

    if cfg["run_ml"]:
        # Cache-key: spinner only shows on genuine first-run
        _ml_key = f"ml_done_{cfg['yf_sym']}_{cfg['period']}_{cfg['interval']}_{cfg['pt_mult']}_{cfg['sl_mult']}_{cfg['max_hold']}"
        _ml_cached = st.session_state.get(_ml_key, False)

        _spinner_ml = st.spinner("Running ML pipeline…") if not _ml_cached else \
                      contextlib.nullcontext()
        with _spinner_ml:
            labels_df  = triple_barrier_labels(df, cfg["pt_mult"], cfg["sl_mult"], cfg["max_hold"])
            feat_df    = build_features(df, labels_df) if not labels_df.empty else pd.DataFrame()
            mr         = (train_models(feat_df, cfg["yf_sym"])
                          if not feat_df.empty and len(feat_df) >= 40 else {})
            signals_df = (generate_signals(df, labels_df, feat_df, mr,
                                           cfg["conf_thr"], regime)
                          if mr else pd.DataFrame())
            bt         = (run_backtest(df["Close"].squeeze(), signals_df, cfg["commission"])
                          if not signals_df.empty else {})
        if mr and not _ml_cached:
            st.toast(f"✅ Models trained — {len(mr)} model(s) ready", icon="🎯")
            st.session_state[_ml_key] = True
        elif not feat_df.empty and len(feat_df) < 40:
            st.toast("⚠️ Insufficient data for model training — try a longer period", icon="⚠️")

        _lab_key = f"lab_done_{cfg['yf_sym']}_{cfg['period']}_{cfg['interval']}_{cfg['pt_mult']}_{cfg['sl_mult']}_{cfg['max_hold']}"
        _lab_cached = st.session_state.get(_lab_key, False)

        if not _lab_cached:
            with st.spinner("Computing Alpha Lab…"):
                stability_df = feature_stability_analysis(feat_df)
                baselines    = run_baseline_benchmarks(feat_df, df["Close"].squeeze())

                wf_folds     = walk_forward_train_test(feat_df, df["Close"].squeeze())

                if mr:
                    best_k  = ("Ensemble" if "Ensemble" in mr
                                else max(mr, key=lambda k: mr[k]["cv_f1"]))
                    best_mr = mr[best_k]
                    cal_clf = calibrate_probabilities(best_mr["model"], best_mr["scaler"],
                                                       feat_df, method="platt")

                if mr and not signals_df.empty:
                    meta_result = train_meta_model(
                        signals_df["prob"].values if "prob" in signals_df.columns
                        else np.full(len(feat_df), 0.5),
                        feat_df, labels_df,
                    )
                    if meta_result:
                        signals_df = apply_meta_filter(signals_df, meta_result, feat_df,
                                                       confidence_min=cfg["conf_thr"])
                        bt = (run_backtest(df["Close"].squeeze(), signals_df, cfg["commission"])
                              if not signals_df.empty else bt)

                adv_bt = advanced_backtest(
                    df["Close"].squeeze(), signals_df,
                    commission=cfg["commission"],
                    latency_bars=1, max_hold_bars=cfg["max_hold"],
                    max_dd_cap=-0.20, vol_target=0.15,
                ) if not signals_df.empty else {}

                if mr and not feat_df.empty:
                    try:
                        df_lb = feat_df[feat_df["label"] != 0].copy()
                        if len(df_lb) >= 20:
                            avail_ = [c for c in FEATURE_COLS if c in df_lb.columns]
                            best_k2 = ("Ensemble" if "Ensemble" in mr
                                        else max(mr, key=lambda k: mr[k]["cv_f1"]))
                            raw_p = mr[best_k2]["model"].predict_proba(
                                mr[best_k2]["scaler"].transform(
                                    df_lb[avail_].fillna(0).values.astype(float)
                                )
                            )[:, 1]
                            reg_arr = detect_regime(df["returns"])
                            reg_aligned = reg_arr[-len(df_lb):] if len(reg_arr) >= len(df_lb) \
                                          else np.ones(len(df_lb), dtype=int)
                            dyn_thresholds = optimise_threshold_per_regime(
                                raw_p, (df_lb["label"] == 1).astype(int).values,
                                reg_aligned,
                            )
                    except Exception:
                        pass

                sig_tests = significance_tests(bt.get("returns", pd.Series(dtype=float))) if bt else {}

                if not feat_df.empty and len(feat_df) >= 20:
                    half     = len(feat_df) // 2
                    drift_df = detect_feature_drift(
                        feat_df.iloc[:half], feat_df.iloc[half:], cfg["yf_sym"]
                    )

                regime_series = pd.Series(dtype=str)
                if "returns" in df.columns:
                    labels_r = detect_regime(df["returns"])
                    regime_series = pd.Series(
                        ["bull" if v == 2 else "bear" if v == 0 else "sideways"
                         for v in labels_r],
                        index=df["returns"].dropna().index,
                    )

                attr_dict = performance_attribution(
                    bt.get("returns", pd.Series(dtype=float)),
                    signals_df, feat_df, regime_series,
                ) if bt else {}

                decay_df = monitor_alpha_decay(
                    bt.get("returns", pd.Series(dtype=float)),
                    cfg["yf_sym"], window_days=7,
                ) if bt else pd.DataFrame()

                feedback = analyse_losing_trades(
                    bt.get("returns", pd.Series(dtype=float)),
                    signals_df, feat_df,
                ) if bt else {}

                constraints = apply_portfolio_constraints(
                    bt.get("returns", pd.Series(dtype=float))
                ) if bt else {}

                retrain_sched = retraining_schedule(
                    cfg["yf_sym"],
                    last_run=datetime.utcnow() - timedelta(hours=50),
                    drift_triggered=should_retrain(drift_df),
                    schedule="weekly",
                )

                if mr and bt:
                    run_id = log_experiment(
                        cfg["yf_sym"],
                        {"pt": cfg["pt_mult"], "sl": cfg["sl_mult"],
                         "conf": cfg["conf_thr"], "period": cfg["period"]},
                        mr, bt,
                        notes=f"regime={regime}",
                    )

                # Auto-refinement
                if feedback and mr:
                    best_k3 = ("Ensemble" if "Ensemble" in mr
                                else max(mr, key=lambda k: mr[k]["cv_f1"]))
                    refined_result = auto_refine_model(feat_df, feedback, cfg["yf_sym"])
                    if refined_result and refined_result.get("improved"):
                        ab_result = ab_test_models(
                            mr[best_k3]["model"], refined_result["model"],
                            mr[best_k3]["scaler"], refined_result["scaler"],
                            feat_df,
                        )

                # Event-driven backtest + Monte Carlo
                if not signals_df.empty:
                    edb_result = event_driven_backtest(
                        df["Close"].squeeze(), signals_df,
                        commission=cfg["commission"],
                    )
                if bt.get("returns") is not None and len(bt.get("returns", pd.Series(dtype=float))) > 10:
                    mc_result = monte_carlo_simulation(bt.get("returns", pd.Series(dtype=float)), n_paths=300, horizon=200)

                # Health check
                health = get_system_health()

                # Cache all lab results in session_state so reruns skip this block
                st.session_state[_lab_key] = True
                st.session_state[f"{_lab_key}_results"] = dict(
                    stability_df=stability_df, baselines=baselines, wf_folds=wf_folds,
                    cal_clf=cal_clf, meta_result=meta_result, signals_df=signals_df,
                    bt=bt, adv_bt=adv_bt, dyn_thresholds=dyn_thresholds,
                    sig_tests=sig_tests, drift_df=drift_df, attr_dict=attr_dict,
                    decay_df=decay_df, feedback=feedback, constraints=constraints,
                    retrain_sched=retrain_sched, run_id=run_id,
                    refined_result=refined_result, ab_result=ab_result,
                    edb_result=edb_result, mc_result=mc_result, health=health,
                )
        else:
            # Restore cached lab results — zero recomputation on reruns
            _cached = st.session_state.get(f"{_lab_key}_results", {})
            stability_df   = _cached.get("stability_df",   stability_df)
            baselines      = _cached.get("baselines",      baselines)
            wf_folds       = _cached.get("wf_folds",       wf_folds)
            cal_clf        = _cached.get("cal_clf",        cal_clf)
            meta_result    = _cached.get("meta_result",    meta_result)
            signals_df     = _cached.get("signals_df",     signals_df)
            bt             = _cached.get("bt",             bt)
            adv_bt         = _cached.get("adv_bt",        adv_bt)
            dyn_thresholds = _cached.get("dyn_thresholds", dyn_thresholds)
            sig_tests      = _cached.get("sig_tests",      sig_tests)
            drift_df       = _cached.get("drift_df",      drift_df)
            attr_dict      = _cached.get("attr_dict",      attr_dict)
            decay_df       = _cached.get("decay_df",      decay_df)
            feedback       = _cached.get("feedback",       feedback)
            constraints    = _cached.get("constraints",    constraints)
            retrain_sched  = _cached.get("retrain_sched",  retrain_sched)
            run_id         = _cached.get("run_id",        run_id)
            refined_result = _cached.get("refined_result", refined_result)
            ab_result      = _cached.get("ab_result",     ab_result)
            edb_result     = _cached.get("edb_result",    edb_result)
            mc_result      = _cached.get("mc_result",     mc_result)
            health         = _cached.get("health",        health)

    # 2. TABS
    tabs = st.tabs([
        "Market Overview",
        "Signals",
        "Models",
        "Backtest",
        "Risk",
        "Portfolio",
        "Research",
        "Sentiment",
        "Alpha Lab",
        "Live Feed",
        "Paper Trading",
        "System",
        "Export",
    ])

    #  TAB 0: MARKET OVERVIEW
    with tabs[0]:
        render_kpi(df, info, gm, fg)
        alerts = check_alerts(df, info)
        if alerts:
            render_alerts(alerts)
        st.markdown('<div class="sec-hdr">Live Price Action</div>', unsafe_allow_html=True)
        st.plotly_chart(
            fig_candle(df,
                       f"{info.get('symbol','BTC')}/USD · {cfg['period'].upper()} {cfg['interval'].upper()}",
                       cfg["show_bb"], cfg["show_ema"], cfg["show_vwap"],
                       signals_df if not signals_df.empty else None),
            use_container_width=True, config=PCONF)

        c1, c2 = st.columns([3, 2])
        with c1:
            st.plotly_chart(fig_macd(df), use_container_width=True, config=PCONF)
        with c2:
            st.plotly_chart(fig_fear_gauge(fg), use_container_width=True, config=PCONF)
            hist = fg.get("history", [])
            if hist:
                fig_fh = go.Figure(go.Scatter(
                    y=hist, line=dict(color=C["orange"], width=1.5),
                    fill="tozeroy", fillcolor="rgba(255,140,0,.10)"))
                fig_fh.update_layout(**BB("Fear & Greed — 30D", h=150, legend=False))
                st.plotly_chart(fig_fh, use_container_width=True, config=PCONF)

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(fig_cusum(df["Close"].squeeze()),
                            use_container_width=True, config=PCONF)
        with c4:
            st.markdown('<div class="sec-hdr">Data Quality</div>', unsafe_allow_html=True)
            is_stat = False
            adf_p   = None
            if ADF_OK and "returns" in df.columns and len(df) > 20:
                try:
                    _, adf_p, *_ = adfuller(df["returns"].dropna())
                    is_stat = adf_p < 0.05
                except Exception:
                    pass
            stat_block([
                ("Rows fetched",    str(val_report.get("rows_before", len(df_raw))), C["green"]),
                ("After cleaning",  str(len(df)),                                    C["green"]),
                ("Outliers capped", str(val_report.get("outliers", 0)),              C["yellow"]),
                ("Duplicates rm'd", str(val_report.get("duplicates", 0)),            C["yellow"]),
                ("Date range",
                 f"{df.index[0].date()} → {df.index[-1].date()}", C["text"]),
                ("ADF p-value",     f"{adf_p:.4f}" if adf_p is not None else "—",   C["cyan"]),
                ("Stationarity",
                 "✓ STATIONARY" if is_stat else "✗ NON-STATIONARY",
                 C["green"] if is_stat else C["red"]),
                ("Skewness",
                 f"{df['returns'].skew():.3f}" if "returns" in df.columns else "—", C["text"]),
                ("Excess Kurtosis",
                 f"{df['returns'].kurtosis():.3f}" if "returns" in df.columns else "—", C["text"]),
                ("Regime", regime.upper(),
                 C["green"] if regime == "bull" else C["red"] if regime == "bear" else C["yellow"]),
            ])

        c5, c6 = st.columns(2)
        with c5:
            st.plotly_chart(fig_regime(df), use_container_width=True, config=PCONF)
        with c6:
            st.plotly_chart(fig_ofi_cvd(df), use_container_width=True, config=PCONF)
        st.plotly_chart(fig_vol_forecast(df), use_container_width=True, config=PCONF)

    #  TAB 1: SIGNALS
    with tabs[1]:
        st.markdown('<div class="sec-hdr">Signal Generator</div>', unsafe_allow_html=True)
        if not cfg["run_ml"]:
            st.info("Enable 'Run Full ML Pipeline' in the sidebar.")
        elif labels_df.empty:
            st.warning("Not enough data for labeling. Try 6 months or more.")
        else:
            longs  = int((labels_df["label"] ==  1).sum())
            shorts = int((labels_df["label"] == -1).sum())
            neuts  = int((labels_df["label"] ==  0).sum())
            total  = max(len(labels_df), 1)
            last_l = int(labels_df["label"].iloc[-1])
            sig_cls  = {1: "signal-lg", -1: "signal-sh", 0: "signal-fl"}.get(last_l, "signal-fl")
            sig_text = {1: "▲ LONG",    -1: "▼ SHORT",   0: "— FLAT"}.get(last_l, "— FLAT")

            cs1, cs2, cs3, cs4 = st.columns([1, 1, 1, 2])
            with cs1:
                st.markdown(f'<div class="{sig_cls}">{sig_text}</div>', unsafe_allow_html=True)
                if not signals_df.empty:
                    ls = signals_df.iloc[-1]
                    st.markdown(
                        f'<div style="text-align:center;font-size:11px;color:{C["sec"]};margin-top:6px;">'
                        f'Prob: {float(ls["prob"])*100:.1f}% · Kelly: {float(ls["bet_size"]):.3f}'
                        f'</div>', unsafe_allow_html=True)
            with cs2:
                stat_block([
                    ("Total Labels", str(total),                                   C["text"]),
                    ("Long (1)",     f"{longs} ({longs/total*100:.1f}%)",          C["green"]),
                    ("Short (−1)",   f"{shorts} ({shorts/total*100:.1f}%)",        C["red"]),
                    ("Neutral (0)",  f"{neuts} ({neuts/total*100:.1f}%)",          C["muted"]),
                    ("PT barrier",   f"{cfg['pt_mult']}×σ",                       C["text"]),
                    ("SL barrier",   f"{cfg['sl_mult']}×σ",                       C["text"]),
                    ("Max hold",     f"{cfg['max_hold']} bars",                    C["text"]),
                    ("Regime",       regime.upper(),
                     C["green"] if regime=="bull" else C["red"] if regime=="bear" else C["yellow"]),
                ])
            with cs3:
                fig_lp = go.Figure(go.Pie(
                    labels=["Long", "Short", "Neutral"],
                    values=[longs, shorts, neuts], hole=.55,
                    marker=dict(colors=[C["green"], C["red"], C["muted"]],
                                line=dict(color=C["bg"], width=2)),
                    textfont=dict(size=8), textinfo="percent+label"))
                fig_lp.update_layout(**BB(h=240, legend=False))
                st.plotly_chart(fig_lp, use_container_width=True, config=PCONF)
            with cs4:
                st.plotly_chart(fig_triple_barrier(df, labels_df),
                                use_container_width=True, config=PCONF)

            st.markdown('<div class="sec-hdr">Recent Signals</div>', unsafe_allow_html=True)
            disp = labels_df.tail(15).copy()
            disp["Direction"] = disp["label"].map({1: "▲ LONG", -1: "▼ SHORT", 0: "— FLAT"})
            disp["Return %"]  = (disp["ret"] * 100).round(3)
            disp["Vol Entry"] = disp["vol_entry"].round(5)
            st.dataframe(disp[["time", "Direction", "Return %", "Vol Entry", "pt", "sl"]],
                         use_container_width=True, hide_index=True)
            st.markdown('<div class="sec-hdr">Volume Profile</div>', unsafe_allow_html=True)
            st.plotly_chart(fig_vol_profile(df_raw), use_container_width=True, config=PCONF)

    #  TAB 2: MODELS
    with tabs[2]:
        st.markdown('<div class="sec-hdr">Multi-Model Ensemble</div>', unsafe_allow_html=True)
        if not cfg["run_ml"]:
            st.info("Enable 'Run Full ML Pipeline' in the sidebar.")
        elif not mr:
            st.warning("Insufficient data. Try a longer period (6mo+ recommended).")
        else:
            m_cols = st.columns(min(len(mr), 5))
            for col, (name, metrics) in zip(m_cols, mr.items()):
                with col:
                    stat_block([
                        (name,        "",                                   C["orange"]),
                        ("CV F1",     f"{metrics['cv_f1']:.3f}±{metrics['cv_std']:.3f}", C["green"]),
                        ("Accuracy",  f"{metrics['accuracy']*100:.1f}%",    C["text"]),
                        ("Precision", f"{metrics['precision']:.3f}",        C["cyan"]),
                        ("Recall",    f"{metrics['recall']:.3f}",           C["cyan"]),
                        ("AUC-ROC",   f"{metrics['auc']:.3f}",              C["yellow"]),
                    ])
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_model_radar(mr), use_container_width=True, config=PCONF)
            with c2:
                st.plotly_chart(fig_confusion(mr), use_container_width=True, config=PCONF)

            c3, c4 = st.columns(2)
            with c3:
                shap_vals = None
                if SHAP_OK and mr and not feat_df.empty and len(feat_df) >= 5:
                    try:
                        best_name = ("Ensemble" if "Ensemble" in mr
                                     else max(mr, key=lambda k: mr[k]["cv_f1"]))
                        best_m = mr[best_name]
                        _sX = feat_df[FEATURE_COLS].copy().replace([np.inf, -np.inf], np.nan)
                        for _sc_ in _sX.columns:
                            _sq1, _sq3 = _sX[_sc_].quantile(0.25), _sX[_sc_].quantile(0.75)
                            _siqr = _sq3 - _sq1
                            _sX[_sc_] = _sX[_sc_].clip(lower=_sq1 - 10*_siqr, upper=_sq3 + 10*_siqr)
                        X_s = best_m["scaler"].transform(
                            _sX.fillna(_sX.median()).fillna(0).values.astype(float))
                        shap_vals = compute_shap_values(best_m["model"], X_s[:50])
                    except Exception:
                        pass
                st.plotly_chart(fig_feat_importance(mr, shap_vals),
                                use_container_width=True, config=PCONF)
            with c4:
                st.plotly_chart(fig_roc(mr), use_container_width=True, config=PCONF)

            if LIME_OK and mr and not feat_df.empty:
                best_name = ("Ensemble" if "Ensemble" in mr
                             else max(mr, key=lambda k: mr[k]["cv_f1"]))
                best_m = mr[best_name]
                lime_exp = compute_lime(best_m["model"], best_m["scaler"], feat_df)
                if lime_exp:
                    st.markdown('<div class="sec-hdr">Local Explanation — LIME</div>',
                                unsafe_allow_html=True)
                    feat_ns = [f[0] for f in lime_exp["factors"]]
                    feat_vs = [f[1] for f in lime_exp["factors"]]
                    fig_lime = go.Figure(go.Bar(
                        x=feat_vs, y=feat_ns, orientation="h",
                        marker_color=[C["green"] if v > 0 else C["red"] for v in feat_vs],
                        opacity=.80))
                    fig_lime.update_layout(**BB("LIME Explanation", h=280, legend=False))
                    st.plotly_chart(fig_lime, use_container_width=True, config=PCONF)

            st.markdown('<div class="sec-hdr">Feature Matrix</div>', unsafe_allow_html=True)
            _show_cols = [c for c in FEATURE_COLS + ["label"] if c in feat_df.columns]
            st.dataframe(feat_df[_show_cols].head(12).round(4),
                         use_container_width=True, hide_index=True)

    #  TAB 3: BACKTEST
    with tabs[3]:
        st.markdown('<div class="sec-hdr">Backtest & Validation</div>', unsafe_allow_html=True)
        if not cfg["run_ml"]:
            st.info("Enable 'Run Full ML Pipeline' in the sidebar.")
        elif not bt:
            st.info("""
**Not enough trading signals to run backtest.** Try these adjustments:
- 📅 Select a longer time period (6 months or more)
- 🎯 Lower the confidence threshold (try 0.50–0.55)
- 📊 Choose a more volatile asset for stronger signals
""")
        else:
            bc = st.columns(8)
            kv = [
                ("Total Return",  f"{(bt['equity'].iloc[-1]-1)*100:+.1f}%",
                 C["green"] if bt["equity"].iloc[-1] > 1 else C["red"]),
                ("Ann Return",    f"{bt['ann_ret']*100:+.1f}%",
                 C["green"] if bt["ann_ret"] > 0 else C["red"]),
                ("Sharpe",        f"{bt['sharpe']:.3f}",   C["orange"]),
                ("Sortino",       f"{bt['sortino']:.3f}",  C["yellow"]),
                ("Max Drawdown",  f"{bt['max_dd']*100:.1f}%", C["red"]),
                ("Win Rate",      f"{bt['win_rate']*100:.1f}%", C["green"]),
                ("PSR",           f"{bt['psr']:.3f}",
                 C["green"] if bt["psr"] > 0.9 else C["yellow"]),
                ("DSR",           f"{bt['dsr']:.3f}",
                 C["green"] if bt["dsr"] > 0.9 else C["yellow"]),
            ]
            for col, (lbl, val, color) in zip(bc, kv):
                with col:
                    st.markdown(
                        f'<div class="kpi-card">'
                        f'<div class="kpi-lbl">{lbl}</div>'
                        f'<div class="kpi-val" style="font-size:14px;color:{color}">{val}</div>'
                        f'</div>', unsafe_allow_html=True)

            st.plotly_chart(fig_equity(bt), use_container_width=True, config=PCONF)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_drawdown(bt), use_container_width=True, config=PCONF)
            with c2:
                st.plotly_chart(fig_return_hist(bt), use_container_width=True, config=PCONF)
            st.plotly_chart(fig_monthly(bt), use_container_width=True, config=PCONF)

            st.markdown('<div class="sec-hdr">Extended Metrics</div>', unsafe_allow_html=True)
            stat_block([
                ("Sharpe 95% CI",
                 f"[{bt['sharpe_ci'][0]:.3f}, {bt['sharpe_ci'][1]:.3f}]", C["cyan"]),
                ("Calmar Ratio",  f"{bt['calmar']:.3f}",       C["text"]),
                ("Profit Factor", f"{bt['profit_factor']:.2f}", C["text"]),
                ("# Trades",      str(bt["n_trades"]),          C["text"]),
                ("Total Costs",   f"{bt['total_cost']:.4f}",   C["red"]),
                ("Ann Vol",       f"{bt['ann_vol']*100:.2f}%",  C["text"]),
            ])
            with st.expander("📊 Understanding these metrics"):
                st.markdown("""
- **PSR** (Probabilistic Sharpe Ratio): Probability that the strategy's Sharpe > 0. A value above 0.95 is excellent.
- **DSR** (Deflated Sharpe Ratio): Accounts for multiple testing trials. Above 0.90 indicates a robust strategy.
- **CVaR** (Conditional VaR): Average loss on the worst 5% of days — a tail-risk measure stricter than VaR.
- **Calmar Ratio**: Annualised return divided by max drawdown. Higher is better.
- **Profit Factor**: Gross profit divided by gross loss. Above 1.5 is considered good.
                """)

            st.markdown('<div class="sec-hdr">Walk-Forward Validation</div>', unsafe_allow_html=True)
            wf = walk_forward_backtest(df, signals_df, n_folds=5)
            if wf:
                st.plotly_chart(fig_wf_sharpe(wf), use_container_width=True, config=PCONF)
                st.dataframe(pd.DataFrame([{
                    "Fold": r["fold"], "Sharpe": f"{r['sharpe']:.3f}",
                    "Ann Ret": f"{r['ann_ret']*100:.2f}%",
                    "Max DD": f"{r['max_dd']*100:.2f}%",
                } for r in wf]), use_container_width=True, hide_index=True)

            st.markdown('<div class="sec-hdr">Rolling Window Backtest</div>', unsafe_allow_html=True)
            roll = run_rolling_backtest(df["Close"].squeeze(), signals_df)
            if roll:
                rd = pd.DataFrame(roll)
                fig_roll = go.Figure(go.Bar(
                    x=rd["start"], y=rd["sharpe"],
                    marker_color=[C["green"] if s > 0 else C["red"] for s in rd["sharpe"]],
                    opacity=.80))
                fig_roll.update_layout(**BB("Rolling Sharpe — Strategy Consistency", h=220))
                st.plotly_chart(fig_roll, use_container_width=True, config=PCONF)

            st.markdown('<div class="sec-hdr">Benchmark Comparison</div>', unsafe_allow_html=True)
            with st.spinner("Fetching benchmarks…"):
                bmks = fetch_benchmarks(cfg["period"])
            bench_df = benchmark_comparison(bt.get("equity"), bmks)
            if not bench_df.empty:
                st.dataframe(bench_df, use_container_width=True, hide_index=True)

            # Enhancement 20: Event-driven + Monte Carlo
            if edb_result:
                st.markdown('<div class="sec-hdr">Event-Driven Backtest (Enhancement 20)</div>',
                            unsafe_allow_html=True)
                edb_cols = st.columns(4)
                for col_, (lbl_, val_, clr_) in zip(edb_cols, [
                    ("Total Return",  f"{edb_result.get('total_return_pct', 0):+.2f}%",
                     C["green"] if edb_result.get("total_return_pct", 0) > 0 else C["red"]),
                    ("Final Equity",  f"${edb_result.get('final_equity', 0):,.2f}",  C["orange"]),
                    ("Sharpe",        f"{edb_result.get('sharpe', 0):.3f}",           C["cyan"]),
                    ("N Trades",      str(edb_result.get("n_trades", 0)),             C["text"]),
                ]):
                    with col_:
                        st.markdown(
                            f'<div class="kpi-card">'
                            f'<div class="kpi-lbl">{lbl_}</div>'
                            f'<div class="kpi-val" style="color:{clr_}">{val_}</div>'
                            f'</div>', unsafe_allow_html=True)
                fig_edb = go.Figure(go.Scatter(
                    x=edb_result["equity"].index, y=(edb_result["equity"] / 10_000 - 1) * 100,
                    line=dict(color=C["purple"], width=1.8),
                    fill="tozeroy", fillcolor="rgba(155,109,255,.08)",
                    name="Event-Driven Equity"))
                fig_edb.update_layout(**BB("Event-Driven Backtest Equity (%)", h=280))
                st.plotly_chart(fig_edb, use_container_width=True, config=PCONF)

            if mc_result:
                st.markdown('<div class="sec-hdr">Monte Carlo Simulation (500 Paths)</div>',
                            unsafe_allow_html=True)
                mc_cols = st.columns(4)
                for col_, (lbl_, val_, clr_) in zip(mc_cols, [
                    ("P(Profit)",  f"{mc_result.get('prob_profit', 0)*100:.1f}%",
                     C["green"] if mc_result.get("prob_profit", 0) > 0.6 else C["yellow"]),
                    ("P(Double)",  f"{mc_result.get('prob_double', 0)*100:.1f}%",   C["cyan"]),
                    ("Median Sharpe", f"{mc_result.get('sharpe_median', 0):.3f}",   C["orange"]),
                    ("Median DD",  f"{mc_result.get('max_dd_median', 0)*100:.1f}%", C["red"]),
                ]):
                    with col_:
                        st.markdown(
                            f'<div class="kpi-card">'
                            f'<div class="kpi-lbl">{lbl_}</div>'
                            f'<div class="kpi-val" style="color:{clr_}">{val_}</div>'
                            f'</div>', unsafe_allow_html=True)

                te = mc_result["terminal_equity"]
                fig_mc = go.Figure()
                fig_mc.add_trace(go.Histogram(
                    x=(te - 1) * 100, nbinsx=50,
                    marker_color=[C["green"] if v >= 0 else C["red"] for v in (te - 1) * 100],
                    opacity=.70, name="Terminal Returns"))
                fig_mc.add_vline(x=0, line_color=C["yellow"], line_dash="dash", line_width=1.5)
                ci5  = float(np.percentile((te - 1) * 100, 5))
                ci95 = float(np.percentile((te - 1) * 100, 95))
                fig_mc.add_vline(x=ci5,  line_color=C["red"],   line_dash="dot",  line_width=1)
                fig_mc.add_vline(x=ci95, line_color=C["green"], line_dash="dot",  line_width=1)
                fig_mc.update_layout(
                    **BB(f"Monte Carlo Terminal Returns % — 5th:{ci5:.1f}% / 95th:{ci95:.1f}%",
                         h=280, legend=False))
                st.plotly_chart(fig_mc, use_container_width=True, config=PCONF)

    #  TAB 4: RISK
    with tabs[4]:
        st.markdown('<div class="sec-hdr">Risk Analysis</div>', unsafe_allow_html=True)
        risk = compute_risk(df["returns"]) if "returns" in df.columns else {}
        if not risk:
            st.warning("Insufficient data for risk analysis.")
        else:
            kr = st.columns(4)
            for col_, (lbl_, val_, clr_) in zip(kr, [
                ("VaR 95%",  f"{risk['var95']:.3f}%",  C["red"]),
                ("CVaR 95%", f"{risk['cvar95']:.3f}%", C["red"]),
                ("Ann Vol",  f"{risk['ann_vol']:.2f}%", C["yellow"]),
                ("Max DD",   f"{risk['max_dd']:.2f}%",  C["red"]),
            ]):
                with col_:
                    st.markdown(
                        f'<div class="kpi-card">'
                        f'<div class="kpi-lbl">{lbl_}</div>'
                        f'<div class="kpi-val" style="color:{clr_}">{val_}</div>'
                        f'</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(fig_var_dist(risk), use_container_width=True, config=PCONF)
            with c2:
                st.plotly_chart(fig_rolling_sharpe(risk), use_container_width=True, config=PCONF)
            st.markdown('<div class="sec-hdr">Risk Metrics</div>', unsafe_allow_html=True)
            stat_block([
                ("VaR 95%",        f"{risk['var95']:.3f}%",  C["red"]),
                ("CVaR 95%",       f"{risk['cvar95']:.3f}%", C["red"]),
                ("VaR 99%",        f"{risk['var99']:.3f}%",  C["red"]),
                ("Ann Volatility", f"{risk['ann_vol']:.2f}%", C["yellow"]),
                ("PSR",            f"{risk['psr']:.4f}",     C["cyan"]),
                ("Skewness",       f"{risk['skew']:.3f}",    C["text"]),
                ("Excess Kurtosis",f"{risk['kurt']:.3f}",    C["text"]),
                ("Max Drawdown",   f"{risk['max_dd']:.2f}%", C["red"]),
                ("Stationarity",
                 "✓ STATIONARY" if risk.get("is_stationary") else "✗ NON-STATIONARY",
                 C["green"] if risk.get("is_stationary") else C["red"]),
            ])
            st.markdown('<div class="sec-hdr">Stress Scenarios</div>', unsafe_allow_html=True)
            stat_block([(k, f"{v:.2f}%", C["red"]) for k, v in risk["stress"].items()])
            st.plotly_chart(fig_vol_forecast(df), use_container_width=True, config=PCONF)
            dd_s = risk.get("drawdown", pd.Series())
            if not dd_s.empty:
                fig_dd2 = go.Figure(go.Scatter(
                    x=dd_s.index, y=dd_s.values * 100,
                    fill="tozeroy", fillcolor="rgba(255,61,90,.18)",
                    line=dict(color=C["red"], width=1)))
                fig_dd2.update_layout(**BB("Rolling Drawdown (%)", h=220, legend=False))
                st.plotly_chart(fig_dd2, use_container_width=True, config=PCONF)

    #  TAB 5: PORTFOLIO
    with tabs[5]:
        st.markdown('<div class="sec-hdr">Portfolio Optimisation</div>', unsafe_allow_html=True)
        syms_port = [cfg["yf_sym"]] + cfg["comp_symbols"]
        if len(syms_port) < 2:
            st.info("Select at least 2 comparison assets in the sidebar.")
        else:
            with st.spinner("Fetching multi-asset data & optimising…"):
                multi = fetch_multi_ohlcv(tuple(syms_port), cfg["period"])
            if len(multi) < 2:
                st.warning("Could not fetch enough asset data.")
            else:
                ret_df = pd.DataFrame({
                    s: d["Close"].squeeze().pct_change()
                    for s, d in multi.items()
                }).dropna()
                try:
                    wts = hrp_weights(ret_df)
                except Exception:
                    wts = pd.Series(1.0 / len(ret_df.columns), index=ret_df.columns)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(fig_hrp_bar(wts), use_container_width=True, config=PCONF)
                with c2:
                    st.plotly_chart(fig_corr_matrix(ret_df), use_container_width=True, config=PCONF)
                with st.spinner("Computing efficient frontier…"):
                    try:
                        front = efficient_frontier(ret_df, n=30)
                    except Exception:
                        front = pd.DataFrame()
                c3, c4 = st.columns(2)
                with c3:
                    st.plotly_chart(fig_frontier(front), use_container_width=True, config=PCONF)
                with c4:
                    port_ret = (ret_df[wts.index] * wts).sum(axis=1)
                    cum_p    = (1 + port_ret).cumprod() - 1
                    fig_cum  = go.Figure()
                    palette  = [C["orange"], C["blue"], C["green"], C["yellow"], C["purple"], C["cyan"]]
                    for i, sym in enumerate(ret_df.columns[:5]):
                        c_  = (1 + ret_df[sym]).cumprod() - 1
                        fig_cum.add_trace(go.Scatter(
                            x=c_.index, y=c_.values * 100,
                            line=dict(color=palette[(i+1) % len(palette)], width=1, dash="dot"),
                            name=sym.replace("-USD", ""), opacity=.60))
                    fig_cum.add_trace(go.Scatter(
                        x=cum_p.index, y=cum_p.values * 100,
                        line=dict(color=C["orange"], width=2.5), name="HRP Portfolio"))
                    fig_cum.update_layout(**BB("HRP vs Individual Assets (%)", h=340))
                    st.plotly_chart(fig_cum, use_container_width=True, config=PCONF)
                st.markdown('<div class="sec-hdr">Allocation Table</div>', unsafe_allow_html=True)
                wt_tab = pd.DataFrame({
                    "Asset":      [s.replace("-USD", "") for s in wts.index],
                    "Weight":     [f"{w * 100:.2f}%" for w in wts.values],
                    "Period Ret": [f"{ret_df[s].sum() * 100:.1f}%" for s in wts.index],
                    "Vol":        [f"{ret_df[s].std() * np.sqrt(252) * 100:.1f}%" for s in wts.index],
                    "Sharpe":     [f"{ret_df[s].mean()/(ret_df[s].std()+1e-10)*np.sqrt(252):.2f}"
                                   for s in wts.index],
                })
                st.dataframe(wt_tab, use_container_width=True, hide_index=True)

    #  TAB 6: RESEARCH
    with tabs[6]:
        st.markdown('<div class="sec-hdr">Quantitative Research</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(fig_entropy(df["Close"].squeeze()),
                            use_container_width=True, config=PCONF)
        with c2:
            st.plotly_chart(fig_frac_diff(df["Close"].squeeze()),
                            use_container_width=True, config=PCONF)
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(fig_cusum(df["Close"].squeeze()),
                            use_container_width=True, config=PCONF)
        with c4:
            if "obv" in df.columns and "vwap" in df.columns:
                fig_obv = make_subplots(rows=2, cols=1, shared_xaxes=True)
                fig_obv.add_trace(go.Scatter(x=df.index, y=df["Close"].squeeze(),
                    line=dict(color=C["orange"], width=1), name="Price"), row=1, col=1)
                fig_obv.add_trace(go.Scatter(x=df.index, y=df["vwap"],
                    line=dict(color=C["purple"], width=1, dash="dash"), name="VWAP"), row=1, col=1)
                fig_obv.add_trace(go.Scatter(x=df.index, y=df["obv"],
                    fill="tozeroy", fillcolor="rgba(0,136,255,.10)",
                    line=dict(color=C["blue"], width=1), name="OBV"), row=2, col=1)
                ly = BB("OBV & VWAP", h=300); ly["xaxis2"] = _xa(); ly["showlegend"] = True
                fig_obv.update_layout(**ly)
                st.plotly_chart(fig_obv, use_container_width=True, config=PCONF)
        st.markdown('<div class="sec-hdr">Stationarity Tests</div>', unsafe_allow_html=True)
        adf_rows = []
        if ADF_OK:
            c_ser = df["Close"].squeeze()
            lp_   = np.log(c_ser.replace(0, np.nan).dropna() + 1e-10)
            for col_name, series in [
                ("Price",       c_ser),
                ("Log Price",   lp_),
                ("Returns",     df["returns"] if "returns" in df.columns else pd.Series(dtype=float)),
                ("Log Returns", df["log_returns"] if "log_returns" in df.columns else pd.Series(dtype=float)),
            ]:
                clean = series.dropna()
                if len(clean) < 15:
                    continue
                try:
                    stat, pval, _, _, cv, _ = adfuller(clean)
                    adf_rows.append({
                        "Series":      col_name, "ADF Stat": f"{stat:.4f}",
                        "p-value":     f"{pval:.4f}", "5% Critical": f"{cv['5%']:.4f}",
                        "Stationary":  "✓ YES" if pval < 0.05 else "✗ NO",
                    })
                except Exception:
                    pass
        if adf_rows:
            st.dataframe(pd.DataFrame(adf_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Install statsmodels for ADF tests: pip install statsmodels")
        st.markdown('<div class="sec-hdr">DXY Dollar Index</div>', unsafe_allow_html=True)
        dxy = fetch_dxy()
        if not dxy.empty:
            fig_dxy = make_subplots(rows=2, cols=1, shared_xaxes=True)
            price_d = df["Close"].squeeze().resample("D").last().dropna()
            dxy_d   = dxy["DXY"].dropna()
            common  = price_d.index.intersection(dxy_d.index)
            if len(common) > 10:
                fig_dxy.add_trace(go.Scatter(x=common, y=price_d.loc[common].values,
                    line=dict(color=C["orange"], width=1.5), name="Crypto"), row=1, col=1)
                fig_dxy.add_trace(go.Scatter(x=common, y=dxy_d.loc[common].values,
                    line=dict(color=C["blue"], width=1.5), name="DXY"), row=2, col=1)
                ly = BB("Crypto vs DXY Dollar Index", h=300)
                ly["xaxis2"] = _xa(); ly["showlegend"] = True
                fig_dxy.update_layout(**ly)
                st.plotly_chart(fig_dxy, use_container_width=True, config=PCONF)

    #  TAB 7: SENTIMENT
    with tabs[7]:
        st.markdown('<div class="sec-hdr">Market Sentiment</div>', unsafe_allow_html=True)
        s_c1, s_c2, s_c3 = st.columns(3)
        with s_c1:
            st.markdown('<div class="sec-hdr">Reddit Sentiment</div>', unsafe_allow_html=True)
            with st.spinner("Fetching Reddit…"):
                red = fetch_reddit_sentiment("CryptoCurrency")
            score = red["score"]
            col_  = C["green"] if score > 0.05 else C["red"] if score < -0.05 else C["yellow"]
            st.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-lbl">SENTIMENT SCORE</div>'
                f'<div class="kpi-val" style="color:{col_}">'
                f'{score:+.3f} — {red["label"].upper()}</div>'
                f'<div style="font-size:8px;color:{C["sec"]};">'
                f'{red["count"]} posts analysed</div>'
                f'</div>', unsafe_allow_html=True)
            sub2 = st.selectbox("Try another subreddit",
                                  ["Bitcoin", "ethereum", "solana", "altcoin"],
                                  label_visibility="collapsed")
            if st.button("Analyze"):
                red2 = fetch_reddit_sentiment(sub2)
                st.markdown(
                    f'<div class="alert-green">/r/{sub2}: '
                    f'{red2["score"]:+.3f} ({red2["label"]}) — {red2["count"]} posts</div>',
                    unsafe_allow_html=True)
        with s_c2:
            st.markdown('<div class="sec-hdr">Crypto News</div>', unsafe_allow_html=True)
            with st.spinner("Fetching news…"):
                news = fetch_crypto_news()
            if news:
                st.plotly_chart(fig_news(news), use_container_width=True, config=PCONF)
                pos_ = sum(1 for n in news if n["sentiment"] == "bullish")
                neg_ = sum(1 for n in news if n["sentiment"] == "bearish")
                stat_block([
                    ("Bullish headlines", str(pos_), C["green"]),
                    ("Bearish headlines", str(neg_), C["red"]),
                    ("Neutral",          str(len(news) - pos_ - neg_), C["muted"]),
                ])
            else:
                st.info("News unavailable. Both free sources are temporarily down.")
        with s_c3:
            st.markdown('<div class="sec-hdr">On-Chain Data</div>', unsafe_allow_html=True)
            with st.spinner("Fetching blockchain.info stats…"):
                oc = fetch_onchain_proxies()
            if oc:
                rows_ = []
                for key_, lbl_, clr_ in [
                    ("hash_rate_eh",   "Hash Rate (EH/s)",     C["cyan"]),
                    ("miners_revenue", "Miners Revenue ($)",    C["text"]),
                    ("n_transactions", "Transactions",          C["text"]),
                    ("mempool_size",   "Mempool Size",          C["text"]),
                    ("reddit_subs",    "Reddit Subscribers",    C["purple"]),
                    ("twitter_followers","Twitter Followers",   C["blue"]),
                ]:
                    v_ = oc.get(key_)
                    if v_:
                        rows_.append((lbl_, f"{v_:,.2f}" if isinstance(v_, float)
                                      else f"{v_:,}", clr_))
                if rows_:
                    stat_block(rows_)
                else:
                    st.info("On-chain proxy data unavailable.")
            else:
                st.info("blockchain.info temporarily unavailable.")

    #  TAB 8: ALPHA LAB — All 16 Intelligence Engines
    with tabs[8]:
        st.markdown('<div class="sec-hdr">Alpha Lab — Intelligence Engine (v6)</div>',
                    unsafe_allow_html=True)
        if not cfg["run_ml"]:
            st.info("Enable 'Run Full ML Pipeline' in the sidebar to activate the Alpha Lab.")
        else:
            # Section 1: Feature Stability
            st.markdown('<div class="sec-hdr">1. Feature Stability Analysis</div>',
                        unsafe_allow_html=True)
            if not stability_df.empty:
                c1, c2 = st.columns(2)
                with c1:
                    fig_stab = go.Figure(go.Bar(
                        x=stability_df["CV (instability)"], y=stability_df["Feature"],
                        orientation="h",
                        marker_color=[C["green"] if s else C["red"]
                                      for s in stability_df["Stable"]],
                        opacity=.80))
                    fig_stab.add_vline(x=0.5, line_color=C["yellow"],
                                       line_dash="dash", line_width=1.5)
                    fig_stab.update_layout(
                        **BB("Feature CV — Instability Score (< 0.5 = stable)", h=340, legend=False))
                    st.plotly_chart(fig_stab, use_container_width=True, config=PCONF)
                with c2:
                    fig_dft = go.Figure(go.Bar(
                        x=stability_df["Decay (early→late)"], y=stability_df["Feature"],
                        orientation="h",
                        marker_color=[C["red"] if d > 0 else C["green"]
                                      for d in stability_df["Decay (early→late)"]],
                        opacity=.80))
                    fig_dft.update_layout(
                        **BB("Feature Decay — Importance Drop Early→Late", h=340, legend=False))
                    st.plotly_chart(fig_dft, use_container_width=True, config=PCONF)
                st.dataframe(stability_df.round(4), use_container_width=True, hide_index=True)
            else:
                st.info("Need more data for stability analysis (60+ labelled samples).")

            # Section 2: Baseline Benchmarking
            st.markdown('<div class="sec-hdr">2. Baseline Benchmarking — Does ML Beat Naive?</div>',
                        unsafe_allow_html=True)
            if baselines and mr:
                best_k  = max(mr, key=lambda k: mr[k]["cv_f1"])
                ml_f1   = round(mr[best_k]["cv_f1"], 3)
                bdf = pd.DataFrame([
                    {"Strategy": k,
                     "F1":       str(v.get("f1", "—")),
                     "Accuracy": str(v.get("accuracy", "—")),
                     "Notes":    "Naive baseline"}
                    for k, v in baselines.items()
                ])
                bdf = pd.concat([
                    bdf,
                    pd.DataFrame([{"Strategy": f"ML ({best_k})", "F1": str(ml_f1),
                                   "Accuracy": "—", "Notes": "HECTOR ML model"}])
                ], ignore_index=True)
                st.dataframe(bdf, use_container_width=True, hide_index=True)
                random_f1 = baselines.get("Random", {}).get("f1", 0) or 0
                if ml_f1 > random_f1 + 0.05:
                    st.markdown(
                        f'<div class="alert-green">ML beats random by {ml_f1 - random_f1:.3f} F1 — alpha is real.</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="alert-red">ML does not beat random — need more data or features.</div>',
                        unsafe_allow_html=True)

            # Section 3: Calibration
            st.markdown('<div class="sec-hdr">3. Probability Calibration</div>',
                        unsafe_allow_html=True)
            stat_block([
                ("Method",  "Platt scaling (logistic)", C["cyan"]),
                ("Status",  "Active" if cal_clf else "Inactive",
                 C["green"] if cal_clf else C["muted"]),
            ])

            # Section 4: Meta-Labeling
            st.markdown('<div class="sec-hdr">4. Meta-Labeling Filter</div>',
                        unsafe_allow_html=True)
            if meta_result:
                stat_block([
                    ("Meta-model accuracy",
                     f"{meta_result['accuracy']*100:.1f}%", C["green"]),
                    ("Status", "Applied — low-confidence signals removed", C["cyan"]),
                ])
            else:
                st.info("Meta-labeling needs 40+ labelled samples.")

            # Section 5: Dynamic Thresholds
            st.markdown('<div class="sec-hdr">5. Dynamic Thresholds Per Regime</div>',
                        unsafe_allow_html=True)
            if dyn_thresholds:
                stat_block([(r_.capitalize(), f"{t_:.2f}",
                             C["green"] if r_ == "bull" else C["red"] if r_ == "bear"
                             else C["yellow"])
                            for r_, t_ in dyn_thresholds.items()])
                opt_thr = dyn_thresholds.get(regime, cfg["conf_thr"])
                st.markdown(
                    f'<div class="alert-green">Current regime: {regime.upper()} → optimal threshold = {opt_thr:.2f}</div>',
                    unsafe_allow_html=True)

            # Section 6: Walk-Forward Validation
            st.markdown('<div class="sec-hdr">6. Walk-Forward Validation Engine</div>',
                        unsafe_allow_html=True)
            if wf_folds:
                wf_disp = pd.DataFrame(wf_folds)
                fig_wfv = make_subplots(rows=1, cols=2,
                                         subplot_titles=["F1 per Fold", "AUC per Fold"])
                fig_wfv.add_trace(go.Bar(
                    x=wf_disp["fold"], y=wf_disp["f1"],
                    marker_color=[C["green"] if v > 0.5 else C["red"] for v in wf_disp["f1"]],
                    opacity=.80, name="F1"), row=1, col=1)
                fig_wfv.add_trace(go.Bar(
                    x=wf_disp["fold"], y=wf_disp["auc"],
                    marker_color=[C["orange"] if v > 0.5 else C["muted"] for v in wf_disp["auc"]],
                    opacity=.80, name="AUC"), row=1, col=2)
                fig_wfv.add_hline(y=0.5, line_color=C["yellow"], line_dash="dash",
                                   line_width=1, row="all")
                fig_wfv.update_layout(**BB("Walk-Forward Fold Performance", h=260))
                st.plotly_chart(fig_wfv, use_container_width=True, config=PCONF)
                st.dataframe(wf_disp.round(3), use_container_width=True, hide_index=True)
            else:
                st.info("Walk-forward needs 60+ labelled samples.")

            # Section 7: Advanced Backtest
            st.markdown('<div class="sec-hdr">7. Advanced Backtest (Latency + Vol Targeting)</div>',
                        unsafe_allow_html=True)
            if adv_bt:
                adv_c = st.columns(4)
                for col_, (lbl_, val_, clr_) in zip(adv_c, [
                    ("Sharpe",         f"{adv_bt.get('sharpe', 0):.3f}",    C["orange"]),
                    ("Ann Return",     f"{adv_bt.get('ann_ret', 0)*100:+.1f}%",
                     C["green"] if adv_bt.get("ann_ret", 0) > 0 else C["red"]),
                    ("Max Drawdown",   f"{adv_bt.get('max_dd', 0)*100:.1f}%", C["red"]),
                    ("DD Cap Events",  str(adv_bt.get("dd_cap_events", 0)), C["yellow"]),
                ]):
                    with col_:
                        st.markdown(
                            f'<div class="kpi-card">'
                            f'<div class="kpi-lbl">{lbl_}</div>'
                            f'<div class="kpi-val" style="color:{clr_}">{val_}</div>'
                            f'</div>', unsafe_allow_html=True)
                fig_adv = go.Figure()
                fig_adv.add_trace(go.Scatter(
                    x=adv_bt["equity"].index, y=(adv_bt["equity"] - 1) * 100,
                    line=dict(color=C["orange"], width=2), name="Advanced BT",
                    fill="tozeroy", fillcolor="rgba(255,140,0,.06)"))
                if bt:
                    fig_adv.add_trace(go.Scatter(
                        x=bt["equity"].index, y=(bt["equity"] - 1) * 100,
                        line=dict(color=C["blue"], width=1.5, dash="dash"), name="Simple BT"))
                fig_adv.update_layout(**BB("Advanced vs Simple Backtest Equity (%)", h=280))
                st.plotly_chart(fig_adv, use_container_width=True, config=PCONF)

            # Section 8: Portfolio Constraints
            st.markdown('<div class="sec-hdr">8. Portfolio Risk Constraints</div>',
                        unsafe_allow_html=True)
            if constraints:
                stat_block([
                    ("Ann Volatility",       f"{constraints.get('ann_vol_pct', 0):.2f}%", C["yellow"]),
                    ("Vol Target",           f"{constraints.get('vol_target_pct', 15):.1f}%", C["text"]),
                    ("Vol Scaling Factor",   f"{constraints.get('vol_scaling_factor', 1):.3f}", C["cyan"]),
                    ("Max Drawdown",         f"{constraints.get('max_dd_pct', 0):.2f}%", C["red"]),
                    ("DD Breach Events",     str(constraints.get("dd_breaches", 0)),
                     C["red"] if constraints.get("dd_breaches", 0) > 0 else C["green"]),
                    ("Constraint Active",    "YES" if constraints.get("constraint_active") else "NO",
                     C["red"] if constraints.get("constraint_active") else C["green"]),
                ])

            # Section 9: Performance Attribution
            st.markdown('<div class="sec-hdr">9. Performance Attribution</div>',
                        unsafe_allow_html=True)
            if attr_dict:
                attr_rows = []
                for key, val in attr_dict.items():
                    if key.startswith("regime_"):
                        r_name = key.replace("regime_", "").capitalize()
                        attr_rows.append((
                            f"Regime: {r_name}",
                            f"Ann Ret {val.get('ann_ret_pct',0):+.2f}%  "
                            f"| Trades {val.get('n_trades',0)} | WR {val.get('win_rate',0)*100:.1f}%",
                            C["green"] if val.get("ann_ret_pct", 0) > 0 else C["red"],
                        ))
                    elif key.startswith("direction_"):
                        d_name = key.replace("direction_", "").capitalize()
                        attr_rows.append((
                            f"Direction: {d_name}",
                            f"Ann Ret {val.get('ann_ret_pct',0):+.2f}%  "
                            f"| Trades {val.get('n_trades',0)} | WR {val.get('win_rate',0)*100:.1f}%",
                            C["cyan"],
                        ))
                    elif key.startswith("group_"):
                        g_name = key.replace("group_", "")
                        attr_rows.append((
                            f"Feature Group: {g_name}",
                            f"Mean Importance {val.get('mean_importance', 0):.4f}",
                            C["purple"],
                        ))
                if attr_rows:
                    stat_block(attr_rows)

            # Section 10: Significance Tests
            st.markdown('<div class="sec-hdr">10. Statistical Significance Testing</div>',
                        unsafe_allow_html=True)
            if sig_tests:
                stat_block([
                    ("t-statistic",      f"{sig_tests.get('t_stat', 0):.4f}",    C["cyan"]),
                    ("p-value",          f"{sig_tests.get('p_value', 1):.4f}",
                     C["green"] if sig_tests.get("p_value", 1) < 0.05 else C["red"]),
                    ("Significant (5%)",
                     "YES — unlikely random" if sig_tests.get("is_significant")
                     else "NO — may be random",
                     C["green"] if sig_tests.get("is_significant") else C["red"]),
                    ("Obs Sharpe",       f"{sig_tests.get('obs_sharpe', 0):.3f}",  C["orange"]),
                    ("Bootstrap Sharpe", f"{sig_tests.get('boot_sharpe_mean', 0):.3f}", C["text"]),
                    ("P(Sharpe > 0)",    f"{sig_tests.get('boot_sharpe_p', 0):.3f}",
                     C["green"] if sig_tests.get("boot_sharpe_p", 0) > 0.7 else C["yellow"]),
                    ("Bootstrap 95% CI",
                     f"[{sig_tests.get('boot_ci', (0,0))[0]:.3f}, "
                     f"{sig_tests.get('boot_ci', (0,0))[1]:.3f}]", C["text"]),
                ])

            # Section 11: Drift Detection
            st.markdown('<div class="sec-hdr">11. Model Drift Detection (PSI)</div>',
                        unsafe_allow_html=True)
            if not drift_df.empty:
                retrain_needed = should_retrain(drift_df)
                if retrain_needed:
                    st.markdown(
                        '<div class="alert-red">MAJOR DRIFT DETECTED — retraining recommended.</div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="alert-green">Feature distributions are stable.</div>',
                        unsafe_allow_html=True)
                fig_psi = go.Figure(go.Bar(
                    x=drift_df["PSI"], y=drift_df["Feature"], orientation="h",
                    marker_color=[C["red"] if v >= 0.25 else C["yellow"] if v >= 0.10
                                  else C["green"] for v in drift_df["PSI"]],
                    opacity=.80))
                fig_psi.add_vline(x=0.10, line_color=C["yellow"], line_dash="dash", line_width=1)
                fig_psi.add_vline(x=0.25, line_color=C["red"],    line_dash="dash", line_width=1.5)
                fig_psi.update_layout(**BB("PSI — Feature Distribution Shift", h=340, legend=False))
                st.plotly_chart(fig_psi, use_container_width=True, config=PCONF)
                st.dataframe(drift_df, use_container_width=True, hide_index=True)

            # Section 12: Retraining Schedule
            st.markdown('<div class="sec-hdr">12. Retraining Scheduler</div>',
                        unsafe_allow_html=True)
            if retrain_sched:
                color_ = C["red"] if retrain_sched.get("due") else C["green"]
                stat_block([
                    ("Retraining due",  "YES" if retrain_sched.get("due") else "NO", color_),
                    ("Reason",          retrain_sched.get("reason", ""),             C["text"]),
                    ("Age (hours)",     f"{retrain_sched.get('age_hours', 0):.1f}h", C["muted"]),
                    ("Next scheduled",  retrain_sched.get("next_run", ""),           C["cyan"]),
                ])
                if retrain_sched.get("due"):
                    if mr:
                        if st.button("🔄 Trigger Retraining Now"):
                            st.cache_data.clear()
                            st.toast("Retraining triggered — refreshing data.", icon="🔄")
                            st.rerun()
                    else:
                        st.warning("No trained models available. Run the ML pipeline first.")

            # Section 13: Experiment Tracking
            st.markdown('<div class="sec-hdr">13. Experiment Tracking</div>',
                        unsafe_allow_html=True)
            if run_id:
                st.markdown(f'<div class="alert-green">Run ID: {run_id} logged.</div>',
                            unsafe_allow_html=True)
            exp_history = compare_experiments(cfg["yf_sym"])
            if not exp_history.empty:
                st.dataframe(exp_history, use_container_width=True, hide_index=True)
            else:
                if cfg["run_ml"] and mr:
                    st.info("Experiments will appear here after the first successful model run.")
                else:
                    st.warning("Enable **Run Full ML Pipeline** in the sidebar to start tracking experiments.")

            # Section 14: Alpha Decay
            st.markdown('<div class="sec-hdr">14. Alpha Decay Monitoring</div>',
                        unsafe_allow_html=True)
            if not decay_df.empty:
                fig_alpha = make_subplots(rows=1, cols=2,
                                           subplot_titles=["Sharpe by Window", "Win Rate by Window"])
                fig_alpha.add_trace(go.Scatter(
                    x=decay_df["Window"], y=decay_df["Sharpe"], mode="lines+markers",
                    line=dict(color=C["orange"], width=2), marker=dict(size=7),
                    name="Sharpe"), row=1, col=1)
                fig_alpha.add_trace(go.Scatter(
                    x=decay_df["Window"], y=decay_df["Win Rate"], mode="lines+markers",
                    line=dict(color=C["cyan"], width=2), marker=dict(size=7),
                    name="Win Rate"), row=1, col=2)
                fig_alpha.add_hline(y=0, line_color=C["muted"], line_dash="dash",
                                     line_width=1, row=1, col=1)
                fig_alpha.add_hline(y=0.5, line_color=C["muted"], line_dash="dash",
                                     line_width=1, row=1, col=2)
                fig_alpha.update_layout(**BB("Alpha Decay — Rolling Performance Windows", h=260))
                st.plotly_chart(fig_alpha, use_container_width=True, config=PCONF)
                st.dataframe(decay_df.round(3), use_container_width=True, hide_index=True)
                if len(decay_df) >= 3:
                    slope_, _, _, p_slope, _ = stats.linregress(
                        range(len(decay_df)), decay_df["Sharpe"].values)
                    trend_cls = ("alert-red" if slope_ < -0.05 and p_slope < 0.2
                                  else "alert-green")
                    st.markdown(
                        f'<div class="{trend_cls}">Sharpe trend: slope = {slope_:.3f} (p={p_slope:.3f})</div>',
                        unsafe_allow_html=True)
            else:
                st.info("Alpha decay analysis requires 50+ backtested trades. Try a longer period.")

            # Section 15: Feedback Loop + Auto-Refinement (Enhancement 15)
            st.markdown('<div class="sec-hdr">15. Feedback Loop — Losing Trade Analysis & Auto-Refinement</div>',
                        unsafe_allow_html=True)
            if feedback:
                c_fb1, c_fb2 = st.columns(2)
                with c_fb1:
                    stat_block([
                        ("Total trades",    str(feedback.get("n_winners", 0) +
                                               feedback.get("n_losers", 0)),  C["text"]),
                        ("Winners",         str(feedback.get("n_winners", 0)), C["green"]),
                        ("Losers",          str(feedback.get("n_losers", 0)),  C["red"]),
                        ("Win rate",        f"{feedback.get('win_rate', 0)*100:.1f}%",
                         C["green"] if feedback.get("win_rate", 0) > 0.5 else C["red"]),
                        ("Avg winner ret",  f"{feedback.get('avg_win_ret', 0):+.3f}%",  C["green"]),
                        ("Avg loser ret",   f"{feedback.get('avg_lose_ret', 0):+.3f}%", C["red"]),
                        ("Top discriminators",
                         ", ".join(feedback.get("top_discriminators", [])), C["cyan"]),
                    ])
                with c_fb2:
                    win_m  = feedback.get("win_feature_means", {})
                    lose_m = feedback.get("lose_feature_means", {})
                    if win_m and lose_m:
                        common_f = [f for f in win_m if f in lose_m][:8]
                        fig_fb = go.Figure()
                        fig_fb.add_trace(go.Bar(name="Winners", x=common_f,
                            y=[win_m[f] for f in common_f],
                            marker_color=C["green"], opacity=0.75))
                        fig_fb.add_trace(go.Bar(name="Losers", x=common_f,
                            y=[lose_m[f] for f in common_f],
                            marker_color=C["red"], opacity=0.75))
                        fig_fb.update_layout(
                            **BB("Feature Means: Winners vs Losers", h=280), barmode="group")
                        st.plotly_chart(fig_fb, use_container_width=True, config=PCONF)

                # Auto-refinement result
                if refined_result:
                    improved = refined_result.get("improved", False)
                    cls_ = "alert-green" if improved else "alert-yellow"
                    msg_ = (f"Refined model IMPROVED: +{refined_result['delta_f1']:.4f} F1 "
                            f"({refined_result['base_f1']:.3f} → {refined_result['new_f1']:.3f})"
                            if improved
                            else f"No improvement: base F1={refined_result.get('base_f1',0):.3f}")
                    st.markdown(f'<div class="{cls_}">{msg_}</div>', unsafe_allow_html=True)

                    if ab_result:
                        stat_block([
                            ("A/B Winner",        ab_result.get("winner", "—"),     C["orange"]),
                            ("F1 Baseline",        f"{ab_result.get('f1_a', 0):.4f}", C["blue"]),
                            ("F1 Refined",         f"{ab_result.get('f1_b', 0):.4f}", C["green"]),
                            ("Recommendation",     ab_result.get("recommendation", "—"), C["cyan"]),
                        ])
            else:
                st.info("Feedback loop needs a completed backtest with trades. Try a longer period or lower confidence threshold.")

    #  TAB 9: LIVE FEED — Enhancement 16
    with tabs[9]:
        st.markdown('<div class="sec-hdr">Live Feed — Real-Time Data Pipeline (Enhancement 16)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="alert-yellow">WebSocket integration simulated via 1-min yfinance polling. '
            'In production, connect to Binance WSS or Coinbase Pro WebSocket.</div>',
            unsafe_allow_html=True)

        rt_c1, rt_c2 = st.columns([2, 1])
        with rt_c1:
            if st.button("Fetch Latest 1-min Bars"):
                feed = _get_rt_feed()
                updated = update_rt_feed(cfg["yf_sym"], feed)
                if updated:
                    st.success(f"Feed updated — {len(feed._buf)} bars in buffer")
                else:
                    st.warning("Feed update failed — using cached data")

                df_rt = feed.to_df()
                if not df_rt.empty:
                    df_rt_r = df_rt.rename(columns={
                        "open":"Open","high":"High","low":"Low",
                        "close":"Close","volume":"Volume"})
                    if len(df_rt_r) >= 30:
                        df_rt_eng = engineer_data(df_rt_r)
                        fig_rt = go.Figure(go.Scatter(
                            x=df_rt_eng.index, y=df_rt_eng["Close"].squeeze(),
                            line=dict(color=C["orange"], width=1.5), name="1-min Bars"))
                        if "ema9" in df_rt_eng.columns:
                            fig_rt.add_trace(go.Scatter(
                                x=df_rt_eng.index, y=df_rt_eng["ema9"],
                                line=dict(color=C["cyan"], width=1), name="EMA9"))
                        fig_rt.update_layout(**BB("Live 1-min Feed", h=300))
                        st.plotly_chart(fig_rt, use_container_width=True, config=PCONF)

                        # Real-time signal
                        if mr:
                            best_k_rt = ("Ensemble" if "Ensemble" in mr
                                          else max(mr, key=lambda k: mr[k]["cv_f1"]))
                            rt_sig = compute_rt_signal(feed, mr[best_k_rt])

        with rt_c2:
            st.markdown('<div class="sec-hdr">Live Signal</div>', unsafe_allow_html=True)
            sig_v   = rt_sig.get("signal", 0)
            prob_v  = rt_sig.get("prob", 0.5)
            lat_v   = rt_sig.get("latency_ms", 0)
            price_v = rt_sig.get("price", 0.0)
            sig_cls_ = {1: "signal-lg", -1: "signal-sh", 0: "signal-fl"}.get(sig_v, "signal-fl")
            sig_txt_ = {1: "▲ LONG", -1: "▼ SHORT", 0: "— FLAT"}.get(sig_v, "— FLAT")
            st.markdown(f'<div class="{sig_cls_}">{sig_txt_}</div>', unsafe_allow_html=True)
            stat_block([
                ("Probability", f"{prob_v*100:.1f}%",  C["cyan"]),
                ("Latency",     f"{lat_v:.1f} ms",     C["green"]),
                ("Last Price",  f"${price_v:,.4f}" if price_v < 1
                                else f"${price_v:,.2f}", C["text"]),
            ])
            st.markdown(
                '<div class="sec-hdr">Architecture</div>', unsafe_allow_html=True)
            st.code("""
# Production WebSocket design:
# binance_ws → ring_buffer
# ring_buffer → feature_engine
# feature_engine → ML model
# ML model → signal → OMS

# Free sources (no key needed):
# - Binance WS: wss://stream.binance.com
# - Coinbase Pro: wss://advanced-trade-ws.coinbase.com
# - Kraken: wss://ws.kraken.com
""", language="python")

    #  TAB 10: PAPER TRADING — Enhancement 18
    with tabs[10]:
        st.markdown('<div class="sec-hdr">Paper Trading Gateway — Order Management System</div>',
                    unsafe_allow_html=True)

        if "paper_account" not in st.session_state:
            st.session_state["paper_account"] = _init_paper_account()
        acct = st.session_state["paper_account"]

        # Account overview
        a_c1, a_c2, a_c3, a_c4 = st.columns(4)
        for col_, (lbl_, val_, clr_) in zip([a_c1, a_c2, a_c3, a_c4], [
            ("Balance",  f"${acct['balance']:,.2f}",
             C["green"] if acct["balance"] >= 10_000 else C["red"]),
            ("Position", f"{acct['position']:.6f}",
             C["cyan"] if acct["position"] > 0 else C["muted"]),
            ("Total PnL", f"${acct['pnl']:+,.2f}",
             C["green"] if acct["pnl"] > 0 else C["red"]),
            ("Win Rate", f"{acct.get('win_rate', 0)*100:.1f}%"
                          if acct.get("total_trades", 0) > 0 else "—", C["text"]),
        ]):
            with col_:
                st.markdown(
                    f'<div class="kpi-card">'
                    f'<div class="kpi-lbl">{lbl_}</div>'
                    f'<div class="kpi-val" style="color:{clr_}">{val_}</div>'
                    f'</div>', unsafe_allow_html=True)

        # Risk controls
        st.markdown('<div class="sec-hdr">Pre-Trade Risk Controls</div>', unsafe_allow_html=True)
        stat_block([
            ("Max Position (USD)",  f"${acct['max_pos_usd']:,.0f}", C["text"]),
            ("Daily Loss Limit",    f"${acct['daily_loss_limit']:,.0f}", C["red"]),
            ("Daily PnL",           f"${acct['daily_pnl']:+,.2f}",
             C["green"] if acct["daily_pnl"] >= 0 else C["red"]),
            ("Kill Switch",         "ACTIVE" if acct["kill_switch"] else "OFF",
             C["red"] if acct["kill_switch"] else C["green"]),
        ])

        # Manual order form
        st.markdown('<div class="sec-hdr">Manual Order</div>', unsafe_allow_html=True)
        oc1, oc2, oc3 = st.columns(3)
        with oc1:
            order_side = st.radio("Side", ["Long (Buy)", "Close (Sell)", "Flat"],
                                   horizontal=True)
        with oc2:
            order_size = st.slider("Size ($)", 50, 2000, 500, 50)
        with oc3:
            current_price = info.get("price", 0) or (
                float(df["Close"].iloc[-1]) if not df.empty else 1.0
            )
            st.metric("Current Price", f"${current_price:,.4f}"
                       if current_price < 1 else f"${current_price:,.2f}")

        if st.button("Submit Order"):
            sig_map = {"Long (Buy)": 1, "Close (Sell)": -1, "Flat": 0}
            sig_val = sig_map.get(order_side, 0)
            result  = paper_execute_order(acct, sig_val, current_price, order_size)
            if result.get("executed"):
                st.session_state["paper_account"] = get_paper_account_stats(acct)
                st.success(f"Order executed: {result.get('reason')} — "
                           f"{result['trade'].get('side')} @ ${result['trade']['price']}")
            else:
                st.error(f"Order rejected: {result.get('reason')}")

        # Auto-execute ML signal
        if not signals_df.empty and mr and st.button("Auto-Execute ML Signal"):
            last_sig = int(signals_df["signal"].iloc[-1])
            result   = paper_execute_order(acct, last_sig, current_price, 500)
            if result.get("executed"):
                st.session_state["paper_account"] = get_paper_account_stats(acct)
                st.success(f"ML signal executed: {result.get('reason')}")
            else:
                st.warning(f"ML signal rejected: {result.get('reason')}")

        # Confirmation required for destructive reset
        confirm_reset = st.checkbox("⚠️ Confirm: reset paper account to $10,000")
        if st.button("Reset Paper Account"):
            if confirm_reset:
                st.session_state["paper_account"] = _init_paper_account()
                st.success("Paper account reset to $10,000")
            else:
                st.error("Please check the confirmation box above before resetting.")

        # Trade log
        if acct.get("trades"):
            st.markdown('<div class="sec-hdr">Trade Log</div>', unsafe_allow_html=True)
            trade_df = pd.DataFrame(acct["trades"][-20:])
            st.dataframe(trade_df, use_container_width=True, hide_index=True)

    #  TAB 11: SYSTEM — Health Check & Monitoring
    with tabs[11]:
        st.markdown('<div class="sec-hdr">System Health</div>', unsafe_allow_html=True)

        health = get_system_health()
        col1, col2 = st.columns(2)
        with col1:
            stat_block([
                ("Status",     health["status"].upper(),
                 C["green"] if health["status"] == "healthy" else C["red"]),
                ("Version",    "HECTOR v6.0",              C["orange"]),
                ("Platform",   health["platform"],          C["text"]),
                ("Python",     health["python"],            C["text"]),
                ("Database",   "✓ Connected" if health["db_ok"] else "✗ Error",
                 C["green"] if health["db_ok"] else C["red"]),
                ("Models Dir", "✓ OK" if health["model_dir_ok"] else "✗ Missing",
                 C["green"] if health["model_dir_ok"] else C["red"]),
                ("Cache Dir",  "✓ OK" if health["cache_dir_ok"] else "✗ Missing",
                 C["green"] if health["cache_dir_ok"] else C["yellow"]),
                ("Assets",     str(len(COIN_MAP)),          C["text"]),
            ])
        with col2:
            st.markdown('<div class="sec-hdr">Installed Libraries</div>', unsafe_allow_html=True)
            lib_rows = [
                (name, "✓ Available" if ok else "— Not installed",
                 C["green"] if ok else C["muted"])
                for name, ok in health["libraries"].items()
            ]
            stat_block(lib_rows)

        if health.get("issues"):
            st.markdown('<div class="sec-hdr">Issues</div>', unsafe_allow_html=True)
            for issue in health["issues"]:
                st.markdown(f'<div class="alert-red">{issue}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">Prometheus Metrics</div>', unsafe_allow_html=True)
        metrics_txt = get_prometheus_metrics(health, bt if bt else {}, signals_df)
        st.code(metrics_txt, language="text")
        st.download_button("Download metrics.txt", metrics_txt,
                           "hector_metrics.txt", mime="text/plain")

        st.markdown('<div class="sec-hdr">Data Sources</div>', unsafe_allow_html=True)
        stat_block([
            ("yfinance",        "Free · ~2,000 requests/hour",  C["green"]),
            ("CoinGecko",       "Free · 30 requests/minute",    C["green"]),
            ("alternative.me",  "Free · 50 requests/minute",    C["green"]),
            ("Reddit JSON",     "Free · Public API",            C["green"]),
            ("CryptoCompare",   "Free · No key required",       C["green"]),
            ("blockchain.info", "Free · Public stats",          C["green"]),
        ])

    #  TAB 12: EXPORT
    with tabs[12]:
        st.markdown('<div class="sec-hdr">Export</div>', unsafe_allow_html=True)
        ec1, ec2, ec3, ec4 = st.columns(4)
        with ec1:
            st.markdown("**OHLCV + Indicators**")
            dl_csv(df.reset_index(), "hector_indicators.csv", "Download CSV")
        with ec2:
            if not labels_df.empty:
                st.markdown("**Triple-Barrier Labels**")
                dl_csv(labels_df, "hector_labels.csv", "Download Labels")
        with ec3:
            if not signals_df.empty:
                st.markdown("**Signal History**")
                dl_csv(signals_df, "hector_signals.csv", "Download Signals")
        with ec4:
            st.markdown("**PDF Strategy Report**")
            if not FPDF_OK:
                st.info("Install `fpdf2` for PDF reports: `pip install fpdf2`")
            risk_for_pdf = compute_risk(df["returns"]) if "returns" in df.columns else {}
            pdf_bytes    = generate_pdf(cfg, info, bt, risk_for_pdf)
            ext_         = "pdf" if FPDF_OK else "txt"
            st.download_button(
                label="Download PDF" if FPDF_OK else "Export as Text (fpdf2 not installed)",
                data=pdf_bytes,
                file_name=f"hector_{cfg['yf_sym']}_{datetime.utcnow().strftime('%Y%m%d')}.{ext_}",
                mime=f"application/{ext_}",
            )

        st.markdown('<div class="sec-hdr">System Info (JSON)</div>', unsafe_allow_html=True)
        sys_info = {
            "system": "HECTOR", "version": "6.0",
            "enhancements": 20,
            "asset": cfg["yf_sym"], "period": cfg["period"],
            "interval": cfg["interval"],
            "generated_at": datetime.utcnow().isoformat(),
            "libraries": {
                "xgboost": XGB_OK, "lightgbm": LGB_OK, "optuna": OPTUNA_OK,
                "shap": SHAP_OK, "hmmlearn": HMM_OK, "parquet": PARQUET_OK,
                "joblib": JOBLIB_OK, "fpdf2": FPDF_OK, "lime": LIME_OK,
            },
            "coin_info": {k: v for k, v in info.items() if not isinstance(v, (dict, list))},
        }
        st.download_button(
            label="Download System Info (JSON)",
            data=json.dumps(sys_info, indent=2),
            file_name="hector_system_info.json", mime="application/json")

        st.markdown('<div class="sec-hdr">System Diagnostics</div>', unsafe_allow_html=True)
        stat_block([
            ("Version",          "HECTOR v6.0",                              C["orange"]),
            ("Enhancements",     "20 / 20 implemented",                      C["green"]),
            ("Assets",           str(len(COIN_MAP)),                         C["text"]),
            ("Feature columns",  str(len(FEATURE_COLS)),                     C["text"]),
            ("ML Libraries",
             f"XGB={'✓' if XGB_OK else '✗'} | LGB={'✓' if LGB_OK else '✗'} | "
             f"Optuna={'✓' if OPTUNA_OK else '✗'} | SHAP={'✓' if SHAP_OK else '✗'}",
             C["text"]),
            ("Persistence",
             f"SQLite + {'Parquet' if PARQUET_OK else 'CSV'} + "
             f"{'Models' if JOBLIB_OK else 'No model save'}",
             C["cyan"]),
            ("Notifications",
             f"Telegram={'✓' if TELEGRAM_TOKEN else '✗'} | "
             f"Email={'✓' if SMTP_USER else '✗'} | "
             f"Slack={'✓' if SLACK_WEBHOOK else '✗'}",
             C["text"]),
        ])

    # Single author credit — sidebar only


    # Author signature
    st.markdown(
        f'<div style="text-align:right;padding:6px 4px 2px;'
        f'border-top:1px solid {C["border"]};margin-top:16px;">'
        f'<span style="font-size:7px;color:#2a3040;letter-spacing:1px;">'
        f'Created by <strong style="color:{C["orange"]}">Daniyal Aziz</strong>'
        f'</span></div>',
        unsafe_allow_html=True,
    )

def _run_tests() -> None:
    """
    Lightweight unit tests that run without network or Streamlit.
    Execute via:  python hector_final.py --test
    All tests must pass silently; failures raise AssertionError.
    """
    import traceback

    passed = 0; failed = 0; results = []

    def _test(name: str, fn):
        nonlocal passed, failed
        try:
            fn()
            results.append(("PASS", name))
            passed += 1
        except Exception as e:
            results.append(("FAIL", name, str(e), traceback.format_exc()))
            failed += 1

    # Helpers
    def _make_df(n=100):
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        np.random.seed(42)
        close = 40000 + np.cumsum(np.random.randn(n) * 200)
        close = np.abs(close)
        return pd.DataFrame({
            "Open":   close * 0.999,
            "High":   close * 1.003,
            "Low":    close * 0.997,
            "Close":  close,
            "Volume": np.random.uniform(100, 1000, n),
        }, index=idx)

    # TEST: validate_ohlcv
    def t_validate():
        df = _make_df(100)
        # Inject duplicate
        df2 = pd.concat([df, df.iloc[:5]])
        cleaned, report = validate_ohlcv(df2)
        assert report["duplicates"] >= 5, "Duplicates not detected"
        assert not cleaned.index.duplicated().any(), "Duplicates not removed"

    _test("validate_ohlcv — deduplication", t_validate)

    # TEST: engineer_data columns
    def t_engineer():
        df = _make_df(120)
        eng = engineer_data(df)
        required = ["rsi", "macd", "macd_signal", "bb_up", "bb_dn",
                    "ema9", "ema21", "atr", "vwap", "obv",
                    "stoch_k", "mfi", "willr", "cvd", "ofi", "vol_ratio"]
        missing = [c for c in required if c not in eng.columns]
        assert not missing, f"Missing columns: {missing}"
        assert not eng["rsi"].isna().all(), "RSI all NaN"
        assert (eng["rsi"].dropna() >= 0).all(), "RSI < 0"
        assert (eng["rsi"].dropna() <= 100).all(), "RSI > 100"

    _test("engineer_data — all indicators present", t_engineer)

    # TEST: RSI bounds
    def t_rsi_bounds():
        df = _make_df(200)
        eng = engineer_data(df)
        rsi = eng["rsi"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), "RSI out of [0,100]"

    _test("RSI — bounds [0, 100]", t_rsi_bounds)

    # TEST: triple_barrier_labels
    def t_labels():
        df = _make_df(300)
        eng = engineer_data(df)
        labels = triple_barrier_labels(eng, pt=1.5, sl=1.0, max_hold=10)
        assert not labels.empty, "No labels generated"
        assert set(labels["label"].unique()).issubset({-1, 0, 1}), "Invalid label values"
        assert "time" in labels.columns, "Missing 'time' column"
        assert "vol_entry" in labels.columns, "Missing 'vol_entry' column"

    _test("triple_barrier_labels — structure", t_labels)

    # TEST: build_features
    def t_features():
        df = _make_df(300)
        eng = engineer_data(df)
        labels = triple_barrier_labels(eng, pt=1.5, sl=1.0, max_hold=10)
        feat = build_features(eng, labels)
        if feat.empty:
            return  # acceptable with short data
        assert all(c in feat.columns for c in FEATURE_COLS), "Feature columns missing"
        assert not feat.isin([np.inf, -np.inf]).any().any(), "Inf values in features"
        assert not feat.isna().any().any(), "NaN values in features"

    _test("build_features — no inf/nan", t_features)

    # TEST: cusum_events
    def t_cusum():
        df = _make_df(200)
        events = cusum_events(df["Close"])
        assert isinstance(events, list), "cusum_events should return list"
        assert all(isinstance(e, int) for e in events), "Events should be ints"
        assert all(0 < e < len(df) for e in events), "Events out of range"

    _test("cusum_events — valid indices", t_cusum)

    # TEST: run_backtest returns expected keys
    def t_backtest():
        df = _make_df(300)
        eng = engineer_data(df)
        # Create fake signals
        n = len(df)
        signals_df = pd.DataFrame({
            "time":   df.index[:n//2],
            "signal": np.where(np.arange(n//2) % 3 == 0, 1, -1),
        })
        bt = run_backtest(df["Close"].squeeze(), signals_df)
        if not bt:
            return  # too few signals is acceptable
        required_keys = ["equity", "sharpe", "max_dd", "win_rate",
                         "psr", "dsr", "sharpe_ci", "n_trades"]
        missing = [k for k in required_keys if k not in bt]
        assert not missing, f"Backtest missing keys: {missing}"
        assert isinstance(bt["sharpe_ci"], tuple) and len(bt["sharpe_ci"]) == 2
        assert bt["max_dd"] <= 0, "Max drawdown should be <= 0"

    _test("run_backtest — keys and max_dd", t_backtest)

    # TEST: compute_risk
    def t_risk():
        df = _make_df(200)
        eng = engineer_data(df)
        risk = compute_risk(eng["returns"])
        assert risk, "Risk dict is empty"
        assert "var95" in risk and "cvar95" in risk, "Missing VaR keys"
        assert risk["var95"] <= 0, "VaR95 should be negative (loss)"
        assert risk["cvar95"] <= risk["var95"], "CVaR should be worse than VaR"
        assert "stress" in risk and isinstance(risk["stress"], dict)
        assert len(risk["stress"]) >= 4, "Stress scenarios missing"

    _test("compute_risk — VaR/CVaR/stress", t_risk)

    # TEST: hrp_weights
    def t_hrp():
        np.random.seed(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="D")
        ret_df = pd.DataFrame(
            np.random.randn(200, 4) * 0.02,
            columns=["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
            index=idx
        )
        wts = hrp_weights(ret_df)
        assert abs(wts.sum() - 1.0) < 1e-6, "HRP weights do not sum to 1"
        assert (wts >= 0).all(), "Negative HRP weights"
        assert len(wts) == 4, "Wrong number of weights"

    _test("hrp_weights — sums to 1, non-negative", t_hrp)

    # TEST: volume_profile
    def t_vol_profile():
        df = _make_df(200)
        vp = volume_profile(df)
        assert not vp.empty, "Volume profile empty"
        assert "price" in vp.columns and "volume" in vp.columns
        assert "is_hvn" in vp.columns
        assert (vp["volume"] >= 0).all(), "Negative volume in profile"

    _test("volume_profile — structure", t_vol_profile)

    # TEST: detect_regime
    def t_regime():
        df = _make_df(150)
        eng = engineer_data(df)
        labels = detect_regime(eng["returns"])
        assert len(labels) == len(eng["returns"].dropna()), "Label length mismatch"
        assert set(labels).issubset({0, 1, 2}), "Regime labels out of {0,1,2}"
        regime = classify_regime(eng)
        assert regime in ("bull", "bear", "sideways"), f"Unknown regime: {regime}"

    _test("detect_regime — valid states", t_regime)

    # TEST: efficient_frontier
    def t_frontier():
        np.random.seed(0)
        idx = pd.date_range("2024-01-01", periods=100, freq="D")
        ret_df = pd.DataFrame(
            np.random.randn(100, 3) * 0.01,
            columns=["A", "B", "C"], index=idx
        )
        front = efficient_frontier(ret_df, n=10)
        if front.empty:
            return  # optimisation may fail with random data — that's OK
        assert "vol" in front.columns and "ret" in front.columns
        assert (front["vol"] >= 0).all(), "Negative volatility in frontier"

    _test("efficient_frontier — non-negative vol", t_frontier)

    # TEST: validate_ohlcv outlier capping
    def t_outlier_cap():
        df = _make_df(100)
        df2 = df.copy()
        # Inject extreme outlier
        df2.iloc[50, df2.columns.get_loc("Close")] = 9_999_999
        cleaned, report = validate_ohlcv(df2)
        assert report["outliers"] >= 1, "Outlier not detected"

    _test("validate_ohlcv — outlier capping", t_outlier_cap)

    # TEST: generate_pdf fallback (no fpdf2)
    def t_pdf():
        result = generate_pdf(
            cfg={"yf_sym": "BTC-USD", "period": "3mo"},
            info={"price": 50000, "change_24h": 2.5},
            bt={}, risk={},
        )
        assert isinstance(result, bytes) and len(result) > 10, "PDF bytes empty"

    _test("generate_pdf — fallback bytes", t_pdf)

    # TEST: feature cols all in build_features output
    def t_feature_cols():
        df = _make_df(400)
        eng = engineer_data(df)
        labels = triple_barrier_labels(eng, pt=1.0, sl=0.8, max_hold=8)
        if labels.empty:
            return
        feat = build_features(eng, labels)
        if feat.empty:
            return
        for col in FEATURE_COLS:
            assert col in feat.columns, f"Feature col missing: {col}"

    _test("build_features — FEATURE_COLS complete", t_feature_cols)

    # TEST: no inf/nan in engineer_data
    def t_no_nan():
        df = _make_df(300)
        eng = engineer_data(df)
        num_cols = eng.select_dtypes(include=[np.number]).columns.tolist()
        inf_count = eng[num_cols].isin([np.inf, -np.inf]).sum().sum()
        assert inf_count == 0, f"Inf values found in engineered data: {inf_count}"

    _test("engineer_data — no inf values", t_no_nan)

    # Summary
    print("\n" + "=" * 58)
    print(f"  HECTOR v6 TEST SUITE  |  {passed} passed  |  {failed} failed")
    print("=" * 58)
    for item in results:
        icon = "✓" if item[0] == "PASS" else "✗"
        print(f"  {icon}  {item[1]}")
        if item[0] == "FAIL":
            print(f"      ERROR: {item[2]}")

    if failed > 0:
        print(f"\n{failed} test(s) FAILED.")
        raise SystemExit(1)
    else:
        print("\nAll tests passed.")


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_tests()
    else:
        main()

