"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CALYX CONTAINERS - S&OP COMMAND CENTER
    Production-Grade Sales & Operations Planning System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path
import re

# Stats & ML
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════════════════════
# 🎨 CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Calyx S&OP Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# 🎭 LEGENDARY CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1600px; }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main::before {
        content: '';
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        z-index: -1;
        opacity: 0.8;
    }
    
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    h1 {
        font-size: 3.5rem !important;
        font-weight: 900 !important;
        background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        line-height: 1.1 !important;
    }
    
    h2 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.02em;
    }
    
    h3 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    .stTabs {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    
    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        background: transparent;
        border-radius: 12px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 600;
        border: none;
        transition: all 0.3s;
        padding: 0 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    [data-testid="stDataFrame"] table { color: #ffffff !important; }
    
    [data-testid="stDataFrame"] thead tr th {
        background: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95);
        backdrop-filter: blur(20px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.6);
    }
    
    .stAlert {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 📊 DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

TABS_CONFIG = {
    "invoices_header": "_NS_Invoices_Data",
    "invoice_line": "Invoice Line Item",
    "so_line": "Sales Order Line Item",
    "so_header": "_NS_SalesOrders_Data",
    "customers": "_NS_Customer_List",
    "items": "Raw_Items",
    "vendors": "Raw_Vendors",
    "avg_leadtimes": "Average Leadtimes",
    "deals": "Deals",
    "inventory": "Raw_Inventory",
}

@dataclass
class ForecastConfig:
    model: str = "exp"
    horizon: int = 12
    freq: str = "MS"
    winsorize: bool = True
    
@dataclass
class Scenario:
    name: str
    growth_rate: float
    demand_weight: float
    sales_weight: float
    created_at: str
    forecast_data: dict

SCENARIO_DIR = Path(".scenarios")
SCENARIO_DIR.mkdir(exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🛠️ UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(x: Any) -> float:
    if pd.isna(x) or x == "": return 0.0
    try:
        return float(str(x).replace(",", "").replace("$", "").strip())
    except: return 0.0

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def format_currency(x: float) -> str:
    if x >= 1_000_000: return f"${x/1_000_000:.2f}M"
    if x >= 1_000: return f"${x/1_000:.1f}K"
    return f"${x:,.0f}"

def format_qty(x: float) -> str:
    if x >= 1_000_000: return f"{x/1_000_000:.2f}M"
    if x >= 1_000: return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"

ALIASES = {
    "date": ["date", "trandate", "transactiondate", "createddate", "orderdate", "closedate", "actualshipdate"],
    "sku": ["sku", "item", "itemname", "itemid", "product"],
    "customer": ["customer", "customername", "entity", "company", "customercompanyname", "correctedcustomer"],
    "qty": ["quantity", "qty", "quantityordered", "quantityfulfilled"],
    "amount": ["amount", "total", "totalamount", "revenue", "amounttransactiontotal"],
    "category": ["category", "producttype", "calyxproducttype", "type"],
    "sales_rep": ["salesrep", "salesperson", "rep", "owner", "repmaster", "masterrep"],
    "lead_time": ["leadtime", "purchaseleadtime", "avgleadtime"],
    "so_number": ["sonumber", "documentnumber", "internalid"],
    "status": ["status", "orderstatus", "invoicestatus"],
}

def resolve_col(df: pd.DataFrame, role: str) -> Optional[str]:
    if df.empty: return None
    cols = df.columns.tolist()
    norm_cols = {_norm(c): c for c in cols}
    for alias in ALIASES.get(role, []):
        if _norm(alias) in norm_cols:
            return norm_cols[_norm(alias)]
    for col in cols:
        for alias in ALIASES.get(role, []):
            if _norm(alias) in _norm(col):
                return col
    return None

# ══════════════════════════════════════════════════════════════════════════════
# 📥 DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> Dict[str, pd.DataFrame]:
    try:
        from google.oauth2.service_account import Credentials
        import gspread

        creds_dict = None
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        elif "service_account" in st.secrets:
            creds_dict = dict(st.secrets["service_account"])
        else:
            raise ValueError("Missing Google credentials")
        
        sheet_id = None
        if "SPREADSHEET_ID" in st.secrets:
            sheet_id = st.secrets["SPREADSHEET_ID"]
        elif "gsheets" in st.secrets:
            sheet_id = st.secrets["gsheets"].get("spreadsheet_id")
        
        if not sheet_id:
            raise ValueError("Missing spreadsheet ID")

        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        sh = client.open_by_key(sheet_id)
        
        data = {}
        progress = st.progress(0, text="🚀 Loading data...")
        
        for i, (key, tab_name) in enumerate(TABS_CONFIG.items()):
            try:
                ws = sh.worksheet(tab_name)
                rows = ws.get_all_values()
                header_idx = 1 if tab_name == "Deals" else 0
                
                if len(rows) > header_idx + 1:
                    headers = rows[header_idx]
                    seen = {}
                    clean_headers = []
                    for h in headers:
                        h = str(h).strip()
                        if h in seen:
                            seen[h] += 1
                            h = f"{h}_{seen[h]}"
                        else:
                            seen[h] = 0
                        clean_headers.append(h)
                    
                    df = pd.DataFrame(rows[header_idx + 1:], columns=clean_headers)
                    df = df.replace('', np.nan)
                    data[key] = df
                else:
                    data[key] = pd.DataFrame()
            except:
                data[key] = pd.DataFrame()
            
            progress.progress((i + 1) / len(TABS_CONFIG), text=f"Loading {tab_name}...")
        
        progress.empty()
        return data
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# 📊 DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_demand_history(data: Dict[str, pd.DataFrame], freq="MS") -> pd.DataFrame:
    """Prepare demand history with Corrected Customer and Master Rep"""
    df = data["so_line"].copy()
    if df.empty:
        return pd.DataFrame()
    
    # Resolve columns - prioritize Corrected Customer and Master Rep
    c_date = resolve_col(df, "date")
    c_sku = resolve_col(df, "sku")
    c_qty = resolve_col(df, "qty")
    c_so = resolve_col(df, "so_number")
    
    # Try to find Corrected Customer first, fallback to Customer
    c_customer = None
    for col in df.columns:
        if "corrected" in _norm(col) and "customer" in _norm(col):
            c_customer = col
            break
    if not c_customer:
        c_customer = resolve_col(df, "customer")
    
    if not all([c_date, c_sku, c_qty]):
        return pd.DataFrame()
    
    df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
    df = df.dropna(subset=[c_date])
    
    period_freq = "M" if freq == "MS" else freq
    df["ds"] = df[c_date].dt.to_period(period_freq).dt.to_timestamp()
    df["sku"] = df[c_sku].astype(str).str.strip()
    df["qty"] = df[c_qty].apply(_safe_float)
    df["customer"] = df[c_customer].astype(str).str.strip() if c_customer else "Unknown"
    df["so_number"] = df[c_so].astype(str).str.strip() if c_so else ""
    
    # Aggregate
    agg_cols = ["ds", "sku", "customer"]
    if c_so:
        out = df.groupby(agg_cols + ["so_number"], as_index=False)["qty"].sum()
    else:
        out = df.groupby(agg_cols, as_index=False)["qty"].sum()
    
    # Add categories
    items = data["items"]
    if not items.empty:
        c_item_sku = resolve_col(items, "sku")
        c_category = resolve_col(items, "category")
        if c_item_sku and c_category:
            cat_map = dict(zip(
                items[c_item_sku].astype(str).str.strip(),
                items[c_category].astype(str).str.strip()
            ))
            out["category"] = out["sku"].map(cat_map).fillna("Uncategorized")
        else:
            out["category"] = "Uncategorized"
    else:
        out["category"] = "Uncategorized"
    
    return out.sort_values("ds")

def prepare_invoice_history(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prepare invoice history for AR analysis"""
    df = data["invoices_header"].copy()
    if df.empty:
        return pd.DataFrame()
    
    c_date = resolve_col(df, "date")
    c_customer = None
    for col in df.columns:
        if "corrected" in _norm(col) and "customer" in _norm(col):
            c_customer = col
            break
    if not c_customer:
        c_customer = resolve_col(df, "customer")
    
    c_amount = resolve_col(df, "amount")
    c_status = resolve_col(df, "status")
    
    if not all([c_date, c_customer, c_amount]):
        return pd.DataFrame()
    
    df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
    df = df.dropna(subset=[c_date])
    df["customer"] = df[c_customer].astype(str).str.strip()
    df["amount"] = df[c_amount].apply(_safe_float)
    df["status"] = df[c_status].astype(str).str.strip() if c_status else "Unknown"
    
    return df[["customer", "amount", "status", c_date]].rename(columns={c_date: "date"})

def prepare_pipeline(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prepare HubSpot pipeline data"""
    df = data["deals"].copy()
    if df.empty:
        return pd.DataFrame()
    
    c_date = resolve_col(df, "date")
    c_amount = resolve_col(df, "amount")
    c_category = resolve_col(df, "category")
    
    if not all([c_date, c_amount]):
        return pd.DataFrame()
    
    df[c_date] = pd.to_datetime(df[c_date], errors="coerce")
    df = df.dropna(subset=[c_date])
    df["ds"] = df[c_date].dt.to_period("M").dt.to_timestamp()
    df["amount"] = df[c_amount].apply(_safe_float)
    df["category"] = df[c_category].astype(str).str.strip() if c_category else "Uncategorized"
    
    return df.groupby(["ds", "category"], as_index=False)["amount"].sum()

# ══════════════════════════════════════════════════════════════════════════════
# 🔮 FORECASTING ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def forecast_exponential_smoothing(history: pd.Series, cfg: ForecastConfig) -> pd.Series:
    """Exponential Smoothing with seasonality detection"""
    if len(history) < 2:
        return _naive_forecast(history, cfg.horizon, cfg.freq)
    
    y = history.fillna(0).copy()
    if y.index.duplicated().any():
        y = y.groupby(y.index).sum()
    
    if cfg.winsorize and len(y) > 2:
        y = y.clip(y.quantile(0.01), y.quantile(0.99))
    
    try:
        model = ExponentialSmoothing(
            y,
            trend="add" if len(y) > 10 else None,
            seasonal="add" if len(y) > 24 else None,
            seasonal_periods=12 if len(y) > 24 else None
        )
        fit = model.fit(optimized=True, disp=False)
        return fit.forecast(cfg.horizon).clip(lower=0)
    except:
        return _naive_forecast(history, cfg.horizon, cfg.freq)

def forecast_arima(history: pd.Series, cfg: ForecastConfig) -> pd.Series:
    """ARIMA/SARIMA forecasting"""
    if len(history) < 10:
        return forecast_exponential_smoothing(history, cfg)
    
    y = history.fillna(0).copy()
    if y.index.duplicated().any():
        y = y.groupby(y.index).sum()
    
    try:
        # Simple ARIMA(1,1,1) - can be made auto with pmdarima
        model = SARIMAX(y, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        return fit.forecast(cfg.horizon).clip(lower=0)
    except:
        return forecast_exponential_smoothing(history, cfg)

def forecast_ml(history: pd.Series, cfg: ForecastConfig) -> pd.Series:
    """Machine Learning forecast with lag features"""
    if len(history) < 12:
        return forecast_exponential_smoothing(history, cfg)
    
    y = history.fillna(0).copy()
    
    # Create features
    df = pd.DataFrame({"y": y})
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    df["month"] = df.index.month
    df["rolling_3"] = df["y"].rolling(3).mean()
    df["rolling_6"] = df["y"].rolling(6).mean()
    df = df.dropna()
    
    if len(df) < 6:
        return forecast_exponential_smoothing(history, cfg)
    
    X = df.drop(columns=["y"])
    Y = df["y"]
    
    try:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, Y)
        
        # Recursive forecasting
        future_preds = []
        last_row = df.iloc[-1:].copy()
        last_date = y.index[-1]
        future_dates = pd.date_range(start=last_date, periods=cfg.horizon + 1, freq=cfg.freq)[1:]
        
        for d in future_dates:
            new_row = last_row.copy()
            new_row["month"] = d.month
            # Simplified lag update
            for lag in [1, 2, 3, 6, 12]:
                if f"lag_{lag}" in new_row.columns:
                    new_row[f"lag_{lag}"] = future_preds[-lag] if len(future_preds) >= lag else last_row["y"].iloc[0]
            
            pred = model.predict(new_row.drop(columns=["y"], errors="ignore"))[0]
            future_preds.append(max(0, pred))
            last_row = new_row
            last_row["y"] = pred
        
        return pd.Series(future_preds, index=future_dates)
    except:
        return forecast_exponential_smoothing(history, cfg)

def _naive_forecast(history: pd.Series, horizon: int, freq: str) -> pd.Series:
    """Fallback naive forecast"""
    last_val = history.iloc[-1] if len(history) > 0 else 0.0
    future_dates = pd.date_range(
        start=history.index[-1] if len(history) > 0 else pd.Timestamp.now(),
        periods=horizon + 1, freq=freq
    )[1:]
    return pd.Series([last_val] * horizon, index=future_dates)

def run_forecast(history: pd.Series, cfg: ForecastConfig) -> pd.Series:
    """Route to appropriate forecast model"""
    if cfg.model == "exp":
        return forecast_exponential_smoothing(history, cfg)
    elif cfg.model == "arima":
        return forecast_arima(history, cfg)
    elif cfg.model == "ml":
        return forecast_ml(history, cfg)
    else:
        return forecast_exponential_smoothing(history, cfg)

# ══════════════════════════════════════════════════════════════════════════════
# 📈 VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def create_multi_layer_forecast_chart(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    pipeline: Optional[pd.DataFrame] = None,
    title: str = "Demand Forecast"
):
    """Create comprehensive forecast chart with multiple overlays"""
    fig = go.Figure()
    
    # Historical demand
    fig.add_trace(go.Scatter(
        x=historical['ds'],
        y=historical['qty'],
        mode='lines',
        name='Historical Demand',
        line=dict(color='rgba(255, 255, 255, 0.8)', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)',
        hovertemplate='<b>%{x|%b %Y}</b><br>Qty: %{y:,.0f}<extra></extra>'
    ))
    
    # Forecasted demand
    if not forecast.empty:
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['qty_forecast'],
            mode='lines+markers',
            name='Demand Forecast',
            line=dict(color='#00f2fe', width=3, dash='dash'),
            marker=dict(size=8, color='#00f2fe'),
            fill='tozeroy',
            fillcolor='rgba(79, 172, 254, 0.2)',
            hovertemplate='<b>%{x|%b %Y}</b><br>Forecast: %{y:,.0f}<extra></extra>'
        ))
    
    # Pipeline overlay
    if pipeline is not None and not pipeline.empty:
        fig.add_trace(go.Scatter(
            x=pipeline['ds'],
            y=pipeline['pipeline_amount'],
            mode='lines',
            name='Pipeline Trend',
            line=dict(color='#f5576c', width=2, dash='dot'),
            hovertemplate='<b>%{x|%b %Y}</b><br>Pipeline: %{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color='white'), x=0.5, xanchor='center'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor='rgba(255,255,255,0.1)', bordercolor='rgba(255,255,255,0.2)', borderwidth=1
        ),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)', zeroline=False, title='Units'),
        height=500
    )
    
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# 💾 SCENARIO MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def save_scenario(scenario: Scenario):
    """Save scenario to disk"""
    filepath = SCENARIO_DIR / f"{scenario.name.replace(' ', '_')}.json"
    with open(filepath, 'w') as f:
        json.dump(asdict(scenario), f)

def load_scenarios() -> List[Scenario]:
    """Load all saved scenarios"""
    scenarios = []
    for filepath in SCENARIO_DIR.glob("*.json"):
        with open(filepath, 'r') as f:
            data = json.load(f)
            scenarios.append(Scenario(**data))
    return scenarios

# ══════════════════════════════════════════════════════════════════════════════
# 🚀 MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ═══ SIDEBAR ═══
    with st.sidebar:
        st.markdown("## ⚙️ Forecast Controls")
        st.markdown("---")
        
        model = st.selectbox(
            "📊 Model",
            ["Exponential Smoothing", "ARIMA/SARIMA", "Machine Learning"],
            help="Select forecasting algorithm"
        )
        model_map = {
            "Exponential Smoothing": "exp",
            "ARIMA/SARIMA": "arima",
            "Machine Learning": "ml"
        }
        
        horizon = st.select_slider(
            "📅 Horizon",
            options=[3, 6, 9, 12, 18, 24],
            value=12,
            help="Forecast horizon in months"
        )
        
        freq = st.selectbox("📊 Frequency", ["Monthly", "Weekly"], index=0)
        freq_code = "MS" if freq == "Monthly" else "W"
        
        winsorize = st.checkbox("🎯 Clip Outliers", value=True, help="Remove extreme values")
        
        cfg = ForecastConfig(model=model_map[model], horizon=horizon, freq=freq_code, winsorize=winsorize)
        
        st.markdown("---")
        st.markdown("### 📡 Status")
        st.success("✓ Connected")
        st.caption(f"Updated: {datetime.now().strftime('%I:%M %p')}")
    
    # ═══ HEADER ═══
    st.markdown('<h1>🚀 S&OP COMMAND CENTER</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">Sales & Operations Planning Intelligence</p>', unsafe_allow_html=True)
    
    # ═══ LOAD DATA ═══
    with st.spinner("🔮 Loading intelligence..."):
        data = load_data()
    
    dem = prepare_demand_history(data, freq=cfg.freq)
    invoices = prepare_invoice_history(data)
    pipeline = prepare_pipeline(data)
    
    if dem.empty:
        st.error("🚫 No demand data")
        st.stop()
    
    # Get lists
    customers = sorted(dem["customer"].dropna().unique())
    skus = sorted(dem["sku"].dropna().unique())
    categories = sorted(dem["category"].dropna().unique())
    
    # Sales reps from Master Rep field
    sales_reps = []
    rep_map = {}
    if not data["customers"].empty:
        c_cust = None
        for col in data["customers"].columns:
            if "corrected" in _norm(col) and "customer" in _norm(col):
                c_cust = col
                break
        if not c_cust:
            c_cust = resolve_col(data["customers"], "customer")
        
        c_rep = None
        for col in data["customers"].columns:
            if "master" in _norm(col) and "rep" in _norm(col):
                c_rep = col
                break
        if not c_rep:
            c_rep = resolve_col(data["customers"], "sales_rep")
        
        if c_cust and c_rep:
            rep_map = dict(zip(
                data["customers"][c_cust].astype(str).str.strip(),
                data["customers"][c_rep].astype(str).str.strip()
            ))
            sales_reps = sorted(set(rep_map.values()))
    
    # ═══ METRICS ═══
    current_date = dem['ds'].max()
    l12m = dem[dem['ds'] > (current_date - pd.DateOffset(months=12))]['qty'].sum()
    prev_l12m = dem[
        (dem['ds'] <= (current_date - pd.DateOffset(months=12))) &
        (dem['ds'] > (current_date - pd.DateOffset(months=24)))
    ]['qty'].sum()
    delta = ((l12m - prev_l12m) / prev_l12m * 100) if prev_l12m > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 L12M UNITS", format_qty(l12m), f"{delta:+.1f}%")
    col2.metric("🎯 ACTIVE SKUS", f"{len(skus):,}")
    col3.metric("👥 CUSTOMERS", f"{len(customers):,}")
    col4.metric("📊 CATEGORIES", f"{len(categories):,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ═══ TABS ═══
    tabs = st.tabs(["📊 Sales Rep View", "🏭 Operations", "🔮 Scenario Planning"])
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: SALES REP VIEW
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("## 💎 Sales Intelligence")
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            rep = st.selectbox("👤 Sales Rep", ["🌐 All Reps"] + sales_reps, index=0)
        
        with c2:
            if rep != "🌐 All Reps" and rep_map:
                rep_customers = [c for c, r in rep_map.items() if r == rep]
                customer = st.selectbox("🏢 Customer", ["🌐 All"] + rep_customers)
            else:
                customer = st.selectbox("🏢 Customer", ["🌐 All"] + customers)
        
        with c3:
            if customer != "🌐 All" and not dem.empty:
                cust_dem = dem[dem["customer"] == customer]
                cust_skus = sorted(cust_dem["sku"].unique())
            else:
                cust_dem = dem
                cust_skus = skus
            
            sku = st.selectbox("📦 SKU", ["🌐 All"] + cust_skus, index=0)
        
        # Filter
        filtered = cust_dem.copy()
        if sku != "🌐 All":
            filtered = filtered[filtered["sku"] == sku]
        
        if not filtered.empty:
            # SKU Breakdown Table
            st.markdown("---")
            st.markdown("### 📊 SKU Performance Breakdown")
            
            sku_summary = []
            for s, g in filtered.groupby("sku"):
                total_qty = g["qty"].sum()
                order_count = len(g)
                so_numbers = g["so_number"].unique() if "so_number" in g.columns else []
                so_list = ", ".join([so for so in so_numbers if so][:5])  # First 5 SOs
                if len(so_numbers) > 5:
                    so_list += f" (+{len(so_numbers)-5} more)"
                
                sku_summary.append({
                    "SKU": s,
                    "Total Qty": f"{total_qty:,.0f}",
                    "Orders": order_count,
                    "SO Numbers": so_list if so_list else "N/A",
                    "Avg per Order": f"{total_qty/order_count:,.1f}" if order_count > 0 else "0"
                })
            
            sku_df = pd.DataFrame(sku_summary)
            st.dataframe(sku_df, use_container_width=True, hide_index=True)
            
            # Forecast Chart
            st.markdown("---")
            st.markdown("### 📈 Demand Forecast")
            
            fc_results = []
            skipped = 0
            
            for s, g in filtered.groupby("sku"):
                hist = g.groupby("ds")["qty"].sum()
                if len(hist) < 2:
                    skipped += 1
                    continue
                fc = run_forecast(hist, cfg)
                fc_df = fc.reset_index()
                fc_df.columns = ["ds", "qty_forecast"]
                fc_df["sku"] = s
                fc_results.append(fc_df)
            
            if skipped > 0:
                st.info(f"ℹ️ {skipped} SKUs skipped (need 2+ data points)")
            
            if fc_results:
                all_fc = pd.concat(fc_results, ignore_index=True)
                agg_fc = all_fc.groupby("ds")["qty_forecast"].sum().reset_index()
                hist_agg = filtered.groupby("ds")["qty"].sum().reset_index()
                
                title = f"🎯 {customer if customer != '🌐 All' else 'All Customers'} - Forecast"
                fig = create_multi_layer_forecast_chart(hist_agg, agg_fc, title=title)
                st.plotly_chart(fig, use_container_width=True)
                
                # Quarterly Breakdown
                st.markdown("---")
                st.markdown("### 📅 Quarterly Recommendations")
                
                agg_fc["quarter"] = pd.to_datetime(agg_fc["ds"]).dt.to_period("Q").astype(str)
                quarterly = agg_fc.groupby("quarter")["qty_forecast"].sum().reset_index()
                quarterly.columns = ["Quarter", "Projected Units"]
                quarterly["Projected Units"] = quarterly["Projected Units"].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(quarterly, use_container_width=True, hide_index=True)
            
            # Invoice Payment Behavior
            if not invoices.empty and customer != "🌐 All":
                st.markdown("---")
                st.markdown("### 💳 Payment Behavior")
                
                cust_inv = invoices[invoices["customer"] == customer]
                if not cust_inv.empty:
                    paid = cust_inv[cust_inv["status"].str.contains("paid", case=False, na=False)]
                    open_inv = cust_inv[cust_inv["status"].str.contains("open", case=False, na=False)]
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("💰 Total Invoiced", format_currency(cust_inv["amount"].sum()))
                    col_b.metric("✅ Paid", format_currency(paid["amount"].sum()))
                    col_c.metric("⏳ Open", format_currency(open_inv["amount"].sum()))
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("## 🏭 Operations Dashboard")
        
        c1, c2 = st.columns([2, 1])
        
        with c1:
            cat_filter = st.selectbox("📂 Category", ["🌐 All"] + categories)
        with c2:
            if cat_filter != "🌐 All":
                cat_skus = sorted(dem[dem["category"] == cat_filter]["sku"].unique())
                sku_filter = st.selectbox("📦 SKU", ["🌐 All"] + cat_skus)
            else:
                sku_filter = st.selectbox("📦 SKU", ["🌐 All"] + skus)
        
        # Filter demand
        ops_dem = dem.copy()
        if cat_filter != "🌐 All":
            ops_dem = ops_dem[ops_dem["category"] == cat_filter]
        if sku_filter != "🌐 All":
            ops_dem = ops_dem[ops_dem["sku"] == sku_filter]
        
        if not ops_dem.empty:
            # Generate forecast
            hist_ops = ops_dem.groupby("ds")["qty"].sum()
            if len(hist_ops) >= 2:
                fc_ops = run_forecast(hist_ops, cfg).reset_index()
                fc_ops.columns = ["ds", "qty_forecast"]
                hist_ops_df = hist_ops.reset_index()
                hist_ops_df.columns = ["ds", "qty"]
                
                # Get pipeline for category
                pipe_cat = None
                if cat_filter != "🌐 All" and not pipeline.empty:
                    pipe_cat = pipeline[pipeline["category"] == cat_filter]
                
                title = f"📊 {cat_filter if cat_filter != '🌐 All' else 'All Categories'} - Multi-Layer Forecast"
                fig = create_multi_layer_forecast_chart(hist_ops_df, fc_ops, pipe_cat, title)
                st.plotly_chart(fig, use_container_width=True)
                
                # Gap Analysis
                if pipe_cat is not None and not pipe_cat.empty:
                    st.markdown("---")
                    st.markdown("### 🎯 Forecast vs Pipeline Gap Analysis")
                    
                    # Merge forecast and pipeline
                    merged = fc_ops.merge(pipe_cat, on="ds", how="outer").fillna(0)
                    merged["gap"] = merged["qty_forecast"] - merged["pipeline_amount"]
                    merged["coverage"] = (merged["pipeline_amount"] / merged["qty_forecast"] * 100).fillna(0)
                    
                    gap_summary = merged[["ds", "qty_forecast", "pipeline_amount", "gap", "coverage"]]
                    gap_summary.columns = ["Month", "Forecast", "Pipeline", "Gap", "Coverage %"]
                    gap_summary["Forecast"] = gap_summary["Forecast"].apply(lambda x: f"{x:,.0f}")
                    gap_summary["Pipeline"] = gap_summary["Pipeline"].apply(lambda x: f"{x:,.0f}")
                    gap_summary["Gap"] = gap_summary["Gap"].apply(lambda x: f"{x:+,.0f}")
                    gap_summary["Coverage %"] = gap_summary["Coverage %"].apply(lambda x: f"{x:.1f}%")
                    
                    st.dataframe(gap_summary, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.markdown("### ⏱️ Lead Times")
            items_df = data["items"]
            if not items_df.empty:
                c_ven = resolve_col(items_df, "vendor")
                c_lt = resolve_col(items_df, "lead_time")
                if c_ven and c_lt:
                    items_df[c_lt] = items_df[c_lt].apply(_safe_float)
                    lt_df = items_df.groupby(c_ven)[c_lt].mean().reset_index()
                    
                    fig = go.Figure(go.Bar(
                        x=lt_df[c_ven], y=lt_df[c_lt],
                        marker=dict(color=lt_df[c_lt], colorscale='Viridis'),
                        text=lt_df[c_lt].round(0), textposition='outside'
                    ))
                    fig.update_layout(
                        title="Avg Lead Time by Vendor",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        yaxis_title="Days",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with col_r:
            st.markdown("### 🎨 Category Mix")
            cat_mix = dem.groupby("category")["qty"].sum().reset_index()
            
            fig = go.Figure(go.Pie(
                labels=cat_mix["category"], values=cat_mix["qty"], hole=0.5,
                marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']),
                textfont=dict(size=14, color='white')
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3: SCENARIO PLANNING
    # ═══════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("## 🔮 Scenario Planning")
        
        # Base forecast
        base_hist = dem.groupby("ds")["qty"].sum()
        base_fc = run_forecast(base_hist, cfg).reset_index()
        base_fc.columns = ["ds", "qty"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### ⚙️ Scenario Controls")
            
            scenario_name = st.text_input("📝 Scenario Name", "Baseline")
            growth = st.slider("📈 Growth Rate (%)", -50, 50, 0, 5) / 100
            demand_weight = st.slider("🎯 Demand Forecast Weight", 0.0, 1.0, 0.7, 0.1)
            sales_weight = 1.0 - demand_weight
            
            st.caption(f"Sales Forecast Weight: {sales_weight:.1f}")
        
        with col2:
            st.markdown("### 💾 Scenario Management")
            
            if st.button("💾 Save Scenario"):
                scenario = Scenario(
                    name=scenario_name,
                    growth_rate=growth,
                    demand_weight=demand_weight,
                    sales_weight=sales_weight,
                    created_at=datetime.now().isoformat(),
                    forecast_data=base_fc.to_dict()
                )
                save_scenario(scenario)
                st.success(f"✅ Saved: {scenario_name}")
            
            saved_scenarios = load_scenarios()
            if saved_scenarios:
                scenario_names = [s.name for s in saved_scenarios]
                load_scenario = st.selectbox("📂 Load Scenario", [""] + scenario_names)
                
                if load_scenario:
                    loaded = next(s for s in saved_scenarios if s.name == load_scenario)
                    st.info(f"Created: {loaded.created_at}")
                    st.json({
                        "Growth": f"{loaded.growth_rate*100:+.1f}%",
                        "Demand Weight": f"{loaded.demand_weight:.1%}",
                        "Sales Weight": f"{loaded.sales_weight:.1%}"
                    })
        
        # Apply scenario
        base_fc["adjusted"] = base_fc["qty"] * (1 + growth)
        
        # Blended forecast (if we had sales forecast, we'd blend here)
        base_fc["blended"] = base_fc["adjusted"] * demand_weight
        
        st.markdown("---")
        st.markdown("### 📊 Scenario Impact")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=base_fc["ds"], y=base_fc["qty"],
            name="Baseline", line=dict(dash="dash", color="rgba(255,255,255,0.5)", width=2)
        ))
        fig.add_trace(go.Scatter(
            x=base_fc["ds"], y=base_fc["adjusted"],
            name="Scenario", line=dict(color="#00f2fe", width=3),
            fill="tonexty", fillcolor="rgba(79, 172, 254, 0.2)"
        ))
        fig.update_layout(
            title="Scenario Impact Analysis",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        impact = base_fc["adjusted"].sum() - base_fc["qty"].sum()
        st.metric("💥 Total Impact", format_qty(impact), f"{impact/base_fc['qty'].sum()*100:+.1f}%")

if __name__ == "__main__":
    main()
