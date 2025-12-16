"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CALYX - HOLISTIC SALES PLANNING & FORECASTING DASHBOARD
    Sources: 
    1. SO & Invoice Data (NetSuite) - Historical & Active
    2. Deals Data (HubSpot) - Pipeline Forecast
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & STYLING
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Calyx Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for metrics scaling and layout
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main {
        background-color: #f8f9fa;
    }
    
    /* CUSTOM METRIC CONTAINERS */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e9ecef;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: transform 0.2s;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1);
        border-color: #ced4da;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6c757d;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem; 
        font-weight: 800;
        color: #212529;
        /* Ensure text doesn't overflow */
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.2;
    }
    
    /* Make the value smaller if it's very long using container queries concept (approximated) */
    @media (max-width: 1400px) { .metric-value { font-size: 1.75rem; } }
    
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    h1, h2, h3 { color: #1a1a1a; }
    
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] p {
        color: #e2e8f0;
    }
    
</style>
""", unsafe_allow_html=True)

# Sheet Names
SHEET_SO_INV = "SO & invoice Data merged"
SHEET_DEALS = "Deals"

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def load_data():
    """Load data from both sheets"""
    try:
        from google.oauth2.service_account import Credentials
        import gspread

        # Credentials setup
        creds_dict = None
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
        elif "service_account" in st.secrets:
            creds_dict = dict(st.secrets["service_account"])
        else:
            return None, None
            
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
        client = gspread.authorize(creds)
        
        # Open Spreadsheet
        sheet_id = st.secrets.get("SPREADSHEET_ID") or st.secrets["gsheets"]["spreadsheet_id"]
        sh = client.open_by_key(sheet_id)
        
        # 1. LOAD SO & INVOICE DATA
        ws_so = sh.worksheet(SHEET_SO_INV)
        rows_so = ws_so.get_all_values()
        df_so = pd.DataFrame()
        if len(rows_so) > 1:
            headers = rows_so[0]
            # Deduplicate headers
            deduped = []
            seen = {}
            for h in headers:
                h = h.strip()
                if h in seen:
                    seen[h] += 1
                    deduped.append(f"{h}_{seen[h]}")
                else:
                    seen[h] = 0
                    deduped.append(h)
            df_so = pd.DataFrame(rows_so[1:], columns=deduped)

        # 2. LOAD DEALS DATA (Headers on Row 2)
        ws_deals = sh.worksheet(SHEET_DEALS)
        rows_deals = ws_deals.get_all_values()
        df_deals = pd.DataFrame()
        if len(rows_deals) > 1:
            # Row 2 is index 1
            headers = rows_deals[1]
            # Data starts at Row 3 (index 2)
            data_rows = rows_deals[2:]
            
            # Deduplicate headers for deals too
            deduped_deals = []
            seen_deals = {}
            for h in headers:
                h = h.strip()
                if h in seen_deals:
                    seen_deals[h] += 1
                    deduped_deals.append(f"{h}_{seen_deals[h]}")
                else:
                    seen_deals[h] = 0
                    deduped_deals.append(h)
            
            df_deals = pd.DataFrame(data_rows, columns=deduped_deals)

        return df_so, df_deals

    except Exception as e:
        st.error(f"Data Connection Error: {e}")
        return pd.DataFrame(), pd.DataFrame()

# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_so_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prep the NetSuite SO/Invoice Data"""
    if df.empty: return df
    
    df = df.copy()
    
    # 1. Combine Rep & Customer
    df['sales_rep_master'] = df['Inv - Rep Master'].combine_first(df['SO - Rep Master'])
    df['customer_corrected'] = df['Inv - Correct Customer'].combine_first(df['SO - Customer Companyname'])
    
    # 2. Numeric Cleaning
    cols_to_num = ['so_amount', 'actual_revenue_billed', 'inv_amount']
    # Map original names if they exist
    mappings = {
        'SO - Amount': 'so_amount',
        'Inv - Amount': 'inv_amount',
        'SO - Quantity Ordered': 'so_qty',
        'SO - Item Rate': 'item_rate'
    }
    
    for old, new in mappings.items():
        if old in df.columns:
            df[new] = pd.to_numeric(df[old].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
            
    # 3. Date Parsing
    date_cols = {
        'SO - Date Created': 'so_date',
        'SO - Pending Fulfillment Date': 'fulfillment_date',
        'Inv - Date': 'inv_date'
    }
    for old, new in date_cols.items():
        if old in df.columns:
            df[new] = pd.to_datetime(df[old], errors='coerce')
            
    # 4. Aggregation for Invoiced Amount
    # (Simplified aggregation logic for this view)
    if 'actual_revenue_billed' not in df.columns:
        # If pre-calculated column doesn't exist, roughly estimate from Inv Amount
        df['actual_revenue_billed'] = df['inv_amount'] 

    # 5. SO Status Logic
    if 'SO - Status' in df.columns:
        df['status_clean'] = df['SO - Status'].str.strip()
    else:
        df['status_clean'] = 'Unknown'
        
    # 6. Filter Cancelled
    df = df[df['status_clean'] != 'Cancelled']
    
    return df

def prepare_deals_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prep the HubSpot Deals Data"""
    if df.empty: return df
    
    df = df.copy()
    
    # 1. Filter: Include? = TRUE
    if 'Include?' in df.columns:
        # Normalize to string, upper, check for TRUE
        df = df[df['Include?'].astype(str).str.upper().isin(['TRUE', 'YES', '1'])]
    
    # 2. Construct Sales Rep Name
    first = df['Deal Owner First Name'].astype(str).replace('None', '')
    last = df['Deal Owner Last Name'].astype(str).replace('None', '')
    df['sales_rep_combined'] = (first + " " + last).str.strip()
    
    # 3. Parse Dates & Filter Timeframe (Oct 2024 - Dec 2025)
    if 'Close Date' in df.columns:
        df['close_date_dt'] = pd.to_datetime(df['Close Date'], errors='coerce')
        
        start_date = pd.Timestamp('2024-10-01')
        end_date = pd.Timestamp('2025-12-31')
        
        df = df[
            (df['close_date_dt'] >= start_date) & 
            (df['close_date_dt'] <= end_date)
        ]
    
    # 4. Numeric Amount
    if 'Amount' in df.columns:
        df['deal_amount'] = pd.to_numeric(df['Amount'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce').fillna(0)
    
    # 5. Pipeline & Stage
    if 'Pipeline' in df.columns:
        df['pipeline_clean'] = df['Pipeline'].str.strip()
        
    return df

# ══════════════════════════════════════════════════════════════════════════════
# UNIFICATION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def create_unified_forecast(df_so, df_deals):
    """
    Creates a single timeline dataframe combining:
    1. Historical Invoiced (inv_date)
    2. Active/Open SOs (fulfillment_date)
    3. Pipeline Deals (close_date)
    """
    unified_rows = []
    
    # --- PROCESS SO DATA ---
    for _, row in df_so.iterrows():
        # Historical / Invoiced
        if row['actual_revenue_billed'] > 0 and pd.notnull(row.get('inv_date')):
            unified_rows.append({
                'Date': row['inv_date'],
                'Amount': row['actual_revenue_billed'],
                'Type': 'Actual (Invoiced)',
                'Rep': row.get('sales_rep_master'),
                'Customer': row.get('customer_corrected'),
                'Item': row.get('SO - Item', 'Unknown'),
                'Status': 'Invoiced'
            })
            
        # Active / Open (Remaining)
        remaining = row['so_amount'] - row['actual_revenue_billed']
        if remaining > 0 and pd.notnull(row.get('fulfillment_date')):
            unified_rows.append({
                'Date': row['fulfillment_date'],
                'Amount': remaining,
                'Type': 'Active Order (Backlog)',
                'Rep': row.get('sales_rep_master'),
                'Customer': row.get('customer_corrected'),
                'Item': row.get('SO - Item', 'Unknown'),
                'Status': row.get('status_clean', 'Open')
            })
            
    # --- PROCESS DEALS DATA ---
    for _, row in df_deals.iterrows():
        if pd.notnull(row.get('close_date_dt')) and row['deal_amount'] > 0:
            unified_rows.append({
                'Date': row['close_date_dt'],
                'Amount': row['deal_amount'],
                'Type': 'Pipeline Forecast',
                'Rep': row.get('sales_rep_combined'),
                'Customer': None, # Deals tab doesn't have reliable customer mapping yet
                'Item': row.get('SKU / Item', 'Unknown'),
                'Status': row.get('Deal Stage', 'Pipeline')
            })
            
    return pd.DataFrame(unified_rows)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    
    # 1. LOAD
    with st.spinner("Connecting to Data Sources..."):
        raw_so, raw_deals = load_data()
        
    if raw_so.empty and raw_deals.empty:
        st.error("No data loaded. Check credentials and sheet structure.")
        st.stop()
        
    # 2. PREP
    df_so = prepare_so_data(raw_so)
    df_deals = prepare_deals_data(raw_deals)
    
    # 3. UNIFY
    df_unified = create_unified_forecast(df_so, df_deals)
    
    if df_unified.empty:
        st.warning("Data loaded but generated no forecast rows. Check filters (dates/status).")
        st.stop()
        
    df_unified['Month'] = df_unified['Date'].dt.to_period('M').dt.to_timestamp()

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR FILTERS
    # ══════════════════════════════════════════════════════════════════════════
    
    with st.sidebar:
        st.header("🔍 Filters")
        
        # A. Sales Rep (Union of both sources)
        reps_so = df_so['sales_rep_master'].dropna().unique().tolist()
        reps_deals = df_deals['sales_rep_combined'].dropna().unique().tolist()
        all_reps = sorted(list(set(reps_so + reps_deals)))
        
        sel_rep = st.selectbox("Sales Rep", ["All"] + all_reps)
        
        # B. Customer (SO Only primarily)
        # Note: selecting a customer will HIDE deals data usually since deals lack customer mapping
        customers = sorted(df_so['customer_corrected'].dropna().unique().tolist())
        sel_cust = st.selectbox("Customer (Active/History Only)", ["All"] + customers)
        
        # C. Item
        items = sorted(df_unified['Item'].dropna().astype(str).unique().tolist())
        sel_item = st.selectbox("Item / SKU", ["All"] + items)
        
        # APPLY FILTERS
        # 1. Filter Unified Data
        filtered_unified = df_unified.copy()
        
        if sel_rep != "All":
            filtered_unified = filtered_unified[filtered_unified['Rep'] == sel_rep]
        
        if sel_cust != "All":
            # Keep rows where Customer matches OR Type is Pipeline (since pipeline has no customer)
            # OR, strictly filter out pipeline if customer is selected (safer for specific account planning)
            filtered_unified = filtered_unified[
                (filtered_unified['Customer'] == sel_cust)
                # Uncomment line below to keep pipeline visible even when customer is filtered (risky if unrelated)
                # | (filtered_unified['Type'] == 'Pipeline Forecast') 
            ]
            
        if sel_item != "All":
            filtered_unified = filtered_unified[filtered_unified['Item'] == sel_item]
            
    # ══════════════════════════════════════════════════════════════════════════
    # DASHBOARD HEADER & METRICS
    # ══════════════════════════════════════════════════════════════════════════
    
    st.title("📊 Calyx Holistic Forecasting")
    st.markdown(f"**View:** {sel_rep} | **Customer:** {sel_cust}")
    
    # Calculate Metrics from Filtered Data
    # 1. Actual (Invoiced)
    metric_actual = filtered_unified[filtered_unified['Type'] == 'Actual (Invoiced)']['Amount'].sum()
    
    # 2. Active (Backlog)
    metric_active = filtered_unified[filtered_unified['Type'] == 'Active Order (Backlog)']['Amount'].sum()
    
    # 3. Pipeline
    metric_pipeline = filtered_unified[filtered_unified['Type'] == 'Pipeline Forecast']['Amount'].sum()
    
    # 4. Total Forecast (Active + Pipeline)
    metric_forecast_total = metric_active + metric_pipeline

    # Custom HTML Metrics
    def metric_card(label, value, color="#212529"):
        return f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color: {color}">${value:,.0f}</div>
        </div>
        """

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(metric_card("Actual Invoiced", metric_actual, "#10b981"), unsafe_allow_html=True) # Green
    with c2: st.markdown(metric_card("Active (Backlog)", metric_active, "#3b82f6"), unsafe_allow_html=True) # Blue
    with c3: st.markdown(metric_card("Pipeline Forecast", metric_pipeline, "#f59e0b"), unsafe_allow_html=True) # Amber
    with c4: st.markdown(metric_card("Total Forecast (Active+Pipe)", metric_forecast_total, "#6366f1"), unsafe_allow_html=True) # Indigo

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # VISUALIZATION ROW 1: HOLISTIC FORECAST
    # ══════════════════════════════════════════════════════════════════════════
    
    st.subheader("📈 Revenue Forecast (Historical + Active + Pipeline)")
    
    if not filtered_unified.empty:
        # Group by Month and Type
        chart_data = filtered_unified.groupby(['Month', 'Type'])['Amount'].sum().reset_index()
        
        # Sort by Month
        chart_data = chart_data.sort_values('Month')
        
        fig_main = px.bar(
            chart_data,
            x='Month',
            y='Amount',
            color='Type',
            title='Unified Revenue Timeline',
            color_discrete_map={
                'Actual (Invoiced)': '#10b981',
                'Active Order (Backlog)': '#3b82f6',
                'Pipeline Forecast': '#f59e0b'
            },
            template='plotly_white'
        )
        
        fig_main.update_layout(
            barmode='stack',
            height=500,
            xaxis=dict(tickformat="%b %Y"),
            legend=dict(orientation="h", y=1.02, yanchor="bottom")
        )
        
        st.plotly_chart(fig_main, use_container_width=True)
    else:
        st.info("No data available for the selected filters.")

    # ══════════════════════════════════════════════════════════════════════════
    # VISUALIZATION ROW 2: DRILL DOWNS
    # ══════════════════════════════════════════════════════════════════════════
    
    col_left, col_right = st.columns(2)
    
    # --- SALES ORDER STATUS BREAKDOWN (Left) ---
    with col_left:
        st.subheader("📋 Sales Order Status Breakdown")
        # Filter for SO data only within unified set
        so_view = filtered_unified[filtered_unified['Type'] == 'Active Order (Backlog)']
        
        if not so_view.empty:
            status_counts = so_view.groupby('Status')['Amount'].sum().reset_index()
            fig_pie = px.pie(
                status_counts, 
                values='Amount', 
                names='Status',
                color_discrete_sequence=px.colors.sequential.Blues,
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption("No active sales orders to display.")

    # --- PIPELINE DEAL STAGES (Right) ---
    with col_right:
        st.subheader("🎯 Pipeline by Stage")
        pipe_view = filtered_unified[filtered_unified['Type'] == 'Pipeline Forecast']
        
        if not pipe_view.empty:
            stage_counts = pipe_view.groupby('Status')['Amount'].sum().reset_index().sort_values('Amount', ascending=True)
            fig_bar = px.bar(
                stage_counts,
                x='Amount',
                y='Status',
                orientation='h',
                color_discrete_sequence=['#f59e0b']
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.caption("No pipeline deals to display.")

    # ══════════════════════════════════════════════════════════════════════════
    # DATA GRID
    # ══════════════════════════════════════════════════════════════════════════
    
    st.markdown("---")
    st.subheader("🔎 Detailed Data View")
    
    tab1, tab2 = st.tabs(["Combined Forecast Data", "Raw Pipeline Deals"])
    
    with tab1:
        st.dataframe(
            filtered_unified[['Date', 'Rep', 'Customer', 'Item', 'Status', 'Type', 'Amount']].sort_values('Date'),
            use_container_width=True,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="MMM YYYY"),
                "Amount": st.column_config.NumberColumn("Revenue", format="$%d")
            }
        )
        
    with tab2:
        # Show raw deals matching the Rep filter
        # We need to re-filter df_deals directly because unified doesn't have all columns
        if sel_rep != "All":
            raw_deals_view = df_deals[df_deals['sales_rep_combined'] == sel_rep]
        else:
            raw_deals_view = df_deals
            
        st.dataframe(
            raw_deals_view,
            use_container_width=True
        )

if __name__ == "__main__":
    main()
