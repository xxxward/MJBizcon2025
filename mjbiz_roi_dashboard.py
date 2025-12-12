import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="MJBiz ROI Dashboard", layout="wide")

# Title
st.title("🎯 MJBiz ROI Dashboard")
st.markdown("### Week-over-Week Activity Tracking")

# Date range selector
col1, col2 = st.columns(2)
with col1:
    last_week_start = st.date_input("Last Week Start", datetime.now() - timedelta(days=14))
with col2:
    this_week_end = st.date_input("This Week End", datetime.now())

st.markdown("---")

# Data Input Section
st.sidebar.header("📊 Data Input Options")
input_method = st.sidebar.radio("Choose input method:", ["Manual Entry", "CSV Upload"])

if input_method == "Manual Entry":
    st.sidebar.markdown("### Enter Data Manually")
    
    # Companies Data
    with st.sidebar.expander("Companies Created", expanded=False):
        st.markdown("**Last Week:**")
        companies_lw_total = st.number_input("Total", min_value=0, value=0, key="comp_lw_total")
        
        st.markdown("**This Week:**")
        companies_tw_total = st.number_input("Total", min_value=0, value=0, key="comp_tw_total")
        
        st.markdown("**By Rep (comma-separated):**")
        st.caption("Format: Name:LastWeek:ThisWeek")
        companies_by_rep_input = st.text_area("Example: Brad Sherman:5:3, Jake Lynch:2:4", key="comp_reps")
    
    # Contacts Data
    with st.sidebar.expander("Contacts Created", expanded=False):
        st.markdown("**Last Week:**")
        contacts_lw_total = st.number_input("Total", min_value=0, value=0, key="cont_lw_total")
        
        st.markdown("**This Week:**")
        contacts_tw_total = st.number_input("Total", min_value=0, value=0, key="cont_tw_total")
        
        st.markdown("**By Rep (comma-separated):**")
        contacts_by_rep_input = st.text_area("Example: Brad Sherman:10:8, Jake Lynch:5:6", key="cont_reps")
    
    # Deals Data
    with st.sidebar.expander("Deals Created", expanded=False):
        st.markdown("**Last Week:**")
        deals_lw_count = st.number_input("Count", min_value=0, value=0, key="deals_lw_count")
        deals_lw_value = st.number_input("$ Value", min_value=0.0, value=0.0, key="deals_lw_value")
        
        st.markdown("**This Week:**")
        deals_tw_count = st.number_input("Count", min_value=0, value=0, key="deals_tw_count")
        deals_tw_value = st.number_input("$ Value", min_value=0.0, value=0.0, key="deals_tw_value")
        
        st.markdown("**By Rep (comma-separated):**")
        st.caption("Format: Name:LWCount:LWValue:TWCount:TWValue")
        deals_by_rep_input = st.text_area("Example: Brad Sherman:3:50000:2:30000", key="deals_reps")
    
    # Meetings Data
    with st.sidebar.expander("Meetings Logged", expanded=False):
        st.markdown("**Last Week:**")
        meetings_lw_total = st.number_input("Total", min_value=0, value=0, key="meet_lw_total")
        
        st.markdown("**By Rep (comma-separated):**")
        meetings_by_rep_input = st.text_area("Example: Brad Sherman:12, Jake Lynch:8", key="meet_reps")
    
    # Parse manual input
    def parse_rep_data(input_str, data_type):
        if not input_str.strip():
            return pd.DataFrame()
        
        rows = []
        for item in input_str.split(','):
            item = item.strip()
            if ':' in item:
                parts = item.split(':')
                if data_type == "companies" or data_type == "contacts":
                    if len(parts) == 3:
                        rows.append({
                            'Rep': parts[0].strip(),
                            'Last Week': int(parts[1]),
                            'This Week': int(parts[2])
                        })
                elif data_type == "deals":
                    if len(parts) == 5:
                        rows.append({
                            'Rep': parts[0].strip(),
                            'Last Week Count': int(parts[1]),
                            'Last Week Value': float(parts[2]),
                            'This Week Count': int(parts[3]),
                            'This Week Value': float(parts[4])
                        })
                elif data_type == "meetings":
                    if len(parts) == 2:
                        rows.append({
                            'Rep': parts[0].strip(),
                            'Count': int(parts[1])
                        })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
    companies_df = parse_rep_data(companies_by_rep_input, "companies")
    contacts_df = parse_rep_data(contacts_by_rep_input, "contacts")
    deals_df = parse_rep_data(deals_by_rep_input, "deals")
    meetings_df = parse_rep_data(meetings_by_rep_input, "meetings")

else:  # CSV Upload
    st.sidebar.markdown("### Upload CSV Files")
    st.sidebar.caption("Export your HubSpot reports as CSV and upload below")
    
    companies_file = st.sidebar.file_uploader("Companies CSV", type=['csv'])
    contacts_file = st.sidebar.file_uploader("Contacts CSV", type=['csv'])
    deals_file = st.sidebar.file_uploader("Deals CSV", type=['csv'])
    meetings_file = st.sidebar.file_uploader("Meetings CSV", type=['csv'])
    
    # Initialize raw dataframes
    companies_raw = pd.read_csv(companies_file) if companies_file else pd.DataFrame()
    contacts_raw = pd.read_csv(contacts_file) if contacts_file else pd.DataFrame()
    deals_raw = pd.read_csv(deals_file) if deals_file else pd.DataFrame()
    meetings_raw = pd.read_csv(meetings_file) if meetings_file else pd.DataFrame()
    
    # Process Deals CSV (HubSpot format)
    if not deals_raw.empty:
        st.sidebar.markdown("#### Deals Configuration")
        
        # Identify the owner column
        owner_col = None
        for col in ['Deal owner', 'Deal Owner', 'Owner', 'HubSpot Owner']:
            if col in deals_raw.columns:
                owner_col = col
                break
        
        if owner_col:
            # Group by owner
            deals_by_owner = deals_raw.groupby(owner_col).size().reset_index(name='Count')
            
            # Manual split input
            st.sidebar.caption("Specify Last Week vs This Week split:")
            lw_count_input = st.sidebar.number_input("Last Week Deal Count", min_value=0, max_value=len(deals_raw), value=0, key="deals_lw_split")
            tw_count_input = st.sidebar.number_input("This Week Deal Count", min_value=0, max_value=len(deals_raw), value=len(deals_raw), key="deals_tw_split")
            
            # Deal values
            st.sidebar.caption("Deal Values (Total):")
            deals_lw_value = st.sidebar.number_input("Last Week Total $", min_value=0.0, value=0.0, step=1000.0, key="deals_lw_value_input")
            deals_tw_value = st.sidebar.number_input("This Week Total $", min_value=0.0, value=0.0, step=1000.0, key="deals_tw_value_input")
            
            deals_lw_count = lw_count_input
            deals_tw_count = tw_count_input
            
            # Create breakdown by rep
            deals_df = deals_by_owner.rename(columns={owner_col: 'Rep'})
            deals_df['Last Week Count'] = 0  # Would need separate exports for accurate split
            deals_df['This Week Count'] = deals_df['Count']
            deals_df['Last Week Value'] = 0.0
            deals_df['This Week Value'] = 0.0
        else:
            deals_lw_count = 0
            deals_tw_count = len(deals_raw)
            deals_lw_value = 0.0
            deals_tw_value = st.sidebar.number_input("Total Deal Value $", min_value=0.0, value=0.0, step=1000.0)
            deals_df = pd.DataFrame()
    else:
        deals_lw_count = 0
        deals_tw_count = 0
        deals_lw_value = 0.0
        deals_tw_value = 0.0
        deals_df = pd.DataFrame()
    
    # Process other CSVs similarly
    if not companies_raw.empty:
        owner_col = None
        for col in ['HubSpot Owner', 'Owner', 'Company owner']:
            if col in companies_raw.columns:
                owner_col = col
                break
        
        if owner_col:
            companies_by_owner = companies_raw.groupby(owner_col).size().reset_index(name='Total')
            companies_df = companies_by_owner.rename(columns={owner_col: 'Rep'})
            companies_df['Last Week'] = 0
            companies_df['This Week'] = companies_df['Total']
        else:
            companies_df = pd.DataFrame()
        
        companies_lw_total = st.sidebar.number_input("Companies - Last Week", min_value=0, value=0, key="comp_lw_csv")
        companies_tw_total = len(companies_raw) - companies_lw_total
    else:
        companies_df = pd.DataFrame()
        companies_lw_total = 0
        companies_tw_total = 0
    
    if not contacts_raw.empty:
        owner_col = None
        for col in ['HubSpot Owner', 'Owner', 'Contact owner']:
            if col in contacts_raw.columns:
                owner_col = col
                break
        
        if owner_col:
            contacts_by_owner = contacts_raw.groupby(owner_col).size().reset_index(name='Total')
            contacts_df = contacts_by_owner.rename(columns={owner_col: 'Rep'})
            contacts_df['Last Week'] = 0
            contacts_df['This Week'] = contacts_df['Total']
        else:
            contacts_df = pd.DataFrame()
        
        contacts_lw_total = st.sidebar.number_input("Contacts - Last Week", min_value=0, value=0, key="cont_lw_csv")
        contacts_tw_total = len(contacts_raw) - contacts_lw_total
    else:
        contacts_df = pd.DataFrame()
        contacts_lw_total = 0
        contacts_tw_total = 0
    
    if not meetings_raw.empty:
        owner_col = None
        for col in ['Meeting owner', 'Owner', 'Sales Activity Owner Name']:
            if col in meetings_raw.columns:
                owner_col = col
                break
        
        if owner_col:
            meetings_by_owner = meetings_raw.groupby(owner_col).size().reset_index(name='Count')
            meetings_df = meetings_by_owner.rename(columns={owner_col: 'Rep'})
        else:
            meetings_df = pd.DataFrame()
        
        meetings_lw_total = len(meetings_raw)
    else:
        meetings_df = pd.DataFrame()
        meetings_lw_total = 0

# Main Dashboard Layout
st.markdown("## 📊 Key Metrics Overview")

# Top-level metrics
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        "New Companies",
        f"{companies_lw_total + companies_tw_total}",
        f"+{companies_tw_total - companies_lw_total} vs Last Week"
    )

with metric_col2:
    st.metric(
        "New Contacts",
        f"{contacts_lw_total + contacts_tw_total}",
        f"+{contacts_tw_total - contacts_lw_total} vs Last Week"
    )

with metric_col3:
    st.metric(
        "New Deals",
        f"{deals_lw_count + deals_tw_count}",
        f"+{deals_tw_count - deals_lw_count} vs Last Week"
    )

with metric_col4:
    st.metric(
        "Total Deal Value",
        f"${(deals_lw_value + deals_tw_value):,.0f}",
        f"+${(deals_tw_value - deals_lw_value):,.0f} vs Last Week"
    )

st.markdown("---")

# Detailed Breakdown
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🏢 Companies Created")
    
    comp_data = {
        'Period': ['Last Week', 'This Week'],
        'Count': [companies_lw_total, companies_tw_total]
    }
    comp_summary = pd.DataFrame(comp_data)
    
    fig_comp = px.bar(comp_summary, x='Period', y='Count', 
                      text='Count', color='Period',
                      color_discrete_map={'Last Week': '#3498db', 'This Week': '#2ecc71'})
    fig_comp.update_traces(textposition='outside')
    fig_comp.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_comp, use_container_width=True)
    
    if not companies_df.empty:
        st.markdown("**By Rep:**")
        st.dataframe(companies_df, use_container_width=True, hide_index=True)

with col_right:
    st.markdown("### 👥 Contacts Created")
    
    cont_data = {
        'Period': ['Last Week', 'This Week'],
        'Count': [contacts_lw_total, contacts_tw_total]
    }
    cont_summary = pd.DataFrame(cont_data)
    
    fig_cont = px.bar(cont_summary, x='Period', y='Count',
                      text='Count', color='Period',
                      color_discrete_map={'Last Week': '#3498db', 'This Week': '#2ecc71'})
    fig_cont.update_traces(textposition='outside')
    fig_cont.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_cont, use_container_width=True)
    
    if not contacts_df.empty:
        st.markdown("**By Rep:**")
        st.dataframe(contacts_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Deals Section
st.markdown("### 💰 Deals Created")

deal_col1, deal_col2 = st.columns(2)

with deal_col1:
    st.markdown("#### Deal Count")
    deal_count_data = {
        'Period': ['Last Week', 'This Week'],
        'Count': [deals_lw_count, deals_tw_count]
    }
    deal_count_df = pd.DataFrame(deal_count_data)
    
    fig_deal_count = px.bar(deal_count_df, x='Period', y='Count',
                            text='Count', color='Period',
                            color_discrete_map={'Last Week': '#e74c3c', 'This Week': '#f39c12'})
    fig_deal_count.update_traces(textposition='outside')
    fig_deal_count.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_deal_count, use_container_width=True)

with deal_col2:
    st.markdown("#### Deal Value")
    deal_value_data = {
        'Period': ['Last Week', 'This Week'],
        'Value': [deals_lw_value, deals_tw_value]
    }
    deal_value_df = pd.DataFrame(deal_value_data)
    
    fig_deal_value = px.bar(deal_value_df, x='Period', y='Value',
                            text='Value', color='Period',
                            color_discrete_map={'Last Week': '#e74c3c', 'This Week': '#f39c12'})
    fig_deal_value.update_traces(textposition='outside', texttemplate='$%{text:,.0f}')
    fig_deal_value.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_deal_value, use_container_width=True)

if not deals_df.empty:
    st.markdown("**By Rep:**")
    st.dataframe(deals_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Meetings Section
st.markdown("### 📅 Meetings Logged (Last Week)")

col_meet1, col_meet2 = st.columns([2, 1])

with col_meet1:
    if not meetings_df.empty:
        fig_meetings = px.bar(meetings_df, x='Rep', y='Count',
                             text='Count', color='Count',
                             color_continuous_scale='Blues')
        fig_meetings.update_traces(textposition='outside')
        fig_meetings.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_meetings, use_container_width=True)
    else:
        st.info("No meeting data available")

with col_meet2:
    st.metric("Total Meetings", f"{meetings_lw_total}")
    
    if not meetings_df.empty:
        st.markdown("**Breakdown:**")
        st.dataframe(meetings_df, use_container_width=True, hide_index=True)

st.markdown("---")

# Summary Section
st.markdown("## 📋 MJBiz ROI Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

with summary_col1:
    st.markdown("### Total New Records")
    st.write(f"**Companies:** {companies_lw_total + companies_tw_total}")
    st.write(f"**Contacts:** {contacts_lw_total + contacts_tw_total}")
    st.write(f"**Deals:** {deals_lw_count + deals_tw_count}")

with summary_col2:
    st.markdown("### Pipeline Value")
    st.write(f"**Total:** ${(deals_lw_value + deals_tw_value):,.0f}")
    st.write(f"**Avg Deal:** ${((deals_lw_value + deals_tw_value) / (deals_lw_count + deals_tw_count)):,.0f}" if (deals_lw_count + deals_tw_count) > 0 else "**Avg Deal:** $0")

with summary_col3:
    st.markdown("### Activity Metrics")
    st.write(f"**Meetings:** {meetings_lw_total}")
    st.write(f"**Meetings/Deal:** {meetings_lw_total / (deals_lw_count + deals_tw_count):.1f}" if (deals_lw_count + deals_tw_count) > 0 else "**Meetings/Deal:** 0")

# Export option
st.markdown("---")
if st.button("📥 Export Summary as CSV"):
    summary_data = {
        'Metric': [
            'Companies - Last Week', 'Companies - This Week', 'Companies - Total',
            'Contacts - Last Week', 'Contacts - This Week', 'Contacts - Total',
            'Deals - Last Week Count', 'Deals - This Week Count', 'Deals - Total Count',
            'Deals - Last Week Value', 'Deals - This Week Value', 'Deals - Total Value',
            'Meetings - Last Week'
        ],
        'Value': [
            companies_lw_total, companies_tw_total, companies_lw_total + companies_tw_total,
            contacts_lw_total, contacts_tw_total, contacts_lw_total + contacts_tw_total,
            deals_lw_count, deals_tw_count, deals_lw_count + deals_tw_count,
            f"${deals_lw_value:,.0f}", f"${deals_tw_value:,.0f}", f"${(deals_lw_value + deals_tw_value):,.0f}",
            meetings_lw_total
        ]
    }
    summary_export = pd.DataFrame(summary_data)
    csv = summary_export.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"mjbiz_roi_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        mime='text/csv'
    )
