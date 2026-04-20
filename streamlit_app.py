"""
Streamlit Dashboard for Particle Filter Engine.
Displays forecasted returns and top ETF picks.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from huggingface_hub import HfApi, hf_hub_download
import json
import config
from us_calendar import USMarketCalendar

st.set_page_config(page_title="P2Quant Particle Filter", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .pick-card { background: #f8f9fa; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .return-positive { color: #28a745; font-weight: 600; }
    .return-negative { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_results():
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO, filename=json_files[0],
            repo_type="dataset", token=config.HF_TOKEN, cache_dir="./hf_cache"
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def return_badge(ret):
    if ret >= 0:
        return f'<span class="return-positive">+{ret*100:.2f}%</span>'
    return f'<span class="return-negative">{ret*100:.2f}%</span>'

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
calendar = USMarketCalendar()
st.sidebar.markdown(f"**📅 Next Trading Day:** {calendar.next_trading_day().strftime('%Y-%m-%d')}")
data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")

st.markdown('<div class="main-header">🎯 P2Quant Particle Filter</div>', unsafe_allow_html=True)
st.markdown('<div>Sequential Monte Carlo – Non‑Linear State Estimation & Next‑Day Return Forecasts</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available.")
    st.stop()

daily = data['daily_trading']
top_picks = daily['top_picks']

tabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, key in zip(tabs, universe_keys):
    with tab:
        picks = top_picks.get(key, [])
        if picks:
            st.markdown(f"### Top {len(picks)} Forecasted Returns")
            cols = st.columns(len(picks))
            for i, pick in enumerate(picks):
                with cols[i]:
                    st.markdown(f"""
                    <div class="pick-card">
                        <h3>#{i+1} {pick['ticker']}</h3>
                        <p style="font-size: 1.8rem;">{return_badge(pick['forecast_mean'])}</p>
                        <p>95% CI: {pick['forecast_lower']*100:.2f}% to {pick['forecast_upper']*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No forecasts available.")
