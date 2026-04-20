"""
Streamlit Dashboard for Particle Filter Engine.
Displays forecasted returns and top ETF picks with standard two‑tab layout.
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
    .main-header { font-size: 2.5rem; font-weight: 600; color: #1f77b4; margin-bottom: 0.5rem; }
    .hero-card { background: linear-gradient(135deg, #1f77b4 0%, #2C5282 100%); border-radius: 16px; padding: 2rem; color: white; text-align: center; }
    .hero-ticker { font-size: 4rem; font-weight: 800; }
    .hero-return { font-size: 2rem; font-weight: 600; }
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

def display_hero_card(ticker: str, forecast_mean: float, forecast_lower: float, forecast_upper: float):
    st.markdown(f"""
    <div class="hero-card">
        <div style="font-size: 1.2rem; opacity: 0.8;">🎯 TOP PICK FOR TOMORROW</div>
        <div class="hero-ticker">{ticker}</div>
        <div class="hero-return">{return_badge(forecast_mean)}</div>
        <div style="margin-top: 1rem;">95% CI: {forecast_lower*100:.2f}% to {forecast_upper*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

def display_forecast_table(universe_data: dict):
    rows = []
    for ticker, data in universe_data.items():
        if data.get('forecast_mean') is not None:
            rows.append({
                'Ticker': ticker,
                'Forecast': f"{data['forecast_mean']*100:.3f}%",
                'Lower 95%': f"{data['forecast_lower']*100:.3f}%",
                'Upper 95%': f"{data['forecast_upper']*100:.3f}%"
            })
    if rows:
        df = pd.DataFrame(rows).sort_values('Forecast', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No forecast data available.")

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
st.sidebar.markdown(f"**Results Repo:** `{config.HF_OUTPUT_REPO}`")
st.sidebar.divider()

calendar = USMarketCalendar()
next_trading = calendar.next_trading_day()
st.sidebar.markdown(f"**📅 Next Trading Day:** {next_trading.strftime('%Y-%m-%d')}")

st.sidebar.divider()
st.sidebar.markdown("### 🎯 Particle Filter Parameters")
st.sidebar.markdown(f"- Particles: **{config.NUM_PARTICLES}**")
st.sidebar.markdown(f"- State Noise: **{config.STATE_NOISE_TYPE}** (scale={config.STATE_NOISE_SCALE})")
st.sidebar.markdown(f"- Obs Noise: scale={config.OBS_NOISE_SCALE}, df={config.OBS_DF}")
st.sidebar.markdown(f"- Lookback: **{config.LOOKBACK_WINDOW} days**")
st.sidebar.divider()

data = load_latest_results()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
else:
    st.sidebar.markdown("*No data available*")

# --- Main Content ---
st.markdown('<div class="main-header">🎯 P2Quant Particle Filter</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sequential Monte Carlo – Non‑Linear State Estimation & Next‑Day Return Forecasts</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available. Please run the daily pipeline first.")
    st.stop()

daily = data['daily_trading']
shrinking = data.get('shrinking_windows', {})

# --- Top‑Level Tabs ---
tab1, tab2 = st.tabs(["📋 Daily Trading", "📆 Shrinking Windows"])

# ------------------------------
# DAILY TRADING TAB
# ------------------------------
with tab1:
    top_picks = daily['top_picks']
    universes_data = daily['universes']
    
    subtabs = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]
    
    for subtab, key in zip(subtabs, universe_keys):
        with subtab:
            if key in universes_data:
                universe_dict = universes_data[key]
                picks = top_picks.get(key, [])
                
                if picks:
                    # Hero card for top pick
                    top = picks[0]
                    st.markdown("### 🏆 Top Pick for Tomorrow")
                    display_hero_card(top['ticker'], top['forecast_mean'], top['forecast_lower'], top['forecast_upper'])
                
                st.markdown("### 📋 All Forecasts")
                display_forecast_table(universe_dict)
                
                # Bar chart of forecasts
                forecasts = {t: d['forecast_mean'] for t, d in universe_dict.items() if d.get('forecast_mean') is not None}
                if forecasts:
                    sorted_items = sorted(forecasts.items(), key=lambda x: x[1], reverse=True)
                    tickers = [item[0] for item in sorted_items]
                    values = [item[1] for item in sorted_items]
                    colors = ['#1f77b4' if t == picks[0]['ticker'] else '#a0aec0' for t in tickers]
                    fig = go.Figure(go.Bar(x=tickers, y=values, marker_color=colors))
                    fig.update_layout(
                        title="Forecasted Next‑Day Returns",
                        xaxis_title="ETF Ticker",
                        yaxis_title="Forecast Return",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"bar_{key}")
            else:
                st.info(f"No data for {key} universe.")

# ------------------------------
# SHRINKING WINDOWS TAB
# ------------------------------
with tab2:
    if not shrinking:
        st.warning("No shrinking windows data available yet. The next run will generate it.")
        st.stop()
    
    st.markdown("### Top Picks Across Historical Windows")
    subtabs_sw = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
    
    for subtab, key in zip(subtabs_sw, universe_keys):
        with subtab:
            rows = []
            windows_sorted = sorted(shrinking.items(), key=lambda x: x[1]['start_year'], reverse=True)
            for label, winfo in windows_sorted:
                top = winfo['top_picks'].get(key, {})
                if top:
                    rows.append({
                        'Window': label,
                        'Top Pick': top.get('ticker', 'N/A'),
                        'Forecast': f"{top.get('forecast_mean', 0)*100:.3f}%"
                    })
            if rows:
                df_win = pd.DataFrame(rows)
                st.dataframe(df_win, use_container_width=True, hide_index=True)
                
                df_chart = df_win.copy()
                df_chart['Forecast_val'] = df_chart['Forecast'].str.rstrip('%').astype(float)
                fig = go.Figure(go.Scatter(
                    x=df_chart['Window'], y=df_chart['Forecast_val'],
                    mode='lines+markers', text=df_chart['Top Pick'],
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.update_layout(
                    title=f"{key} – Top Pick Forecast by Window",
                    xaxis_title="Window Start Year",
                    yaxis_title="Forecast Return (%)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True, key=f"sw_chart_{key}")
            else:
                st.info(f"No shrinking window data for {key}.")
