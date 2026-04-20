"""
Configuration for P2-ETF-PARTICLE-FILTER engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"

HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-particle-filter-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Particle Filter Parameters ---
NUM_PARTICLES = 5000                  # Number of particles
STATE_NOISE_TYPE = "t"                # "gaussian" or "t" (heavy-tailed)
STATE_NOISE_SCALE = 0.01              # Daily vol of latent state
OBS_NOISE_SCALE = 0.1                 # Observation noise scale (if Gaussian)
OBS_DF = 3                            # Degrees of freedom for t-distributed obs noise
RESAMPLE_THRESHOLD = 0.5              # Effective sample size threshold for resampling
LOOKBACK_WINDOW = 252                 # Days of data to use for filter initialization
MIN_OBSERVATIONS = 100

# --- Forecast & Ranking ---
FORECAST_HORIZON = 1                  # Predict next day
TOP_N_DISPLAY = 3                     # Show top 3 ETFs per universe

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2010, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
