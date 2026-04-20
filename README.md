# P2-ETF-PARTICLE-FILTER

**Sequential Monte Carlo / Particle Filter for Robust State Estimation and ETF Return Forecasting**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-PARTICLE-FILTER/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-PARTICLE-FILTER/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--particle--filter--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-particle-filter-results)

## Overview

`P2-ETF-PARTICLE-FILTER` uses **Sequential Monte Carlo (SMC) / Particle Filtering** to estimate the latent "true expected return" of each ETF. Unlike Kalman filters or linear state‑space models, particle filters handle **non‑Gaussian noise, heavy tails, and sudden jumps**—making them robust to market shocks and regime shifts.

The engine outputs **next‑day return forecasts** with full posterior distributions (95% credible intervals) and ranks ETFs by forecasted return within each universe.

## Universe Coverage

| Universe | Tickers |
|----------|---------|
| **FI / Commodities** | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| **Equity Sectors** | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM |
| **Combined** | All tickers above |

Data is sourced from: [`P2SAMAPA/fi-etf-macro-signal-master-data`](https://huggingface.co/datasets/P2SAMAPA/fi-etf-macro-signal-master-data)

## Methodology

### State‑Space Model

The log‑return of each ETF is modeled as a noisy observation of a latent state:
State: x_t = x_{t-1} + v_t (v_t ~ Student‑t or Gaussian)
Observation: y_t = x_t + w_t (w_t ~ Student‑t or Gaussian)

text

- **Heavy‑tailed state noise** captures sudden jumps in expected returns.
- **Student‑t observation noise** provides robustness against outlier daily returns.

### Particle Filtering (SMC)

1. **Initialization**: 5,000 particles drawn from the initial state distribution.
2. **Prediction**: Particles propagated through the state transition.
3. **Update**: Particle weights updated via observation likelihood.
4. **Resampling**: Systematic resampling when effective sample size falls below 50%.
5. **Forecast**: Posterior predictive samples generate next‑day return forecasts with credible intervals.

### Ranking

ETFs are ranked within each universe by **forecasted mean return**. The top 3 are displayed on the dashboard.

## File Structure
P2-ETF-PARTICLE-FILTER/
├── config.py # Paths, universes, particle filter parameters
├── data_manager.py # Data loading and preprocessing
├── particle_filter_model.py # SMC / Particle Filter implementation
├── trainer.py # Main orchestration script
├── push_results.py # Upload results to Hugging Face
├── streamlit_app.py # Interactive dashboard
├── us_calendar.py # U.S. market calendar utilities
├── requirements.txt # Python dependencies
├── .github/workflows/ # Scheduled GitHub Action
└── .streamlit/ # Streamlit theme

text

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_PARTICLES` | 5000 | Number of particles |
| `STATE_NOISE_TYPE` | `"t"` | State noise distribution (`"t"` or `"gaussian"`) |
| `STATE_NOISE_SCALE` | 0.01 | Daily volatility of latent state |
| `OBS_NOISE_SCALE` | 0.1 | Observation noise scale |
| `OBS_DF` | 3 | Degrees of freedom for t‑distributed observation noise |
| `LOOKBACK_WINDOW` | 252 | Days used for filter initialization |

## Running Locally

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-PARTICLE-FILTER.git
cd P2-ETF-PARTICLE-FILTER
pip install -r requirements.txt
export HF_TOKEN="your_token_here"
python trainer.py
streamlit run streamlit_app.py
Dashboard Features
Top 3 ETF Picks: Displayed per universe with forecasted returns and 95% credible intervals.

Next Trading Day: U.S. market calendar integration.

Shrinking Windows: Historical performance of top picks across different lookback periods.

Integration with Other Engines
The particle filter's posterior uncertainty can be used to:

Gate signals from BSTS or Factor Autoencoder when forecast uncertainty is high.

Adjust position sizes inversely proportional to forecast standard deviation.

License
MIT License
