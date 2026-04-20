"""
Main training script for Particle Filter engine.
Runs SMC on each ETF, forecasts next-day returns, and ranks ETFs.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from particle_filter_model import ParticleFilter
import push_results

def run_particle_filter():
    print(f"=== P2-ETF-PARTICLE-FILTER Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    
    all_results = {}
    top_picks = {}
    
    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        universe_results = {}
        
        for ticker in tickers:
            print(f"  Filtering {ticker}...")
            returns = data_manager.prepare_returns_series(df_master, ticker)
            if len(returns) < config.MIN_OBSERVATIONS:
                continue
            # Use recent window
            returns_recent = returns.iloc[-config.LOOKBACK_WINDOW:].values
            
            pf = ParticleFilter(
                num_particles=config.NUM_PARTICLES,
                state_noise_scale=config.STATE_NOISE_SCALE,
                obs_noise_scale=config.OBS_NOISE_SCALE,
                obs_df=config.OBS_DF,
                state_noise_type=config.STATE_NOISE_TYPE,
                resample_threshold=config.RESAMPLE_THRESHOLD
            )
            # Run filter
            pf.filter_series(returns_recent, initial_value=returns_recent[0])
            # Forecast next day
            forecast_mean, forecast_ci = pf.forecast(horizon=1, n_samples=2000)
            if forecast_mean is None:
                continue
            
            universe_results[ticker] = {
                'ticker': ticker,
                'forecast_mean': float(forecast_mean),
                'forecast_lower': float(forecast_ci[0]),
                'forecast_upper': float(forecast_ci[1]),
                'forecast_std': float((forecast_ci[1] - forecast_ci[0]) / 3.92)  # approx std
            }
        
        if universe_results:
            # Sort by forecast_mean descending
            sorted_tickers = sorted(universe_results.items(), key=lambda x: x[1]['forecast_mean'], reverse=True)
            top_picks[universe_name] = [
                {'ticker': t, 'forecast_mean': d['forecast_mean'],
                 'forecast_lower': d['forecast_lower'], 'forecast_upper': d['forecast_upper']}
                for t, d in sorted_tickers[:config.TOP_N_DISPLAY]
            ]
            all_results[universe_name] = universe_results
    
    # Shrinking windows
    shrinking_results = {}
    for start_year in config.SHRINKING_WINDOW_START_YEARS:
        start_date = pd.Timestamp(f"{start_year}-01-01")
        window_label = f"{start_year}-{config.TODAY[:4]}"
        mask = df_master['Date'] >= start_date
        df_window = df_master[mask].copy()
        if len(df_window) < config.MIN_OBSERVATIONS:
            continue
        
        window_top = {}
        for universe_name, tickers in config.UNIVERSES.items():
            best_ticker = None
            best_forecast = -np.inf
            for ticker in tickers:
                returns = data_manager.prepare_returns_series(df_window, ticker)
                if len(returns) < config.MIN_OBSERVATIONS:
                    continue
                returns_recent = returns.iloc[-config.LOOKBACK_WINDOW:].values
                pf = ParticleFilter(**pf_params)
                pf.filter_series(returns_recent, initial_value=returns_recent[0])
                fc, _ = pf.forecast(horizon=1, n_samples=1000)
                if fc is not None and fc > best_forecast:
                    best_forecast = fc
                    best_ticker = ticker
            if best_ticker:
                window_top[universe_name] = {'ticker': best_ticker, 'forecast_mean': best_forecast}
        shrinking_results[window_label] = {'start_year': start_year, 'top_picks': window_top}
    
    output_payload = {
        "run_date": config.TODAY,
        "config": {
            "num_particles": config.NUM_PARTICLES,
            "state_noise_scale": config.STATE_NOISE_SCALE,
            "obs_noise_scale": config.OBS_NOISE_SCALE,
            "obs_df": config.OBS_DF,
            "state_noise_type": config.STATE_NOISE_TYPE
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        },
        "shrinking_windows": shrinking_results
    }
    
    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    # Define pf_params for shrinking windows use
    pf_params = dict(
        num_particles=config.NUM_PARTICLES,
        state_noise_scale=config.STATE_NOISE_SCALE,
        obs_noise_scale=config.OBS_NOISE_SCALE,
        obs_df=config.OBS_DF,
        state_noise_type=config.STATE_NOISE_TYPE,
        resample_threshold=config.RESAMPLE_THRESHOLD
    )
    run_particle_filter()
