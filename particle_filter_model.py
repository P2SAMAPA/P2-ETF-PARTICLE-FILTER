"""
Particle Filter for state-space model:
State: x_t = x_{t-1} + v_t,   v_t ~ Student-t (df=3) or Gaussian
Observation: y_t = x_t + w_t, w_t ~ Gaussian or Student-t
"""

import numpy as np
from scipy import stats
from typing import Tuple

class ParticleFilter:
    def __init__(self, num_particles=5000, state_noise_scale=0.01, obs_noise_scale=0.1,
                 obs_df=3, state_noise_type='t', resample_threshold=0.5):
        self.num_particles = num_particles
        self.state_noise_scale = state_noise_scale
        self.obs_noise_scale = obs_noise_scale
        self.obs_df = obs_df
        self.state_noise_type = state_noise_type
        self.resample_threshold = resample_threshold
        self.particles = None
        self.weights = None
        
    def initialize_particles(self, initial_value=0.0):
        self.particles = np.full(self.num_particles, initial_value)
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def predict(self):
        if self.state_noise_type == 't':
            noise = stats.t.rvs(df=3, size=self.num_particles) * self.state_noise_scale
        else:
            noise = np.random.normal(0, self.state_noise_scale, self.num_particles)
        self.particles += noise
        
    def update(self, observation):
        if np.isnan(observation):
            return
        # Likelihood: observation noise (t-distributed for robustness)
        if self.obs_df > 0:
            log_lik = stats.t.logpdf(observation, df=self.obs_df, loc=self.particles, scale=self.obs_noise_scale)
        else:
            log_lik = stats.norm.logpdf(observation, loc=self.particles, scale=self.obs_noise_scale)
        # Stabilize weights
        max_log = np.max(log_lik)
        weights = np.exp(log_lik - max_log)
        self.weights *= weights
        self.weights /= np.sum(self.weights)
        
    def resample(self):
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.resample_threshold * self.num_particles:
            indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
            
    def step(self, observation):
        self.predict()
        self.update(observation)
        self.resample()
        
    def filter_series(self, observations: np.ndarray, initial_value=0.0):
        self.initialize_particles(initial_value)
        state_means = []
        state_vars = []
        for obs in observations:
            self.step(obs)
            state_means.append(np.average(self.particles, weights=self.weights))
            state_vars.append(np.average((self.particles - state_means[-1])**2, weights=self.weights))
        return np.array(state_means), np.array(state_vars)
    
    def forecast(self, horizon=1, n_samples=1000):
        """Sample from predictive distribution for next step."""
        if self.particles is None:
            return None, None
        # Sample from current posterior
        idx = np.random.choice(self.num_particles, n_samples, p=self.weights)
        states = self.particles[idx]
        # Propagate through state noise
        if self.state_noise_type == 't':
            noise = stats.t.rvs(df=3, size=n_samples) * self.state_noise_scale
        else:
            noise = np.random.normal(0, self.state_noise_scale, n_samples)
        pred_states = states + noise
        # Add observation noise
        if self.obs_df > 0:
            obs_noise = stats.t.rvs(df=self.obs_df, size=n_samples) * self.obs_noise_scale
        else:
            obs_noise = np.random.normal(0, self.obs_noise_scale, n_samples)
        pred_obs = pred_states + obs_noise
        return np.mean(pred_obs), np.percentile(pred_obs, [2.5, 97.5])
