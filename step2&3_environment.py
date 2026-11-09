"""
Step 2 & 3: Custom Gymnasium Digital Twin Environment for Open-Pit Blast Optimization
====================================================================================
Paper: Optimizing Open-Pit Blast Design via Soft Actor-Critic Reinforcement Learning
Journal: Mining, Metallurgy & Exploration (Springer Nature)

Description:
    This module defines the custom Gymnasium environment `BlastOptimizationEnv` representing
    the site-calibrated digital twin of open-pit bench blasting operations at Sin Quyen Copper Mine.
    It encapsulates continuous MDP formulation, geomechanical guardrails, Kuz-Ram & Rosin-Rammler 
    fragmentation dynamics, and multi-objective economic penalty reward calculations.
"""

import argparse
import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        # Standalone Fallback Mock for environments without gymnasium preinstalled
        class MockGym:
            class Env:
                pass
        class MockSpaces:
            class Box:
                def __init__(self, low, high, shape=None, dtype=None):
                    self.low = low
                    self.high = high
                    self.shape = shape or (len(low) if hasattr(low, '__len__') else None)
                    self.dtype = dtype
                def __repr__(self):
                    return f"Box(low={self.low}, high={self.high}, shape={self.shape})"
        gym = MockGym()
        spaces = MockSpaces()


class BlastOptimizationEnv(gym.Env):
    """
    Custom Gymnasium Environment for Open-Pit Bench Blast Optimization.
    
    Observation Space (5D continuous vector):
        - mu_A       : Rolling mean rock mass factor (90-day window)
        - sigma_A    : Rolling std of rock mass factor (spatial-temporal volatility)
        - mu_Cn      : Rolling mean site calibration multiplier
        - sigma_Cn   : Rolling std of site calibration multiplier
        - H          : Bench height (m)
        
    Action Space (3D continuous vector normalized to [-1, 1]):
        - Action 0   : Burden (B) scaling -> mapped to [B_min, B_max]
        - Action 1   : Spacing-to-Burden ratio (S/B) -> mapped to [1.1, 1.5]
        - Action 2   : Powder Factor (PF) scaling -> mapped to [PF_min, PF_max]
    """
    
    metadata = {'render_modes': ['human']}

    def __init__(self, data_df=None, config=None):
        super(BlastOptimizationEnv, self).__init__()
        
        self.config = config or {
            'B_bounds': (3.0, 6.0),       # Burden bounds (m)
            'SB_ratio_bounds': (1.1, 1.5),# Spacing-to-Burden ratio bounds
            'PF_bounds': (0.3, 0.8),      # Powder factor bounds (kg/m3)
            'cost_explosive': 1.85,       # USD / kg
            'cost_drilling': 12.50,       # USD / m
            'penalty_oversize': 15.0,     # USD / m3 penalty for F90 > 80cm
            'penalty_fines': 8.0,         # USD / m3 penalty for F10 < 4cm
            'penalty_confinement': 25.0,  # USD / m3 penalty for stemming deficit (< 0.7B)
            'hole_diameter_mm': 165.0,    # Hole diameter d (mm)
            'explosive_RWS': 100.0,       # ANFO Relative Weight Strength
        }
        
        self.data_df = data_df
        self.current_idx = 0
        
        # Action space: 3 continuous design variables normalized in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        
        # Observation space: 5 continuous features
        # [mu_A, sigma_A, mu_Cn, sigma_Cn, H]
        obs_low = np.array([1.0, 0.0, 0.5, 0.0, 5.0], dtype=np.float32)
        obs_high = np.array([15.0, 5.0, 3.0, 2.0, 25.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    def _denormalize_action(self, action):
        """Map normalized [-1, 1] actions to physical parameter ranges."""
        b_min, b_max = self.config['B_bounds']
        sb_min, sb_max = self.config['SB_ratio_bounds']
        pf_min, pf_max = self.config['PF_bounds']
        
        # Linear rescaling from [-1, 1] to [min, max]
        B = b_min + 0.5 * (action[0] + 1.0) * (b_max - b_min)
        SB_ratio = sb_min + 0.5 * (action[1] + 1.0) * (sb_max - sb_min)
        PF = pf_min + 0.5 * (action[2] + 1.0) * (pf_max - pf_min)
        
        S = B * SB_ratio
        return B, S, PF

    def step(self, action):
        """Execute action, evaluate Kuz-Ram digital twin, and calculate multi-objective reward."""
        action = np.clip(action, -1.0, 1.0)
        B, S, PF = self._denormalize_action(action)
        
        # Retrieve state observations
        obs = self._get_obs()
        mu_A, sigma_A, mu_Cn, sigma_Cn, H = obs
        
        # Hard Geomechanical Guardrail Checks
        guardrail_penalty = 0.0
        if B / H > 0.5:
            guardrail_penalty += 50.0  # Excessive burden for bench height
            
        # Stemming confinement check (L_stemming >= 0.7 * B)
        L_stemming = 0.75 * B  # Standard design practice stemming
        confinement_deficit = max(0.0, 0.7 * B - L_stemming)
        
        # --- Digital Twin Fragmentation Inversion (Kuz-Ram Model) ---
        d_mm = self.config['hole_diameter_mm']
        E_rws = self.config['explosive_RWS']
        
        # Mean fragment size x50 (cm)
        # x50 = A * (K)^(-0.8) * Q^(1/6) * (115 / E)^(19/30)
        K = PF  # Powder factor kg/m3
        Q = PF * (B * S * H) / max(1, int(H / 2))  # Explosive mass per hole (kg)
        x50 = mu_A * (K ** -0.8) * (Q ** (1/6)) * ((115.0 / E_rws) ** (19/30)) * mu_Cn
        
        # Rosin-Rammler Uniformity Index (n)
        n = (2.2 - 14.0 * (B / d_mm)) * np.sqrt((1.0 + S / B) / 2.0) * (1.0 - (0.1 / B)) * (H / max(H, 1.0))
        n = max(0.6, min(n, 2.2))
        
        # Characteristic size xc (cm)
        xc = x50 / ((np.log(2.0)) ** (1.0 / n))
        
        # Cumulative Passing Percentages P(x) = 1 - exp(-(x/xc)^n)
        F90_cm = xc * ((-np.log(1.0 - 0.90)) ** (1.0 / n))
        F10_cm = xc * ((-np.log(1.0 - 0.10)) ** (1.0 / n))
        
        # --- Multi-Objective Economic Cost & Penalty Function ---
        # Direct Drilling & Blasting Cost (USD / m3)
        cost_drilling_per_m3 = (self.config['cost_drilling'] * (H + 1.0)) / (B * S * H)
        cost_explosive_per_m3 = PF * self.config['cost_explosive']
        opex_direct = cost_drilling_per_m3 + cost_explosive_per_m3
        
        # Downstream Comminution & Safety Deficit Penalties
        penalty_oversize = self.config['penalty_oversize'] if F90_cm > 80.0 else 0.0
        penalty_fines = self.config['penalty_fines'] if F10_cm < 4.0 else 0.0
        penalty_confinement = self.config['penalty_confinement'] * confinement_deficit
        
        total_cost = opex_direct + penalty_oversize + penalty_fines + penalty_confinement + guardrail_penalty
        
        # Reward is negative total cost (to be maximized by RL agent)
        reward = -total_cost
        
        # Advance index or signal episode termination
        self.current_idx += 1
        terminated = True  # Single-step contextual decision per blast pattern
        truncated = False
        
        info = {
            'Burden_m': B,
            'Spacing_m': S,
            'PF_kg_m3': PF,
            'x50_cm': x50,
            'F90_cm': F90_cm,
            'F10_cm': F10_cm,
            'OPEX_direct_usd_m3': opex_direct,
            'Total_Cost_usd_m3': total_cost,
            'Penalty_oversize': penalty_oversize,
            'Penalty_fines': penalty_fines
        }
        
        next_obs = self._get_obs()
        return next_obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Construct state observation vector from dataset or sample synthetic state."""
        if self.data_df is not None and self.current_idx < len(self.data_df):
            row = self.data_df.iloc[self.current_idx]
            obs = np.array([
                row.get('mu_A', 7.5),
                row.get('sigma_A', 0.8),
                row.get('mu_Cn', 1.0),
                row.get('sigma_Cn', 0.15),
                row.get('H', 12.0)
            ], dtype=np.float32)
        else:
            # Fallback default observation
            obs = np.array([7.5, 0.8, 1.0, 0.15, 12.0], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        if options and 'idx' in options:
            self.current_idx = options['idx']
        elif self.data_df is not None:
            self.current_idx = np.random.randint(0, len(self.data_df))
        else:
            self.current_idx = 0
            
        obs = self._get_obs()
        info = {}
        return obs, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2 & 3: Custom Blast Digital Twin Environment")
    args = parser.parse_args()
    
    env = BlastOptimizationEnv()
    obs, info = env.reset()
    print("Environment Initialized Successfully.")
    print("Sample Observation Space:", env.observation_space)
    print("Sample Action Space:", env.action_space)
    print("Sample Initial State Vector:", obs)
    
    # Run a test step with zero action (midpoint)
    test_action = np.zeros(3, dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(test_action)
    print("\n--- Test Step Execution Results ---")
    print(f"Design Action (B, S, PF): Burden={step_info['Burden_m']:.2f}m, Spacing={step_info['Spacing_m']:.2f}m, PF={step_info['PF_kg_m3']:.2f}kg/m3")
    print(f"Predicted x50: {step_info['x50_cm']:.2f} cm")
    print(f"Predicted F90: {step_info['F90_cm']:.2f} cm")
    print(f"Direct OPEX: ${step_info['OPEX_direct_usd_m3']:.2f}/m3")
    print(f"Total Cost Reward: ${-reward:.2f}/m3")
