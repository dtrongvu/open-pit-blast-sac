# open-pit-blast-sac
Replication code for Open-Pit Blast Optimization via SAC DRL
# Open-Pit Blast Design Optimization via Soft Actor-Critic (SAC) Reinforcement Learning

[![Journal]([suspicious link removed])](https://link.springer.com/journal/42461)
[![Python 3.10+]([suspicious link removed])]([suspicious link removed])
[![RL-Library]([suspicious link removed])]([suspicious link removed])

This repository contains the official implementation for the paper: **"Optimizing Open-Pit Blast Design via Soft Actor-Critic Reinforcement Learning: A Continuous Spatio-Temporal Approach"** submitted to *Mining, Metallurgy & Exploration* (Springer Nature).

---

## 📌 Overview

Drill and Blast (D&B) engineering in open-pit mining is traditionally reliant on static empirical equations (e.g., Kuz-Ram, Swebrec) or human heuristics, which fail to adapt to local geomechanical volatility across advancing bench horizons.

This codebase implements a continuous **Deep Reinforcement Learning (DRL)** framework based on the **Soft Actor-Critic (SAC)** algorithm. By modeling the blast sequencing process as a continuous Markov Decision Process (MDP), the agent continuously optimizes pattern geometry (Burden \\(B\\), Spacing \\(S\\)) and explosive energy factor (\\(PF\\)) within a site-calibrated digital twin environment.

---

## 🚀 Key Framework Features

1. **Bench-Constrained Temporal Memory Window (90-Day Horizon):** Captures local spatial-temporal geomechanical drift (\\(\mu_A, \sigma_A, \mu_{C_n}, \sigma_{C_n}\\)) to eliminate data leakage and cold-start bias.
2. **Site-Calibrated Digital Twin Environment:** Integrates inverted Kuz-Ram and Rosin-Rammler dynamics with hard geomechanical guardrails (\\(B/H \le 0.5\\), \\(1.1B \le S \le 1.5B\\)).
3. **Multi-Objective Economic Penalty Function:** Balances direct operational expenditure (OPEX) with asymmetric downstream comminution penalties (\\(F_{90} > 80\text{ cm}\\), \\(F_{10} < 4\text{ cm}\\)) and stemming confinement safety thresholds.
4. **Millisecond-Level Inference:** Outperforms static metaheuristics (GA, PSO) in computational speed, enabling real-time edge integration on autonomous drill rigs.

---

## 📁 Repository Structure

```text
├── README.md                        # Documentation and replication guide
├── requirements.txt                 # Python dependencies
├── step1_feature_engineering.py     # Step 1: Spatio-temporal memory window & data split
├── step2_environment.py             # Steps 2 & 3: Custom Gym Blast Environment (Digital Twin)
├── step4_train.py                   # Step 4: SAC Agent Training & Checkpoint Callback
├── step5_validate.py                # Step 5: Forensic Validation Evaluation
└── step6_test_evaluation.py         # Step 6: Final Test Set Offline Chronological Shadow Replay
