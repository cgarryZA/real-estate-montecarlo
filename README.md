# Leveraged Property Investment Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/<yourusername>/real-estate-montecarlo/main/streamlit_app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project models the long-term performance of a **leveraged property investment strategy** using Monte Carlo simulation, including **corporate tax**, **refinancing logic**, and **benchmark comparison** against passive investment and risk-free alternatives.

It provides both an **interactive Streamlit dashboard** and a **command-line simulation engine** for research and quantitative experimentation.

---

## Overview

Traditional property ROI models often ignore leverage, amortisation, refinancing, tax, and reinvestment.  
This simulator provides a **quantitative, stochastic framework** to evaluate a rental property’s leveraged performance under uncertainty — including dynamic reinvestment, refinance decisions, and tax drag.

The model evolves:
- **Property value** as a geometric Brownian motion (GBM) process with drift and volatility.
- **Rental income** with its own stochastic drift and variance.
- **Mortgage balance** through amortisation and optional refinancing.
- **Investment account** that accumulates after-tax surplus cash and refinance proceeds.
- **Corporate tax** on rental profit, with principal treated as an after-tax outflow.
- **Benchmark portfolios** that simulate investing the same initial capital in the market or at a risk-free rate.

Monte Carlo simulation produces full **distributions of outcomes**, including:
- Ending portfolio value (equity + investment account)
- Present-value discounted wealth
- Return multiples on initial outlay
- Mean time evolution of equity, investments, and portfolio
- Comparative performance vs benchmark and risk-free scenarios

---

## Features

- **Full stochastic property and rent modelling**
  - Each evolves independently via GBM with drift and volatility parameters.
- **Refinance and amortisation logic**
  - Implements a *"dumb amortisation rule"* that rebalances to target LTV when equity growth underperforms the market.
- **Corporate tax treatment**
  - Tax = (rental income − expenses − interest) × corporate tax rate.
  - Principal payments are treated as non-deductible cash outflows.
- **Dynamic reinvestment** of surplus post-tax cash and refinance proceeds.
- **Monte Carlo engine** (NumPy-based) for thousands of stochastic paths.
- **Benchmark comparison:**
  - Parallel simulation of a portfolio investing the same initial outlay directly in the market.
  - Includes risk-free growth curve overlay.
  - Computes Sharpe ratios for both strategy and benchmark for direct comparison.
- **Visualisation suite (Streamlit + Matplotlib):**
  - Strategy vs benchmark mean-path comparison (with Sharpe annotations).
  - Ending portfolio value histograms.
  - Present Value (PV) multiple distributions.
  - ECDF (cumulative) plots of discounted final wealth.
  - Deposit-only reference marker (shows net invested capital excluding fees).

---

## Example Outputs

- Mean equity, investment, and total portfolio trajectories.
- Comparative Sharpe ratios for strategy vs passive benchmark.
- Distribution of ending portfolio value and PV multiples.
- ECDF of discounted terminal wealth.
- Risk-free and benchmark overlays for intuitive ROI comparison.

---

## Project Structure

```text
real-estate-montecarlo/
├── streamlit_app.py          # Streamlit interface (interactive dashboard)
├── main.py                   # CLI / research entry point
├── simulate.py               # Core Monte Carlo simulation logic
├── models.py                 # Dataclasses for parameters
├── mortgage.py               # Mortgage repayment and amortisation functions
├── analyze.py                # Summary statistics and analytics utilities
├── requirements.txt          # Dependencies
└── README.md
```

---

## Getting Started

### 1. Local setup

```bash
git clone https://github.com/<yourusername>/real-estate-montecarlo.git
cd real-estate-montecarlo
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the URL displayed in your terminal (typically http://localhost:8501).

### 2. Run in Python

You can also run batch simulations programmatically:

```python
from simulate import run_mc_with_paths
from models import PropertyParams, MortgageParams, RefiParams, InvestmentParams, SimulationParams

prop = PropertyParams(initial_price=300000, price_drift=0.03, price_vol=0.12,
                      rent_initial=2000, rent_drift=0.02, rent_vol=0.08, expense_ratio=0.1)
mort = MortgageParams(initial_ltv=0.8, rate=0.04, term_years=35)
refi = RefiParams(max_ltv=0.75, refi_fee_ratio=0.02, rate_spread=0.005, decision_interval_years=1)
inv = InvestmentParams(exp_return=0.07, vol=0.15)
sim = SimulationParams(years=15, steps_per_year=12, n_paths=5000)

results = run_mc_with_paths(prop, mort, refi, inv, sim, corporate_tax_rate=0.19)
```

---

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
- Streamlit
- (Optional) SciPy for kernel density estimation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Research Direction

This simulator forms the foundation for further work in:
- **Optimal refinance timing** via dynamic programming.
- **Dynamic leverage optimisation** (e.g. reinforcement learning control policies).
- **Multi-property stochastic portfolio modelling.**
- **Cross-tax regime comparisons** (SPV vs personal ownership vs LLP).
- **Integration into portfolio-level risk/return frontiers.**

---

## License

MIT License — feel free to use, modify, and extend this project for academic or commercial use.

---

## Author

**Christian Garry**  
Graduate Communications Engineer @ Siemens  
MSc Scientific Computing and Data Analysis (AI for Engineering), Durham University
