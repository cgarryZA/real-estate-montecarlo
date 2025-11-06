# Leveraged Property Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/<yourusername>/leveraged-property-simulator/main/streamlit_app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project models the long-term performance of a **leveraged property investment strategy** using Monte Carlo simulation.  
It allows users to explore how property prices, rental income, mortgage structure, refinancing rules, and reinvestment of surplus cash interact to determine portfolio outcomes through time.

The simulator can be run interactively in **Streamlit** or directly from Python for research and analysis.

---

## Overview

Conventional property ROI models often ignore leverage, amortisation, refinancing, and reinvestment.  
This simulator provides a dynamic, data-driven framework to evaluate a rental property’s investment performance under uncertainty.

The model evolves:
- **Property value** as a stochastic process with drift and volatility.  
- **Rental income** with its own growth and randomness.  
- **Mortgage balance** through amortisation and optional refinancing.  
- **Investment account** that accumulates reinvested cash flows and refinance proceeds.

Monte Carlo simulation produces the full **distribution** of results rather than a single deterministic forecast, including:
- Ending portfolio value (equity + investment account)  
- Present-value-discounted outcomes  
- Return multiples on initial outlay  
- Mean time evolution of equity and cash investments  

---

## Features

- **Stochastic modelling** of property and rent.  
- **Amortisation & refinance logic**  
  - Implements a “**dumb amortisation rule**”: the mortgage is rebalanced up to a target LTV whenever the modelled equity growth underperforms the investment market, ignoring future expectations.  
  - Serves as a baseline for later optimal-stopping or dynamic-programming approaches.  
- **Investment portfolio** to reinvest cash flow and refinance proceeds.  
- **Acquisition cost breakdown** (stamp duty, solicitor, mortgage, and fixed fees).  
- **Present value discounting** and summary metrics.  
- **Streamlit web interface** with interactive sliders and live Matplotlib charts.  
- **Matplotlib visualisations**:  
  - Portfolio value histograms  
  - PV multiples (total vs investment-only)  
  - ECDF plots and cumulative distributions  
  - Average equity and investment paths over time  

---

## Example Outputs

- Distribution of portfolio value after 15 years.  
- Median and mean PV-adjusted results vs initial outlay.  
- Mean trajectories of property equity and reinvested portfolio.  
- Sensitivity to leverage, interest rates, rent drift, and volatility.  

---

## Project Structure

```
leveraged-property-simulator/
├── streamlit_app.py          # Interactive Streamlit dashboard
├── main.py                   # CLI/batch version
├── simulate.py               # Monte Carlo engine
├── models.py                 # Parameter dataclasses
├── mortgage.py               # Mortgage repayment logic
├── analyze.py                # Summary statistics and utilities
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

Then open the link shown in your terminal (usually http://localhost:8501).

### 2. Run in Python

You can also run simulations directly in Python:

```python
from simulate import run_mc_with_paths
from models import PropertyParams, MortgageParams, RefiParams, InvestmentParams, SimulationParams

prop = PropertyParams(initial_price=300000, price_drift=0.03, price_vol=0.12, rent_initial=2000, rent_drift=0.02, rent_vol=0.08, expense_ratio=0.1)
mort = MortgageParams(initial_ltv=0.8, rate=0.04, term_years=35)
refi = RefiParams(max_ltv=0.75, refi_fee_ratio=0.02, rate_spread=0.005, decision_interval_years=1)
inv = InvestmentParams(exp_return=0.07, vol=0.15)
sim = SimulationParams(years=15, steps_per_year=12, n_paths=5000)

results = run_mc_with_paths(prop, mort, refi, inv, sim)
```

---

## Requirements

- Python 3.10+  
- NumPy  
- Matplotlib  
- Streamlit  
- (Optional) SciPy for kernel density estimation  

Install with:

```bash
pip install -r requirements.txt
```

---

## Research Direction

This simulator provides a foundation for future quantitative extensions, such as:

- Optimal refinance timing under uncertainty.  
- Dynamic leverage control using reinforcement learning.  
- Risk-return frontiers for stochastic real assets.  
- Integration into larger portfolio optimisation frameworks.  

---

## License

MIT License.  
Feel free to use, modify, and extend this project for academic or commercial purposes.

---

## Author

**Christian Garry**  
Graduate Communications Engineer at Siemens  
MSc Scientific Computing & Data Analysis (AI for Engineering)  
Durham University
