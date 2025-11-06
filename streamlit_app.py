import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from simulate import run_mc_with_paths, compute_initial_outlay
from analyze import summarize
from models import (
    PropertyParams,
    MortgageParams,
    RefiParams,
    InvestmentParams,
    SimulationParams,
    AcquisitionCosts,
    StampDutyParams,
)

st.set_page_config(page_title="Leveraged Property Monte Carlo", layout="wide")

st.title("Leveraged Property Investment Simulator")
st.markdown(
    "Play with purchase price, LTV, growth, and refinance rules to see portfolio value distributions and average equity and investment paths."
)

# -------------------------
# SIDEBAR INPUTS
# -------------------------
st.sidebar.header("Property & Rent")
price = st.sidebar.number_input("Purchase price (£)", 50_000, 2_000_000, 295_000, 5_000)
price_drift = st.sidebar.slider("Property drift (annual %)", 0.0, 0.08, 0.03, 0.005)
price_vol = st.sidebar.slider("Property vol (annual %)", 0.0, 0.4, 0.12, 0.01)
monthly_rent = st.sidebar.number_input("Initial monthly rent (£)", 200.0, 10_000.0, 2_100.0, 50.0)
rent_drift = st.sidebar.slider("Rent drift (annual %)", 0.0, 0.08, 0.02, 0.005)
rent_vol = st.sidebar.slider("Rent vol (annual %)", 0.0, 0.3, 0.08, 0.01)
expense_ratio = st.sidebar.slider("Expense ratio of rent", 0.0, 0.5, 0.10, 0.01)

st.sidebar.header("Mortgage")
initial_ltv = st.sidebar.slider("Initial LTV", 0.5, 0.95, 0.85, 0.01)
mort_rate = st.sidebar.slider("Mortgage rate (annual %)", 0.0, 0.1, 0.04, 0.001)
term_years = st.sidebar.slider("Mortgage term (years)", 5, 40, 35, 1)

st.sidebar.header("Refinance")
max_ltv = st.sidebar.slider("Max LTV on refi", 0.5, 0.95, 0.75, 0.01)
refi_fee = st.sidebar.slider("Refi fee (% of new loan)", 0.0, 0.05, 0.02, 0.001)
rate_spread = st.sidebar.slider("Rate improvement on refi (abs %)", 0.0, 0.03, 0.005, 0.001)
refi_interval = st.sidebar.slider("Refi decision interval (years)", 1, 5, 2, 1)

st.sidebar.header("Investment")
inv_return = st.sidebar.slider("Investment exp. return (annual %)", 0.0, 0.15, 0.07, 0.005)
inv_vol = st.sidebar.slider("Investment vol (annual %)", 0.0, 0.5, 0.15, 0.01)

st.sidebar.header("Simulation")
years = st.sidebar.slider("Horizon (years)", 5, 30, 15, 1)
steps_per_year = st.sidebar.selectbox("Steps per year", [12, 4, 1], index=0)
n_paths = st.sidebar.slider("Monte Carlo paths", 200, 10_000, 3_000, 200)

st.sidebar.header("Acquisition costs")
other_fixed = st.sidebar.number_input("Other fixed costs (£)", 0, 10_000, 600, 100)
searches_fixed = st.sidebar.number_input("Searches (£)", 0, 5_000, 400, 50)
solicitor_pct = st.sidebar.number_input("Solicitor (% of price)", 0.0, 0.02, 0.001, 0.0005)
mortgage_fee_pct = st.sidebar.number_input("Mortgage fee (% of loan)", 0.0, 0.02, 0.005, 0.0005)
sdlt_surcharge = st.sidebar.number_input("SDLT surcharge", 0.0, 0.05, 0.03, 0.001)

run_button = st.sidebar.button("Run simulation")

# -------------------------
# BUILD PARAM OBJECTS
# -------------------------
prop = PropertyParams(
    initial_price=price,
    price_drift=price_drift,
    price_vol=price_vol,
    rent_initial=monthly_rent,
    rent_drift=rent_drift,
    rent_vol=rent_vol,
    expense_ratio=expense_ratio,
)

mort = MortgageParams(
    initial_ltv=initial_ltv,
    rate=mort_rate,
    term_years=term_years,
)

refi = RefiParams(
    max_ltv=max_ltv,
    refi_fee_ratio=refi_fee,
    rate_spread=rate_spread,
    decision_interval_years=refi_interval,
)

inv = InvestmentParams(
    exp_return=inv_return,
    vol=inv_vol,
)

sim = SimulationParams(
    years=years,
    steps_per_year=steps_per_year,
    n_paths=n_paths,
)

sdlt = StampDutyParams(
    bands=[
        (0, 250_000, 0.0),
        (250_000, 925_000, 0.05),
        (925_000, 1_500_000, 0.10),
        (1_500_000, float("inf"), 0.12),
    ],
    surcharge=sdlt_surcharge,
)

acq = AcquisitionCosts(
    other_fixed=other_fixed,
    searches_fixed=searches_fixed,
    solicitor_pct_of_price=solicitor_pct,
    mortgage_fee_pct_of_loan=mortgage_fee_pct,
)

if run_button:
    mc = run_mc_with_paths(prop, mort, refi, inv, sim, acq, sdlt)
    finals = mc["finals"]
    equity_paths = mc["equity_paths"]
    inv_paths = mc["inv_paths"]
    portfolio_paths = mc["portfolio_paths"]  # <-- this was wealth_paths before

    initial_outlay = compute_initial_outlay(prop, mort, acq, sdlt)

    # discount to PV
    df = (1 + 0.03) ** years
    finals_pv = finals / df

    stats = summarize(finals)
    stats_pv = summarize(finals_pv)

    st.subheader("Summary statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.json(stats)
    with col2:
        st.json(stats_pv)

    # average paths
    n_steps = years * steps_per_year
    t = np.arange(n_steps) / steps_per_year
    equity_mean = equity_paths.mean(axis=0)
    inv_mean = inv_paths.mean(axis=0)
    portfolio_mean = portfolio_paths.mean(axis=0)  # <-- updated name

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(t, equity_mean, label="Mean equity")
    ax1.plot(t, inv_mean, label="Mean investment")
    ax1.plot(t, portfolio_mean, label="Mean portfolio value", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Years")
    ax1.set_ylabel("£")
    ax1.set_title("Average path across simulations")
    ax1.legend()
    st.pyplot(fig1)

    # hist raw
    bins = int(np.sqrt(n_paths))
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(finals, bins=bins, edgecolor="black")
    ax2.axvline(stats["mean"], color="red", linestyle="--", label=f"Mean £{stats['mean']:.0f}")
    ax2.axvline(initial_outlay, color="blue", linestyle="-", label=f"Initial outlay £{initial_outlay:.0f}")
    ax2.set_title("Ending portfolio value")
    ax2.legend()
    st.pyplot(fig2)

    # multiples
    multiples = finals_pv / initial_outlay
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.hist(multiples, bins=bins, edgecolor="black", color="lightgreen")
    ax3.axvline(1.0, color="blue", linestyle="-", label="1.0× initial cash")
    ax3.axvline(np.median(multiples), color="green", linestyle="--", label=f"Median {np.median(multiples):.2f}×")
    ax3.set_title("PV multiple of initial outlay")
    ax3.legend()
    st.pyplot(fig3)

else:
    st.info("Set your parameters in the sidebar and click **Run simulation**.")
