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
    "Adjust the parameters in the sidebar and simulate how a leveraged property and reinvestment strategy performs over time."
)

# ------------------------------------------------------------
# helper: benchmark “just invest the initial outlay”
# ------------------------------------------------------------
def mc_invest_initial_outlay(
    initial_outlay: float,
    exp_return: float,
    vol: float,
    years: int,
    steps_per_year: int,
    n_paths: int,
    seed: int = 999,
):
    rng = np.random.default_rng(seed)
    dt = 1 / steps_per_year
    n_steps = years * steps_per_year
    paths = np.zeros((n_paths, n_steps), dtype=float)
    for i in range(n_paths):
        level = initial_outlay
        for t in range(n_steps):
            paths[i, t] = level
            if t < n_steps - 1:
                z = rng.normal()
                level *= np.exp(
                    (exp_return - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
                )
    mean_path = paths.mean(axis=0)
    return paths, mean_path

# ------------------------------------------------------------
# helper: Sharpe from final MC distribution (robust)
# ------------------------------------------------------------
def sharpe_from_finals(final_values, initial, years, rf):
    ratios = np.maximum(final_values / initial, 1e-9)
    log_returns = np.log(ratios) / years  # annualised log return
    mean_ret = np.mean(log_returns)
    std_ret = np.std(log_returns, ddof=1)
    if std_ret <= 1e-12:
        return 0.0
    return (mean_ret - rf) / std_ret


# =============== SIDEBAR =================
st.sidebar.header("Property & Rent")
price = st.sidebar.number_input("Purchase price (£)", 50_000, 2_000_000, 295_000, 5_000)
price_drift = st.sidebar.slider("Property drift (annual %)", 0.0, 0.08, 0.03, 0.005)
price_vol = st.sidebar.slider("Property vol (annual %)", 0.0, 0.4, 0.12, 0.01)
monthly_rent = st.sidebar.number_input(
    "Initial monthly rent (£)", 200.0, 10_000.0, 2_100.0, 50.0
)
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
rate_spread = st.sidebar.slider(
    "Rate improvement on refi (abs %)", 0.0, 0.03, 0.005, 0.001
)
refi_interval = st.sidebar.slider("Refi decision interval (years)", 1, 5, 2, 1)

st.sidebar.header("Investment")
inv_return = st.sidebar.slider(
    "Investment exp. return (annual %)", 0.0, 0.15, 0.07, 0.005
)
inv_vol = st.sidebar.slider("Investment vol (annual %)", 0.0, 0.5, 0.15, 0.01)

st.sidebar.header("Simulation")
years = st.sidebar.slider("Horizon (years)", 5, 30, 15, 1)
steps_per_year = st.sidebar.selectbox("Steps per year", [12, 4, 1], index=0)
n_paths = st.sidebar.slider("Monte Carlo paths", 200, 10_000, 3_000, 200)

st.sidebar.header("Acquisition costs")
other_fixed = st.sidebar.number_input("Other fixed costs (£)", 0, 10_000, 600, 100)
searches_fixed = st.sidebar.number_input("Searches (£)", 0, 5_000, 400, 50)
solicitor_pct = st.sidebar.number_input(
    "Solicitor (% of price)", 0.0, 0.02, 0.001, 0.0005
)
mortgage_fee_pct = st.sidebar.number_input(
    "Mortgage fee (% of loan)", 0.0, 0.02, 0.005, 0.0005
)
sdlt_surcharge = st.sidebar.number_input("SDLT surcharge", 0.0, 0.05, 0.03, 0.001)

# tax
st.sidebar.header("Tax")
corp_tax = st.sidebar.slider("Corporate tax rate", 0.0, 0.35, 0.19, 0.01)

# risk free for comparison
st.sidebar.header("Comparison")
rf_rate = st.sidebar.slider("Risk free rate", 0.0, 0.06, 0.03, 0.005)

run_button = st.sidebar.button("Run simulation")

# =============== BUILD PARAM OBJECTS ===============
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

# =============== RUN ===============
if run_button:
    # run with tax aware simulate
    try:
        mc = run_mc_with_paths(
            prop,
            mort,
            refi,
            inv,
            sim,
            acq,
            sdlt,
            corporate_tax_rate=corp_tax,
        )
    except TypeError:
        mc = run_mc_with_paths(
            prop,
            mort,
            refi,
            inv,
            sim,
            acq,
            sdlt,
        )

    final_portfolio = mc["finals"]
    equity_paths = mc["equity_paths"]
    inv_paths = mc["inv_paths"]
    portfolio_paths = mc["portfolio_paths"]

    # actual initial cash out (deposit + fees)
    initial_outlay = compute_initial_outlay(prop, mort, acq, sdlt)
    # deposit only (outlay minus fees)
    deposit_only = price * (1.0 - initial_ltv)

    # discount to PV for comparison (same as before)
    df = (1 + 0.03) ** years
    final_portfolio_pv = final_portfolio / df

    # component finals
    final_equity = equity_paths[:, -1]
    final_invest = inv_paths[:, -1]

    stats_total = summarize(final_portfolio)

    # averaged paths for main strategy
    n_steps = years * steps_per_year
    t = np.arange(n_steps) / steps_per_year
    strategy_mean_path = portfolio_paths.mean(axis=0)

    # BENCHMARK MC: invest the initial outlay
    bench_paths, bench_mean_path = mc_invest_initial_outlay(
        initial_outlay,
        exp_return=inv_return,
        vol=inv_vol,
        years=years,
        steps_per_year=steps_per_year,
        n_paths=n_paths,
    )

    # risk free curve
    risk_free_curve = initial_outlay * (1 + rf_rate) ** t

    # Sharpe (robust)
    strategy_sharpe = sharpe_from_finals(
        final_portfolio, initial_outlay, years, rf_rate
    )
    bench_sharpe = sharpe_from_finals(
        bench_paths[:, -1], initial_outlay, years, rf_rate
    )

    # top metrics
    mean_total = final_portfolio.mean()
    mean_equity = final_equity.mean()
    mean_invest = final_invest.mean()

    st.subheader("Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Mean ending portfolio", f"£{mean_total:,.0f}")
    m2.metric("Mean equity component", f"£{mean_equity:,.0f}")
    m3.metric("Mean investment component", f"£{mean_invest:,.0f}")
    m4.metric("Initial outlay (with fees)", f"£{initial_outlay:,.0f}")
    m5.metric(
        "Sharpe (strategy | benchmark)",
        f"{strategy_sharpe:.2f} | {bench_sharpe:.2f}",
    )

    # =============== LAYOUT FOR PLOTS ===============
    col_left, col_right = st.columns(2)

    # left: strategy vs benchmark vs risk free
    with col_left:
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        ax1.plot(
            t,
            strategy_mean_path,
            label=f"Strategy mean (Sharpe {strategy_sharpe:.2f})",
            linewidth=2,
        )
        ax1.plot(
            t,
            bench_mean_path,
            label=f"Invest initial (Sharpe {bench_sharpe:.2f})",
            linestyle="--",
        )
        ax1.plot(
            t,
            risk_free_curve,
            label=f"Risk free {rf_rate*100:.1f}% p.a.",
            linestyle=":",
        )
        ax1.scatter(
            [0],
            [deposit_only],
            color="black",
            zorder=5,
            label="Deposit only (outlay minus fees)",
        )
        ax1.set_xlabel("Years")
        ax1.set_ylabel("£")
        ax1.set_title("Strategy vs investing initial outlay")
        ax1.legend()
        st.pyplot(fig1)

    # right: ending portfolio histogram
    with col_right:
        bins = int(np.sqrt(n_paths))
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        ax2.hist(final_portfolio, bins=bins, edgecolor="black", alpha=0.8)
        ax2.axvline(
            stats_total["mean"],
            color="red",
            linestyle="--",
            label=f"Mean £{stats_total['mean']:.0f}",
        )
        ax2.axvline(
            initial_outlay,
            color="blue",
            linestyle="-",
            label=f"Initial outlay £{initial_outlay:.0f}",
        )
        ax2.set_title("Ending portfolio value")
        ax2.set_xlabel("£")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        st.pyplot(fig2)

    # second row
    col_left2, col_right2 = st.columns(2)

    # left2: PV multiple
    with col_left2:
        total_multiples = final_portfolio_pv / initial_outlay
        invest_pv = final_invest / df
        invest_multiples = invest_pv / initial_outlay

        bins_mult = int(np.sqrt(n_paths))
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        ax3.hist(
            invest_multiples,
            bins=bins_mult,
            edgecolor="black",
            alpha=0.4,
            label="Investment account PV multiple",
        )
        ax3.hist(
            total_multiples,
            bins=bins_mult,
            edgecolor="black",
            alpha=0.6,
            label="Total PV multiple",
        )
        ax3.axvline(1.0, color="blue", linestyle="-", label="1.0× initial cash")
        ax3.axvline(
            np.median(total_multiples),
            color="green",
            linestyle="--",
            label=f"Total median {np.median(total_multiples):.2f}×",
        )
        ax3.set_title("PV multiples (total vs investment only)")
        ax3.set_xlabel("Multiple of initial outlay (×)")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        st.pyplot(fig3)

    # right2: ECDF
    with col_right2:
        sorted_pv = np.sort(final_portfolio_pv)
        ecdf_y = np.arange(1, len(sorted_pv) + 1) / len(sorted_pv)
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        ax4.plot(sorted_pv, ecdf_y, drawstyle="steps-post", label="ECDF of PV")
        ax4.axvline(initial_outlay, color="blue", linestyle="-", label="Initial outlay")
        ax4.set_title("ECDF of PV of ending portfolio value")
        ax4.set_xlabel("£ (present value)")
        ax4.set_ylabel("Cumulative probability")
        ax4.legend()
        st.pyplot(fig4)

else:
    st.info("Set your parameters in the sidebar and click **Run simulation**.")
