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


def sharpe_from_finals(final_values, initial, years, rf):
    ratios = np.maximum(final_values / initial, 1e-9)
    log_returns = np.log(ratios) / years  # annualised log return
    mean_ret = np.mean(log_returns)
    std_ret = np.std(log_returns, ddof=1)
    if std_ret <= 1e-12:
        return 0.0
    return (mean_ret - rf) / std_ret


def sharpe_from_paths(paths, rf_annual, steps_per_year):
    """
    More stable Sharpe:
    - takes the whole time-series for every MC path
    - flattens all valid period returns
    - annualises using steps_per_year
    This avoids the 'short horizon' wobble you saw.
    """
    # paths: (n_paths, n_steps) of portfolio levels
    prev = paths[:, :-1]
    curr = paths[:, 1:]

    mask = prev > 0
    period_returns = np.zeros_like(curr, dtype=float)
    period_returns[mask] = curr[mask] / prev[mask] - 1.0

    valid = period_returns[mask]
    if valid.size == 0:
        return 0.0

    mean_r = valid.mean()
    std_r = valid.std(ddof=1)
    if std_r <= 1e-12:
        return 0.0

    rf_per_step = rf_annual / steps_per_year
    # annualise Sharpe
    sharpe = (mean_r - rf_per_step) / std_r * np.sqrt(steps_per_year)
    return float(sharpe)


# =============== SIDEBAR =================
st.sidebar.header("Property & Rent")
price = st.sidebar.number_input("Purchase price (£)", 50_000, 2_000_000, 295_000, 5_000)
price_drift = st.sidebar.slider("Property drift (annual %)", 0.0, 0.08, 0.03, 0.005)
price_vol = st.sidebar.slider("Property vol (annual %)", 0.0, 0.4, 0.12, 0.01)
monthly_rent = st.sidebar.number_input(
    "Monthly rent (£)", 300, 5_000, 1_600, 50
)
rent_drift = st.sidebar.slider("Rent drift (annual %)", 0.0, 0.10, 0.025, 0.005)
rent_vol = st.sidebar.slider("Rent vol (annual %)", 0.0, 0.4, 0.10, 0.01)
expense_ratio = st.sidebar.slider("Operating expense ratio", 0.0, 0.5, 0.25, 0.01)

st.sidebar.header("Mortgage & Refi")
initial_ltv = st.sidebar.slider("Initial LTV", 0.4, 0.95, 0.75, 0.01)
mort_rate = st.sidebar.slider("Mortgage rate (annual %)", 0.0, 0.08, 0.045, 0.001)
term_years = st.sidebar.number_input("Mortgage term (years)", 5, 35, 25, 1)
max_ltv = st.sidebar.slider("Max LTV on refi", 0.5, 0.9, 0.75, 0.01)
refi_fee = st.sidebar.number_input("Refi fee (% of new loan)", 0.0, 0.05, 0.02, 0.001)
rate_spread = st.sidebar.number_input(
    "Refi rate improvement (negative = better)", -0.03, 0.03, -0.005, 0.0005
)
refi_interval = st.sidebar.number_input("Refi decision interval (years)", 1, 10, 2, 1)

st.sidebar.header("Investment side-pot")
inv_return = st.sidebar.slider("Investment exp return (annual %)", 0.0, 0.2, 0.08, 0.005)
inv_vol = st.sidebar.slider("Investment vol (annual %)", 0.0, 0.5, 0.18, 0.01)

st.sidebar.header("Simulation")
years = st.sidebar.number_input("Years", 1, 40, 15, 1)
steps_per_year = st.sidebar.number_input("Steps per year", 1, 12, 4, 1)
n_paths = st.sidebar.number_input("Monte Carlo paths", 100, 10_000, 2_000, 100)

st.sidebar.header("Acquisition costs")
other_fixed = st.sidebar.number_input("Other fixed (£)", 0, 20_000, 2_000, 500)
searches_fixed = st.sidebar.number_input("Searches fixed (£)", 0, 5_000, 350, 50)
solicitor_pct = st.sidebar.number_input("Solicitor (% of price)", 0.0, 0.03, 0.01, 0.001)
mortgage_fee_pct = st.sidebar.number_input(
    "Mortgage fee (% of loan)", 0.0, 0.02, 0.005, 0.0005
)
sdlt_surcharge = st.sidebar.number_input("SDLT surcharge", 0.0, 0.10, 0.05, 0.001)

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
    except Exception as e:
        st.error(f"Simulation failed: {e}")
    else:
        finals = mc["finals"]
        equity_paths = mc["equity_paths"]
        inv_paths = mc["inv_paths"]
        portfolio_paths = mc["portfolio_paths"]

        # actual initial cash out (deposit + fees)
        initial_outlay = compute_initial_outlay(prop, mort, acq, sdlt)
        # deposit only (outlay minus fees)
        deposit_only = price * (1.0 - initial_ltv)

        # discount to PV for comparison
        df = (1 + 0.03) ** years
        final_portfolio = finals
        final_portfolio_pv = final_portfolio / df

        # component finals
        final_equity = equity_paths[:, -1]
        final_invest = inv_paths[:, -1]

        # benchmark: invest the same initial outlay in noisy market
        bench_paths, bench_mean_path = mc_invest_initial_outlay(
            initial_outlay,
            exp_return=inv_return,
            vol=inv_vol,
            years=years,
            steps_per_year=steps_per_year,
            n_paths=n_paths,
        )

        # time axis
        t = np.linspace(0, years, years * steps_per_year)

        # risk free curve
        risk_free_curve = initial_outlay * (1 + rf_rate) ** t

        # Sharpe using full time-series (more stable for short horizons)
        strategy_sharpe = sharpe_from_paths(
            portfolio_paths, rf_rate, steps_per_year
        )
        bench_sharpe = sharpe_from_paths(
            bench_paths, rf_rate, steps_per_year
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

            # plot a few sample strategy paths
            n_show = min(30, portfolio_paths.shape[0])
            for i in range(n_show):
                ax1.plot(t, portfolio_paths[i, :], color="gray", alpha=0.08)

            # plot mean strategy
            ax1.plot(
                t,
                portfolio_paths.mean(axis=0),
                color="black",
                linewidth=2,
                label=f"Strategy mean (Sharpe {strategy_sharpe:.2f})",
            )

            # plot benchmark mean
            ax1.plot(
                t,
                bench_mean_path,
                color="tab:blue",
                linewidth=2,
                label=f"Invest initial (Sharpe {bench_sharpe:.2f})",
            )

            # plot risk free
            ax1.plot(
                t,
                risk_free_curve,
                color="tab:green",
                linestyle="--",
                label="Risk free",
            )

            ax1.set_title("Strategy vs benchmark vs risk-free")
            ax1.set_xlabel("Years")
            ax1.set_ylabel("£")
            ax1.legend()
            ax1.grid(alpha=0.2)
            st.pyplot(fig1)

        # right: distribution of final PVs
        with col_right:
            fig2, ax2 = plt.subplots(figsize=(6, 3.5))
            ax2.hist(final_portfolio_pv, bins=40, alpha=0.7, label="Strategy PV")
            ax2.axvline(
                initial_outlay, color="red", linestyle="--", label="Initial outlay"
            )
            ax2.set_title("Distribution of PV of ending portfolio")
            ax2.set_xlabel("£ (present value)")
            ax2.set_ylabel("Frequency")
            ax2.legend()
            st.pyplot(fig2)

        # second row
        col_left2, col_right2 = st.columns(2)

        # left2: multiples
        with col_left2:
            total_multiples = final_portfolio_pv / initial_outlay
            invest_multiples = final_invest / initial_outlay
            fig3, ax3 = plt.subplots(figsize=(6, 3.5))
            ax3.hist(total_multiples, bins=40, alpha=0.7, label="Total PV/Initial")
            ax3.hist(
                invest_multiples,
                bins=40,
                alpha=0.5,
                label="Investment only / Initial",
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
            ax4.axvline(
                initial_outlay, color="blue", linestyle="-", label="Initial outlay"
            )
            ax4.set_title("ECDF of PV of ending portfolio value")
            ax4.set_xlabel("£ (present value)")
            ax4.set_ylabel("Cumulative probability")
            ax4.legend()
            st.pyplot(fig4)

else:
    st.info("Set your parameters in the sidebar and click **Run simulation**.")
