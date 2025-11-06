# main.py

import numpy as np
import matplotlib.pyplot as plt

from simulate import (
    run_mc,              # keep if you still want the simple version
    run_mc_with_paths,   # <-- NEW: returns time-series per path
    compute_initial_outlay,
)
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


def maybe_kde(ax, data, x_min, x_max, color="black", label="KDE"):
    """Plot a gaussian KDE on top of ax if scipy is available."""
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return  # silently skip if scipy not installed

    kde = gaussian_kde(data)
    xs = np.linspace(x_min, x_max, 300)
    ys = kde(xs)

    # scale KDE so it roughly matches histogram scale (density -> counts)
    patches = ax.patches
    if not patches:
        return
    bin_width = patches[0].get_width()
    n_total = len(data)
    ys_scaled = ys * n_total * bin_width

    ax.plot(xs, ys_scaled, color=color, linewidth=1.5, label=label)


def main():
    # --- params ---
    prop = PropertyParams(
        initial_price=295_000,
        price_drift=0.03,
        price_vol=0.12,
        # annual rent 29,536 -> monthly, then minus 4 tenants * £20/wk * 4 wks
        rent_initial=(29_536 / 12 - 4 * 20 * 4),
        rent_drift=0.02,
        rent_vol=0.08,
        expense_ratio=0.1,
    )

    mort = MortgageParams(
        initial_ltv=0.85,
        rate=0.04,
        term_years=35,
    )

    refi = RefiParams(
        max_ltv=0.75,
        refi_fee_ratio=0.02,
        rate_spread=0.005,
        decision_interval_years=2,
    )

    inv = InvestmentParams(
        exp_return=0.07,
        vol=0.15,
    )

    sim = SimulationParams(
        years=15,
        steps_per_year=12,
        n_paths=5000,
    )

    sdlt = StampDutyParams(
        bands=[
            (0, 250_000, 0.0),
            (250_000, 925_000, 0.05),
            (925_000, 1_500_000, 0.10),
            (1_500_000, float("inf"), 0.12),
        ],
        surcharge=0.03,
    )

    acq = AcquisitionCosts(
        other_fixed=600,
        searches_fixed=400,
        solicitor_pct_of_price=0.001,
        mortgage_fee_pct_of_loan=0.005,
    )

    # =====================================================================
    # RUN MC WITH PATHS (so we can average equity/investment/wealth)
    # =====================================================================
    mc = run_mc_with_paths(prop, mort, refi, inv, sim, acq, sdlt)
    final_wealths = mc["finals"]
    equity_paths = mc["equity_paths"]       # shape (n_paths, n_steps)
    inv_paths = mc["inv_paths"]             # shape (n_paths, n_steps)
    wealth_paths = mc["wealth_paths"]       # shape (n_paths, n_steps)

    # initial outlay (day 0)
    initial_outlay = compute_initial_outlay(prop, mort, acq, sdlt)

    # discount to present value
    T = sim.years
    discount_rate = 0.03
    df = (1 + discount_rate) ** T
    final_wealths_pv = final_wealths / df

    stats = summarize(final_wealths)
    stats_pv = summarize(final_wealths_pv)

    print("Simulation Summary Statistics:")
    print("Raw Values:")
    print(stats)
    print("Discounted Values:")
    print(stats_pv)

    # =====================================================================
    # NEW PLOT: average equity vs investment pot vs total wealth over time
    # =====================================================================
    # average across paths -> shape (n_steps,)
    equity_mean = equity_paths.mean(axis=0)
    inv_mean = inv_paths.mean(axis=0)
    wealth_mean = wealth_paths.mean(axis=0)

    # time axis in years
    n_steps = sim.years * sim.steps_per_year
    t = np.arange(n_steps) / sim.steps_per_year

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, equity_mean, label="Mean equity in property")
    ax.plot(t, inv_mean, label="Mean investment pot")
    ax.plot(t, wealth_mean, label="Mean total wealth", linestyle="--", alpha=0.7)
    ax.set_xlabel("Years")
    ax.set_ylabel("£")
    ax.set_title("Average path of equity vs investment pot (across MC sims)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # =====================================================================
    # 1) HISTOGRAMS (raw + discounted) WITH OPTIONAL KDE
    # =====================================================================
    # percentiles to zoom (chop silly right tail)
    raw_lo = float(np.percentile(final_wealths, 0))
    raw_hi = float(np.percentile(final_wealths, 98))
    pv_lo = float(np.percentile(final_wealths_pv, 0))
    pv_hi = float(np.percentile(final_wealths_pv, 98))

    # bins ~ sqrt(N)
    bins = int(np.sqrt(sim.n_paths))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_raw, ax_pv = axes

    # ---------- RAW ----------
    n, raw_bins, patches = ax_raw.hist(final_wealths, bins=bins, edgecolor="black")

    for left, right, patch in zip(raw_bins[:-1], raw_bins[1:], patches):
        if left <= initial_outlay <= right:
            patch.set_facecolor("blue")
        elif right < initial_outlay:
            patch.set_facecolor("red")
        else:
            patch.set_facecolor("green")

    ax_raw.axvline(stats["mean"], color="red", linestyle="--", label=f"Mean £{stats['mean']:.0f}")
    ax_raw.axvline(
        stats["median"], color="green", linestyle="--", label=f"Median £{stats['median']:.0f}"
    )
    ax_raw.axvline(
        initial_outlay, color="blue", linestyle="-", label=f"Initial outlay £{initial_outlay:.0f}"
    )

    ax_raw.set_xlim(raw_lo, raw_hi)
    ax_raw.set_xlabel("Final Wealth Net of Entry Costs (£)")
    ax_raw.set_ylabel("Frequency")
    ax_raw.set_title("Raw Terminal Wealth (15y)")

    # add KDE on top (if scipy present)
    maybe_kde(ax_raw, final_wealths, raw_lo, raw_hi, label="KDE")

    ax_raw.legend()

    # ---------- DISCOUNTED ----------
    n, pv_bins, patches = ax_pv.hist(final_wealths_pv, bins=bins, edgecolor="black")

    for left, right, patch in zip(pv_bins[:-1], pv_bins[1:], patches):
        if left <= initial_outlay <= right:
            patch.set_facecolor("blue")
        elif right < initial_outlay:
            patch.set_facecolor("red")
        else:
            patch.set_facecolor("green")

    ax_pv.axvline(
        stats_pv["mean"], color="red", linestyle="--", label=f"Mean £{stats_pv['mean']:.0f}"
    )
    ax_pv.axvline(
        stats_pv["median"],
        color="green",
        linestyle="--",
        label=f"Median £{stats_pv['median']:.0f}",
    )
    ax_pv.axvline(
        initial_outlay, color="blue", linestyle="-", label=f"Initial outlay £{initial_outlay:.0f}"
    )

    ax_pv.set_xlim(pv_lo, pv_hi)
    ax_pv.set_xlabel("PV of Final Wealth (£)")
    ax_pv.set_ylabel("Frequency")
    ax_pv.set_title("Discounted to Present (3%)")

    maybe_kde(ax_pv, final_wealths_pv, pv_lo, pv_hi, label="KDE")

    ax_pv.legend()

    plt.tight_layout()
    plt.show()

    # =====================================================================
    # 2) ECDF for DISCOUNTED WEALTH
    # =====================================================================
    sorted_pv = np.sort(final_wealths_pv)
    ecdf_y = np.arange(1, len(sorted_pv) + 1) / len(sorted_pv)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sorted_pv, ecdf_y, drawstyle="steps-post", label="ECDF (PV wealth)")

    # line at initial outlay
    ax.axvline(initial_outlay, color="blue", linestyle="-", label="Initial outlay")
    # maybe show p05, median, p95
    p05 = np.percentile(final_wealths_pv, 5)
    p50 = np.percentile(final_wealths_pv, 50)
    p95 = np.percentile(final_wealths_pv, 95)
    ax.axvline(p05, color="red", linestyle=":", label=f"5% £{p05:,.0f}")
    ax.axvline(p50, color="green", linestyle="--", label=f"50% £{p50:,.0f}")
    ax.axvline(p95, color="orange", linestyle=":", label=f"95% £{p95:,.0f}")

    ax.set_xlabel("PV of Final Wealth (£)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("ECDF of PV Terminal Wealth")
    ax.set_xlim(pv_lo, pv_hi)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # =====================================================================
    # 3) NORMALIZED: PV MULTIPLE OF INITIAL OUTLAY
    # =====================================================================
    multiples = final_wealths_pv / initial_outlay

    fig, ax = plt.subplots(figsize=(7, 5))
    m_bins = int(np.sqrt(sim.n_paths))
    n, m_bins, patches = ax.hist(multiples, bins=m_bins, edgecolor="black", color="lightgreen")

    ax.axvline(1.0, color="blue", linestyle="-", label="1.0× initial cash")

    m_p50 = np.median(multiples)
    ax.axvline(m_p50, color="green", linestyle="--", label=f"Median {m_p50:.2f}×")

    ax.set_xlabel("PV multiple of initial outlay (×)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of PV Multiples")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
