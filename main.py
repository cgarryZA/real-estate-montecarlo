# main.py

import numpy as np
import matplotlib.pyplot as plt

from simulate import (
    run_mc_with_paths,
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
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return

    kde = gaussian_kde(data)
    xs = np.linspace(x_min, x_max, 300)
    ys = kde(xs)

    patches = ax.patches
    if not patches:
        return
    bin_width = patches[0].get_width()
    n_total = len(data)
    ys_scaled = ys * n_total * bin_width

    ax.plot(xs, ys_scaled, color=color, linewidth=1.5, label=label)


def main():
    prop = PropertyParams(
        initial_price=295_000,
        price_drift=0.03,
        price_vol=0.12,
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

    mc = run_mc_with_paths(prop, mort, refi, inv, sim, acq, sdlt)
    final_portfolios = mc["finals"]
    equity_paths = mc["equity_paths"]
    inv_paths = mc["inv_paths"]
    portfolio_paths = mc["portfolio_paths"]

    initial_outlay = compute_initial_outlay(prop, mort, acq, sdlt)

    T = sim.years
    discount_rate = 0.03
    df = (1 + discount_rate) ** T
    final_portfolios_pv = final_portfolios / df

    stats = summarize(final_portfolios)
    stats_pv = summarize(final_portfolios_pv)

    print("Simulation Summary Statistics:")
    print("Raw Values:")
    print(stats)
    print("Discounted Values:")
    print(stats_pv)

    # mean paths
    n_steps = sim.years * sim.steps_per_year
    t = np.arange(n_steps) / sim.steps_per_year

    equity_mean = equity_paths.mean(axis=0)
    inv_mean = inv_paths.mean(axis=0)
    portfolio_mean = portfolio_paths.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, equity_mean, label="Mean property equity")
    ax.plot(t, inv_mean, label="Mean investment balance")
    ax.plot(t, portfolio_mean, label="Mean portfolio value", linestyle="--", alpha=0.7)
    ax.set_xlabel("Years")
    ax.set_ylabel("£")
    ax.set_title("Average portfolio components over time")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # histograms
    raw_lo = float(np.percentile(final_portfolios, 0))
    raw_hi = float(np.percentile(final_portfolios, 98))
    pv_lo = float(np.percentile(final_portfolios_pv, 0))
    pv_hi = float(np.percentile(final_portfolios_pv, 98))

    bins = int(np.sqrt(sim.n_paths))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_raw, ax_pv = axes

    # raw
    counts, raw_bins, patches = ax_raw.hist(final_portfolios, bins=bins, edgecolor="black")

    for left, right, patch in zip(raw_bins[:-1], raw_bins[1:], patches):
        if left <= initial_outlay <= right:
            patch.set_facecolor("blue")
        elif right < initial_outlay:
            patch.set_facecolor("red")
        else:
            patch.set_facecolor("green")

    ax_raw.axvline(stats["mean"], color="red", linestyle="--", label=f"Mean £{stats['mean']:.0f}")
    ax_raw.axvline(stats["median"], color="green", linestyle="--", label=f"Median £{stats['median']:.0f}")
    ax_raw.axvline(initial_outlay, color="blue", linestyle="-", label=f"Initial outlay £{initial_outlay:.0f}")
    ax_raw.set_xlim(raw_lo, raw_hi)
    ax_raw.set_xlabel("Ending portfolio value (£)")
    ax_raw.set_ylabel("Frequency")
    ax_raw.set_title("Distribution of ending portfolio value")
    maybe_kde(ax_raw, final_portfolios, raw_lo, raw_hi, label="KDE")
    ax_raw.legend()

    # discounted
    counts, pv_bins, patches = ax_pv.hist(final_portfolios_pv, bins=bins, edgecolor="black")
    for left, right, patch in zip(pv_bins[:-1], pv_bins[1:], patches):
        if left <= initial_outlay <= right:
            patch.set_facecolor("blue")
        elif right < initial_outlay:
            patch.set_facecolor("red")
        else:
            patch.set_facecolor("green")

    ax_pv.axvline(stats_pv["mean"], color="red", linestyle="--", label=f"Mean £{stats_pv['mean']:.0f}")
    ax_pv.axvline(stats_pv["median"], color="green", linestyle="--", label=f"Median £{stats_pv['median']:.0f}")
    ax_pv.axvline(initial_outlay, color="blue", linestyle="-", label=f"Initial outlay £{initial_outlay:.0f}")
    ax_pv.set_xlim(pv_lo, pv_hi)
    ax_pv.set_xlabel("PV of ending portfolio value (£)")
    ax_pv.set_ylabel("Frequency")
    ax_pv.set_title("Distribution discounted to present (3%)")
    maybe_kde(ax_pv, final_portfolios_pv, pv_lo, pv_hi, label="KDE")
    ax_pv.legend()

    plt.tight_layout()
    plt.show()

    # ECDF
    sorted_pv = np.sort(final_portfolios_pv)
    ecdf_y = np.arange(1, len(sorted_pv) + 1) / len(sorted_pv)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sorted_pv, ecdf_y, drawstyle="steps-post", label="ECDF (PV)")
    ax.axvline(initial_outlay, color="blue", linestyle="-", label="Initial outlay")
    ax.set_xlabel("PV of ending portfolio value (£)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("ECDF of PV of ending portfolio value")
    ax.set_xlim(pv_lo, pv_hi)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # multiples
    multiples = final_portfolios_pv / initial_outlay
    fig, ax = plt.subplots(figsize=(7, 5))
    m_bins = int(np.sqrt(sim.n_paths))
    ax.hist(multiples, bins=m_bins, edgecolor="black", color="lightgreen")
    ax.axvline(1.0, color="blue", linestyle="-", label="1.0× initial cash")
    median_mult = np.median(multiples)
    ax.axvline(median_mult, color="green", linestyle="--", label=f"Median {median_mult:.2f}×")
    ax.set_xlabel("PV multiple of initial outlay (×)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of PV multiples")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
