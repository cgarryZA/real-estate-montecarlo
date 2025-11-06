# optimize.py

import numpy as np
import matplotlib.pyplot as plt

from simulate import run_mc_with_paths, compute_initial_outlay
from analyze import path_stats_from_portfolios
from models import (
    PropertyParams,
    MortgageParams,
    RefiParams,
    InvestmentParams,
    SimulationParams,
    AcquisitionCosts,
    StampDutyParams,
)


def main():
    # base params (same as your main, tweak as needed)
    prop = PropertyParams(
        initial_price=295_000,
        price_drift=0.03,
        price_vol=0.12,
        rent_initial=2100.0,
        rent_drift=0.02,
        rent_vol=0.08,
        expense_ratio=0.1,
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
        n_paths=2000,   # can lower to speed up
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

    # LTVs to test
    ltvs = np.linspace(0.55, 0.9, 8)

    all_ann_ret = []
    all_ann_vol = []
    all_sharpe = []

    for ltv in ltvs:
        mort = MortgageParams(
            initial_ltv=ltv,
            rate=0.04,
            term_years=35,
        )

        mc = run_mc_with_paths(
            prop,
            mort,
            refi,
            inv,
            sim,
            acq,
            sdlt,
            corporate_tax_rate=0.19,
        )
        portfolio_paths = mc["portfolio_paths"]

        stats = path_stats_from_portfolios(
            portfolio_paths,
            steps_per_year=sim.steps_per_year,
            risk_free=0.02,  # pick something
        )

        # average across paths for this LTV
        all_ann_ret.append(stats["ann_return"].mean())
        all_ann_vol.append(stats["ann_vol"].mean())
        all_sharpe.append(stats["sharpe"].mean())

    all_ann_ret = np.array(all_ann_ret)
    all_ann_vol = np.array(all_ann_vol)
    all_sharpe = np.array(all_sharpe)

    # find best sharpe
    best_idx = np.argmax(all_sharpe)

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(all_ann_vol, all_ann_ret, c=all_sharpe, cmap="viridis", s=80)
    ax.scatter(all_ann_vol[best_idx], all_ann_ret[best_idx], c="red", s=120, label="Best Sharpe")
    for i, ltv in enumerate(ltvs):
        ax.text(all_ann_vol[i], all_ann_ret[i], f"{ltv:.2f}", fontsize=8)

    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Annualised return")
    ax.set_title("LTV sweep: risk vs return (colour = Sharpe)")
    ax.legend()
    fig.colorbar(sc, label="Sharpe")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
