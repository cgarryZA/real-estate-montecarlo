# simulate.py

import numpy as np
from mortgage import annuity_payment, step_mortgage
from models import (
    PropertyParams,
    MortgageParams,
    RefiParams,
    InvestmentParams,
    SimulationParams,
    AcquisitionCosts,
    StampDutyParams,
)


def compute_initial_outlay(
    prop: PropertyParams,
    mort: MortgageParams,
    acq: AcquisitionCosts = None,
    sdlt: StampDutyParams = None,
) -> float:
    price = prop.initial_price
    loan = price * mort.initial_ltv
    deposit = price * (1.0 - mort.initial_ltv)

    solicitor_fee = price * (acq.solicitor_pct_of_price if acq else 0.0)
    mortgage_fee = loan * (acq.mortgage_fee_pct_of_loan if acq else 0.0)
    searches = acq.searches_fixed if acq else 0.0
    other = acq.other_fixed if acq else 0.0

    stamp_duty = sdlt.calculate(price) if sdlt else 0.0

    return deposit + solicitor_fee + mortgage_fee + searches + other + stamp_duty


def simulate_path(
    prop,
    mort,
    refi,
    inv,
    sim,
    acq: AcquisitionCosts = None,
    sdlt: StampDutyParams = None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    dt = 1 / sim.steps_per_year
    n_steps = int(sim.years * sim.steps_per_year)

    price = prop.initial_price
    rent = prop.rent_initial

    # initial mortgage
    loan0 = prop.initial_price * mort.initial_ltv
    payment = annuity_payment(loan0, mort.rate, mort.term_years, sim.steps_per_year)
    balance = loan0

    initial_outlay = compute_initial_outlay(prop, mort, acq, sdlt)

    inv_value = 0.0
    cash_history = []
    wealth_history = []
    equity_history = []
    inv_history = []

    for step in range(n_steps):
        t_years = step * dt

        # 1) evolve property price
        z_price = rng.normal()
        price *= np.exp(
            (prop.price_drift - 0.5 * prop.price_vol**2) * dt
            + prop.price_vol * np.sqrt(dt) * z_price
        )

        # 2) evolve rent
        z_rent = rng.normal()
        rent *= np.exp(
            (prop.rent_drift - 0.5 * prop.rent_vol**2) * dt
            + prop.rent_vol * np.sqrt(dt) * z_rent
        )

        # 3) rental cashflow
        gross_rent = rent * dt * 12
        opex = gross_rent * prop.expense_ratio

        # 4) mortgage payment
        if balance > 0:
            balance, interest, principal = step_mortgage(
                balance, payment, mort.rate, sim.steps_per_year
            )
            debt_service = payment
        else:
            debt_service = 0.0
            interest = 0.0

        net_cf = gross_rent - opex - debt_service
        if net_cf < 0:
            net_cf = 0.0

        # 5) invest CF
        z_inv = rng.normal()
        inv_return = (
            np.exp(
                (inv.exp_return - 0.5 * inv.vol**2) * dt
                + inv.vol * np.sqrt(dt) * z_inv
            )
            - 1
        )
        inv_value = (inv_value + net_cf) * (1 + inv_return)

        # 6) refinance check
        if refi is not None:
            # only check at decision dates (e.g. yearly)
            is_decision_time = abs((t_years % refi.decision_interval_years)) < 1e-6
            if is_decision_time:
                # leveraged annual growth of equity
                # equity = price - balance
                equity = price - balance
                # guard against division by zero
                if equity > 1e-6:
                    effective_equity_growth = prop.price_drift * (price / equity)
                else:
                    effective_equity_growth = float('inf')

                # compare to investment return (both annual)
                if effective_equity_growth < inv.exp_return:
                    ltv = balance / price
                    if ltv < refi.max_ltv:
                        new_loan = price * refi.max_ltv
                        fee = new_loan * refi.refi_fee_ratio
                        raw_cash_out = new_loan - balance
                        cash_out = max(raw_cash_out - fee, 0.0)

                        inv_value = inv_value + cash_out

                        balance = new_loan
                        new_rate = mort.rate - refi.rate_spread if refi.rate_spread else mort.rate
                        payment = annuity_payment(
                            balance, new_rate, mort.term_years, sim.steps_per_year
                        )

        equity = price - balance
        total_wealth = equity + inv_value

        cash_history.append(net_cf)
        wealth_history.append(total_wealth)
        equity_history.append(equity)
        inv_history.append(inv_value)

    return {
        "initial_outlay": initial_outlay,
        "cash_history": np.array(cash_history),
        "wealth_history": np.array(wealth_history),
        "equity_history": np.array(equity_history),
        "inv_history": np.array(inv_history),
        "final_wealth": wealth_history[-1],
    }


def run_mc(
    prop,
    mort,
    refi,
    inv,
    sim,
    acq: AcquisitionCosts = None,
    sdlt: StampDutyParams = None,
    seed=123,
):
    """Existing simple version: keep for your histograms."""
    rng = np.random.default_rng(seed)
    finals = []
    for _ in range(sim.n_paths):
        res = simulate_path(prop, mort, refi, inv, sim, acq, sdlt, rng)
        finals.append(res["final_wealth"])
    return np.array(finals)


def run_mc_with_paths(
    prop,
    mort,
    refi,
    inv,
    sim,
    acq: AcquisitionCosts = None,
    sdlt: StampDutyParams = None,
    seed=123,
):
    """New version: collect time series for averaging."""
    rng = np.random.default_rng(seed)
    finals = []
    equity_paths = []
    inv_paths = []
    wealth_paths = []

    for _ in range(sim.n_paths):
        res = simulate_path(prop, mort, refi, inv, sim, acq, sdlt, rng)
        finals.append(res["final_wealth"])
        equity_paths.append(res["equity_history"])
        inv_paths.append(res["inv_history"])
        wealth_paths.append(res["wealth_history"])

    finals       = np.array(finals)
    equity_paths = np.vstack(equity_paths)
    inv_paths    = np.vstack(inv_paths)
    wealth_paths = np.vstack(wealth_paths)

    return {
        "finals": finals,
        "equity_paths": equity_paths,
        "inv_paths": inv_paths,
        "wealth_paths": wealth_paths,
    }
