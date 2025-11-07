# models.py

from dataclasses import dataclass
from typing import List, Tuple
import math

@dataclass
class PropertyParams:
    initial_price: float
    price_drift:    float
    price_vol:      float
    rent_initial:   float
    rent_drift:     float
    rent_vol:       float
    expense_ratio:  float

@dataclass
class MortgageParams:
    rate:        float
    term_years:  int
    initial_ltv: float

@dataclass
class RefiParams:
    max_ltv:                 float
    refi_fee_ratio:          float
    rate_spread:             float
    decision_interval_years: float

@dataclass
class InvestmentParams:
    vol:        float
    exp_return: float

@dataclass
class SimulationParams:
    years:          int
    n_paths:        int
    steps_per_year: int

@dataclass
class AcquisitionCosts:
    other_fixed:              float = 0.0
    searches_fixed:           float = 0.0
    solicitor_pct_of_price:   float = 0.0
    mortgage_fee_pct_of_loan: float = 0.0

@dataclass
class StampDutyParams:
    bands: List[Tuple[float, float, float]]
    surcharge: float = 5.0

    def calculate(self, price: float) -> float:
        """
        Calculate UK-style tiered stamp duty.

        For limited companies, the surcharge is added to each band rate.
        For individuals (surcharge = 0), standard rates apply.
        """
        tax = 0.0
        for low, high, rate in self.bands:
            if price > low:
                upper = price if not math.isfinite(high) else min(price, high)
                taxable = max(0.0, upper - low)
                # add surcharge to the rate itself, not total price
                effective_rate = rate + self.surcharge
                tax += taxable * effective_rate
        return tax
