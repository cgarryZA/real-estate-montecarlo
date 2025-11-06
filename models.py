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
    surcharge: float = 0.0

    def calculate(self, price: float) -> float:
        tax = 0.0
        for low, high, rate in self.bands:
            if price > low:
                taxable = min(price, high) - low if math.isfinite(high) else price - low
                tax += taxable * rate
            if self.surcharge > 0:
                tax += price * self.surcharge
        return tax