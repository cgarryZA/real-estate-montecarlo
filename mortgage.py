# mortgage.py

def annuity_payment(principal: float, annual_rate: float, term_years: int, steps_per_year: int):
    """
    Calculate the fixed annuity payment for a mortgage.

    :param principal: The total loan amount (principal).
    :param annual_rate: The annual interest rate (as a decimal, e.g., 0.05 for 5%).
    :param term_years: The term of the loan in years.
    :param steps_per_year: The number of payment periods per year (e.g., 12 for monthly payments).
    :return: The fixed annuity payment amount.
    """
    r = annual_rate / steps_per_year   # periodic interest rate
    n = term_years * steps_per_year    # total number of payments

    if r == 0:
        return principal / n           # No interest case

    payment = principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    return payment

def step_mortgage(balance: float, payment: float, annual_rate: float, steps_per_year: int):
    """
    Perform a single mortgage payment step.

    :param balance: The current balance of the mortgage.
    :param payment: The fixed payment amount.
    :param annual_rate: The annual interest rate (as a decimal).
    :param steps_per_year: The number of payment periods per year.
    :return: A tuple containing the new balance, interest paid, and principal paid.
    """
    r = annual_rate / steps_per_year   # periodic interest rate
    interest = balance * r
    principal = payment - interest
    new_balance = balance - principal

    return new_balance, interest, principal