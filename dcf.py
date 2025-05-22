from alpha_vantage.fundamentaldata import FundamentalData
import requests
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
from dotenv import load_dotenv
import os

load_dotenv()  

## Constants (Can be adjusted)
RISK_FREE_RATE = 0.039  # 10-year Treasury yield (3.9%, as of May 2025, web:12)
MARKET_RISK_PREMIUM = 0.055  # Historical average (5.5%)
TERMINAL_GROWTH_RATE = 0.03  # 3%

# === Setup ===
ticker = "AAPL"
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")
av_client = FundamentalData(key=ALPHA_VANTAGE_API_KEY)

# === Get Alpha Vantage Data ===
overview, _ = av_client.get_company_overview(ticker)
income_statement, _ = av_client.get_income_statement_annual(ticker)
balance_sheet, _ = av_client.get_balance_sheet_annual(ticker)
cash_flow, _ = av_client.get_cash_flow_annual(ticker)


def safe_float(value, default=0.0):
    """Convert to float safely, handling None and missing fields."""
    try:
        if value is None or str(value).lower() == "none":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


# Financial Metrics 
income_data = income_statement.iloc[0]
cash_data = cash_flow.iloc[0]
balance_data = balance_sheet.iloc[0]
revenue = safe_float(income_data.get("totalRevenue"))
operating_income = safe_float(income_data.get("operatingIncome"))
ocf = safe_float(cash_data.get("operatingCashflow"))
capex = safe_float(cash_data.get("capitalExpenditures"))
fcf = ocf - capex if capex < 0 else ocf - abs(capex)
short_term_debt = safe_float(balance_data.get("shortTermDebt"))
long_term_debt = safe_float(balance_data.get("longTermDebt"))
cash = safe_float(balance_data.get("cashAndCashEquivalentsAtCarryingValue"))
net_debt = short_term_debt + long_term_debt - cash
shares_outstanding = safe_float(overview.get("SharesOutstanding", 1e9))
market_cap = safe_float(overview.get("MarketCapitalization"))
current_price = market_cap / shares_outstanding if shares_outstanding > 0 else 0.0

# === Get Growth Estimate from FMP ===
def fetch_fcf_growth_and_peg(ticker):
    """Fetch historical FCF growth and PEG ratio from Alpha Vantage."""
    try:
        overview, _ = av_client.get_company_overview(symbol=ticker)
        peg_ratio = float(overview.get("PEGRatio", 2.5))  # Default 2.5
        forward_pe = float(overview.get("ForwardPE", 20.0))  # Default 20
        cash_flow, _ = av_client.get_cash_flow_annual(symbol=ticker)
        
        # Calculate historical FCF growth
        fcf_values = [float(report["freeCashFlow"]) for report in cash_flow[:5] if "freeCashFlow" in report]
        fcf_growth_rates = [(fcf_values[i] / fcf_values[i+1] - 1) for i in range(len(fcf_values)-1)]
        avg_fcf_growth = np.mean(fcf_growth_rates) if fcf_growth_rates else 0.1
        
        # Infer 1-5 year EPS growth from PEG
        eps_growth_1to5 = forward_pe / (peg_ratio * 100) if peg_ratio > 0 else 0.07
        return avg_fcf_growth, eps_growth_1to5
    except Exception as e:
        print(f"Alpha Vantage error: {e}")
        return 0.1, 0.07  # Fallback: 10% FCF, 7% EPS

def fetch_growth_estimate(ticker):
    """Fetch analyst revenue growth estimates from FMP."""
    try:
        # Analyst estimates
        estimates_url = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{ticker}?apikey={FMP_API_KEY}"
        estimates_data = requests.get(estimates_url).json()
        revenue_growth_1to5 = []
        for estimate in estimates_data[:5]:  # Next 5 years
            if "estimatedRevenueAvg" in estimate and "estimatedRevenueAvg" in estimates_data[0]:
                growth = (estimate["estimatedRevenueAvg"] / estimates_data[0]["estimatedRevenueAvg"] - 1) / estimate["year"]
                revenue_growth_1to5.append(growth)
        avg_revenue_growth_1to5 = np.mean(revenue_growth_1to5) if revenue_growth_1to5 else 0.06
        
        # Historical revenue growth
        income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=10&apikey={FMP_API_KEY}"
        income_data = requests.get(income_url).json()
        revenues = [data["revenue"] for data in income_data[:10]]
        revenue_growth_rates = [(revenues[i] / revenues[i+1] - 1) for i in range(len(revenues)-1)]
        
        return avg_revenue_growth_1to5
    except Exception as e:
        print(f"FMP error: {e}")
        return 0.06, 0.08  # Fallback: 6% analyst, 8% historical

def interpolate_growth_rates(growth_1to5, terminal_growth, years=5):
    """Interpolate growth rates for years 6-10."""
    return np.linspace(growth_1to5, terminal_growth, years + 2)[1:-1]  # Exclude start/end


# === DCF ===
def fetch_market_value_equity(ticker):
    """Fetch beta and market value of equity from Alpha Vantage."""
    try:
        overview, _ = av_client.get_company_overview(symbol=ticker)
        beta = float(overview.get("Beta", 1.2))  # Default to 1.2 if missing
        shares_outstanding = int(overview.get("SharesOutstanding", 1e9))
        current_price = float(overview.get("PreviousClose", 200.0))  # Fallback price
        market_value_equity = current_price * shares_outstanding
        return beta, market_value_equity
    except Exception as e:
        print(f"Alpha Vantage error: {e}")
        return 1.2, 200.0 * 1e9  # Fallback values

def fetch_debt_expense_taxrate(ticker):
    """Fetch total debt, interest expense, and tax rate using Alpha Vantage."""
    try:
        # Balance Sheet endpoint
        balance_url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        balance_data = requests.get(balance_url).json()
        latest_balance = balance_data["annualReports"][0]  # Most recent report

        short_term_debt = safe_float(latest_balance.get("shortTermDebt", 0))
        long_term_debt = safe_float(latest_balance.get("longTermDebt", 0))
        total_debt = short_term_debt + long_term_debt

        # Income Statement endpoint
        income_url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        income_data = requests.get(income_url).json()
        latest_income = income_data["annualReports"][0]

        interest_expense = abs(safe_float(latest_income.get("interestExpense", 0)))
        taxes_paid = safe_float(latest_income.get("incomeTaxExpense", 0))
        pre_tax_income = safe_float(latest_income.get("incomeBeforeTax", 0))

        return total_debt, interest_expense, taxes_paid, pre_tax_income

    except Exception as e:
        print(f"Alpha Vantage error: {e}")
        return 1e9, 1e7, 1e7, 5e7  # fallback values

def calculate_wacc(ticker):
    """Calculate WACC to be used as discount rate."""
 
    beta, market_value_equity = fetch_market_value_equity(ticker)
    total_debt, interest_expense, taxes_paid, pre_tax_income = fetch_debt_expense_taxrate(ticker)

    # Cost of Equity (CAPM)
    cost_of_equity = RISK_FREE_RATE + beta * MARKET_RISK_PREMIUM

    # After-Tax Cost of Debt
    interest_rate = interest_expense / total_debt if total_debt > 0 else 0.03  # Default 3%
    tax_rate = taxes_paid / pre_tax_income if pre_tax_income > 0 else 0.21  # Default 21%
    after_tax_cost_of_debt = interest_rate * (1 - tax_rate)

    # Capital Structure Weights
    total_value = market_value_equity + total_debt
    weight_equity = market_value_equity / total_value if total_value > 0 else 0.8
    weight_debt = total_debt / total_value if total_value > 0 else 0.2

    # WACC Calculation
    wacc = (weight_equity * cost_of_equity) + (weight_debt * after_tax_cost_of_debt)

    return {
        "ticker": ticker,
        "wacc": wacc,
        "cost_of_equity": cost_of_equity,
        "after_tax_cost_of_debt": after_tax_cost_of_debt,
        "weight_equity": weight_equity,
        "weight_debt": weight_debt,
        "beta": beta,
        "tax_rate": tax_rate,
        "market_value_equity": market_value_equity,
        "total_debt": total_debt
    }

def calculate_dcf(ticker, fcf, growth_rates, wacc, terminal_growth, years=10):
    """Calculate DCF intrinsic value."""
    cash_flows = [fcf * np.prod([(1 + g) for g in growth_rates[:t]]) for t in range(1, years + 1)]
    terminal_value = cash_flows[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    discounted_cash_flows = [cf / (1 + wacc) ** t for t, cf in enumerate(cash_flows, 1)]
    discounted_terminal = terminal_value / (1 + wacc) ** years
    return sum(discounted_cash_flows) + discounted_terminal


# === Main Execution ===
if __name__ == "__main__":
    wacc = calculate_wacc(ticker)['wacc']
    perpetual_growth_rate = 0.02
    years = 10
    avg_fcf_growth, eps_growth_1to5 = fetch_fcf_growth_and_peg(ticker)
    avg_revenue_growth_1to5 = fetch_growth_estimate(ticker)
    growth_1to5 = (avg_revenue_growth_1to5 + eps_growth_1to5 + avg_fcf_growth) / 3
    growth_6to10 = interpolate_growth_rates(growth_1to5, TERMINAL_GROWTH_RATE)
    all_growth_rates = [growth_1to5] * 5 + list(growth_6to10)

    wacc_info = calculate_wacc(ticker)
    wacc = wacc_info["wacc"]

    dcf_value = calculate_dcf(ticker, fcf, all_growth_rates, wacc, TERMINAL_GROWTH_RATE)
    intrinsic_value = (dcf_value - net_debt) / shares_outstanding
    print(f"Intrinsic value per share for {ticker}: ${intrinsic_value:.2f}")
    print(f"Current Price: ${current_price:.2f}")
    if intrinsic_value > current_price:
        print("Undervalued")
    else:
        print("Overvalued")


