"""
generate_data.py
Produces a 5,000-row loan dataset with strong ML signal.
Target AUC: 0.80-0.88. Default rate: 28-35%.

Run: python generate_data.py
"""

import numpy as np
import pandas as pd
from numpy.random import default_rng

rng = default_rng(42)
N = 5000

# ── TRADITIONAL FEATURES ──────────────────────────────────────────
age               = rng.integers(21, 66, N)
income            = rng.lognormal(mean=10.9, sigma=0.45, size=N).clip(20000, 300000)
loan_amount       = rng.lognormal(mean=9.1,  sigma=0.40, size=N).clip(3000, 80000)
credit_score      = rng.normal(680, 55, N).clip(300, 850).astype(int)
employment_length = rng.integers(0, 31, N)
debt_to_income_ratio = (loan_amount / income * 12).clip(0.05, 0.95)
open_credit_lines = rng.integers(1, 16, N)
late_payments     = rng.choice([0,1,2,3,4,5], N, p=[0.52,0.20,0.12,0.08,0.05,0.03])
home_ownership    = rng.choice(['OWN','RENT','MORTGAGE'], N, p=[0.28,0.34,0.38])
loan_purpose      = rng.choice(
    ['debt_consolidation','home_improvement','car','medical','education','other'],
    N, p=[0.34,0.20,0.16,0.10,0.10,0.10]
)

# ── BEHAVIORAL FEATURES ───────────────────────────────────────────
app_session_time            = rng.normal(8, 4, N).clip(0, 30)
utility_payment_latency     = rng.choice(
    [0,1,2,3,4,5,10,15,20,30], N,
    p=[0.45,0.15,0.10,0.08,0.06,0.05,0.04,0.03,0.02,0.02]
)
social_sentiment_score      = rng.normal(68, 14, N).clip(0, 100)
overdraft_frequency         = rng.choice(
    [0,1,2,3,4,5,6,7,8], N,
    p=[0.46,0.20,0.12,0.08,0.06,0.04,0.02,0.01,0.01]
)
num_inquiries               = rng.integers(0, 10, N)
bank_loyalty_score          = rng.integers(0, 21, N)
support_contact_frequency   = rng.integers(0, 12, N)
device_type                 = rng.choice(['mobile','desktop'], N, p=[0.72,0.28])
transaction_consistency     = rng.normal(78, 13, N).clip(0, 100)
nighttime_transaction_ratio = rng.normal(0.15, 0.07, N).clip(0, 0.60)

# ── INJECT CORRELATED STRUCTURE ───────────────────────────────────
# Bad-credit borrowers tend to also have bad behavioral signals
# This gives models compound signal to learn, pushing AUC toward 0.85
bad_credit_mask = credit_score < 620
overdraft_frequency         = np.where(bad_credit_mask,
    np.minimum(overdraft_frequency + rng.integers(1,4,N), 8), overdraft_frequency)
late_payments               = np.where(bad_credit_mask,
    np.minimum(late_payments + rng.integers(1,3,N), 5), late_payments)
transaction_consistency     = np.where(bad_credit_mask,
    np.maximum(transaction_consistency - rng.normal(15,5,N), 0), transaction_consistency)
utility_payment_latency     = np.where(bad_credit_mask,
    np.minimum(utility_payment_latency + rng.integers(2,8,N), 30), utility_payment_latency)

good_credit_mask = credit_score > 740
bank_loyalty_score          = np.where(good_credit_mask,
    np.minimum(bank_loyalty_score + rng.integers(2,6,N), 20), bank_loyalty_score)
transaction_consistency     = np.where(good_credit_mask,
    np.minimum(transaction_consistency + rng.normal(10,3,N), 100), transaction_consistency)

# ── NORMALISE ─────────────────────────────────────────────────────
norm_credit      = 1 - (credit_score - 300) / 550
norm_dti         = debt_to_income_ratio
norm_overdraft   = overdraft_frequency / 8
norm_late        = late_payments / 5
norm_utility     = utility_payment_latency / 30
norm_loyalty     = bank_loyalty_score / 20
norm_income      = 1 - (income - 20000) / 280000
norm_social      = 1 - (social_sentiment_score / 100)
norm_transaction = 1 - (transaction_consistency / 100)
norm_night       = nighttime_transaction_ratio / 0.60
norm_inquiries   = num_inquiries / 10

# ── RISK FORMULA — amplified weights for clear signal ─────────────
raw_risk = (
      0.30 * norm_credit          # credit score is king
    + 0.22 * norm_dti             # debt burden
    + 0.18 * norm_overdraft       # cash stress
    + 0.15 * norm_late            # repayment history
    + 0.10 * norm_utility         # bill reliability
    + 0.06 * norm_social
    + 0.05 * norm_transaction
    + 0.04 * norm_night
    + 0.05 * norm_inquiries
    - 0.18 * norm_loyalty         # loyalty is a strong mitigator
    - 0.13 * norm_income
    + rng.normal(0, 0.04, N)      # small noise → realistic but learnable
)

# Aggressive sigmoid: steeper slope means clearer separation at tails
risk = 1 / (1 + np.exp(-10 * (raw_risk - 0.40)))

print("\nRisk statistics:")
print(f"Mean Risk : {risk.mean():.3f}")
print(f"Min Risk  : {risk.min():.3f}")
print(f"Max Risk  : {risk.max():.3f}")

# Probabilistic target — preserves grey zone between clear approvals/rejects
loan_status = (rng.random(N) > risk).astype(int)  # 1=Paid, 0=Default

# ── ASSEMBLE ──────────────────────────────────────────────────────
df = pd.DataFrame({
    'Age': age,
    'Income': income.round(0).astype(int),
    'LoanAmount': loan_amount.round(0).astype(int),
    'CreditScore': credit_score,
    'EmploymentLength': employment_length,
    'DebtToIncomeRatio': debt_to_income_ratio.round(3),
    'OpenCreditLines': open_credit_lines,
    'LatePayments': late_payments,
    'HomeOwnership': home_ownership,
    'LoanPurpose': loan_purpose,
    'AppSessionTime': app_session_time.round(1),
    'UtilityPaymentLatency': utility_payment_latency,
    'SocialSentimentScore': social_sentiment_score.round(1),
    'OverdraftFrequency': overdraft_frequency,
    'NumInquiries': num_inquiries,
    'BankLoyaltyScore': bank_loyalty_score,
    'SupportContactFrequency': support_contact_frequency,
    'DeviceType': device_type,
    'TransactionConsistency': transaction_consistency.round(1),
    'NighttimeTransactionRatio': nighttime_transaction_ratio.round(3),
    'LoanStatus': loan_status
})

df.to_csv('loan_data.csv', index=False)
print(f"\nSaved {len(df)} rows.")
print(f"Default rate : {(1 - df.LoanStatus.mean()):.1%}  (target: 28–35%)")
print(f"Columns      : {list(df.columns)}")
