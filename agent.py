"""
agent.py
import warnings
warnings.filterwarnings('ignore')
Core decision engine — imported by app.py (Streamlit).
Can also be run standalone: python agent.py
"""

import json
import joblib
import numpy as np

model         = joblib.load('best_model.pkl')
scaler        = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

try:
    with open('model_results.json') as f:
        meta = json.load(f)
    BEST_MODEL_NAME = meta.get('best_model', 'Best Model')
    MODEL_RESULTS   = meta.get('results', {})
except Exception:
    BEST_MODEL_NAME = 'Best Model'
    MODEL_RESULTS   = {}

# ── THRESHOLDS ────────────────────────────────────────────────────
APPROVAL_THRESHOLD  = 0.40
REJECTION_THRESHOLD = 0.60
GREY_ZONE_MIN_SCORE = 4

# ── MITIGATING FACTORS ────────────────────────────────────────────
MITIGATING_FACTORS = [
    {'key': 'BankLoyaltyScore',        'condition': lambda v: v >= 5,      'points': 2, 'label': 'Long-term banking relationship (5+ years)'},
    {'key': 'UtilityPaymentLatency',   'condition': lambda v: v == 0,      'points': 2, 'label': 'Perfect utility bill payment record'},
    {'key': 'OverdraftFrequency',      'condition': lambda v: v <= 1,      'points': 2, 'label': 'Very low overdraft frequency'},
    {'key': 'SocialSentimentScore',    'condition': lambda v: v >= 70,     'points': 1, 'label': 'Positive financial sentiment score (70+)'},
    {'key': 'TransactionConsistency',  'condition': lambda v: v >= 75,     'points': 1, 'label': 'High transaction consistency score'},
    {'key': 'EmploymentLength',        'condition': lambda v: v >= 5,      'points': 1, 'label': 'Stable employment (5+ years)'},
    {'key': 'Income',                  'condition': lambda v: v >= 60000,  'points': 1, 'label': 'Above-average income stability'},
]

def evaluate_mitigating_factors(profile):
    triggered, total = [], 0
    for f in MITIGATING_FACTORS:
        val = profile.get(f['key'])
        if val is not None and f['condition'](val):
            total += f['points']
            triggered.append({'label': f['label'], 'points': f['points']})
    return total, triggered

def build_feature_vector(profile):
    """Profile dict → numpy array matching training feature order."""
    row = {f: 0 for f in feature_names}
    for k, v in profile.items():
        if k in row:
            row[k] = v
        one_hot = f'{k}_{v}'
        if one_hot in row:
            row[one_hot] = 1
    arr = np.array([[row[f] for f in feature_names]])
    # Logistic Regression was trained on scaled data
    if 'Logistic' in BEST_MODEL_NAME:
        arr = scaler.transform(arr)
    return arr

def generate_letter(name, decision, prob, reasons, conditions=None):
    lines = [
        f"Dear {name},", "",
        "Thank you for your recent loan application. Following a thorough review of your",
        "financial profile and behavioral indicators, we write to inform you of our decision.", "",
    ]
    if decision == 'APPROVE':
        lines += ["DECISION: APPROVED", "",
                  "Your application demonstrated strong creditworthiness. Positive indicators:", ""]
        for r in reasons:
            lines.append(f"  \u2022 {r['label']}")
        lines += ["", "Your loan will be processed at the full requested amount."]
    elif decision == 'CONDITIONAL':
        lines += ["DECISION: CONDITIONALLY APPROVED", "",
                  "Moderate risk indicators were offset by the following mitigating factors:", ""]
        for r in reasons:
            lines.append(f"  \u2022 {r['label']}")
        if conditions:
            lines += ["", "This approval is subject to:", ""]
            for c in conditions:
                lines.append(f"  * {c}")
        lines += ["", "We look forward to supporting your financial goals under these agreed terms."]
    else:
        lines += ["DECISION: DECLINED", "",
                  "After careful review, we are unable to approve your application at this time.",
                  "Primary factors include elevated credit risk and insufficient mitigating signals.", "",
                  "We recommend:", "  \u2022 Reducing outstanding debt", "  \u2022 Maintaining on-time payments for 6+ months",
                  "  \u2022 Limiting new credit inquiries", "",
                  "You are welcome to reapply in 6 months."]
    lines += ["", "Sincerely,", "AI Underwriting Division",
              f"Model: {BEST_MODEL_NAME}  |  P(default): {prob:.1%}"]
    return "\n".join(lines)

def evaluate_borrower(profile):
    name = profile.get('Name', 'Applicant')
    X    = build_feature_vector(profile)
    prob = float(model.predict_proba(X)[0][0])   # index 0 = class 0 = Default

    conditions = []
    if prob < APPROVAL_THRESHOLD:
        zone, decision = 'green', 'APPROVE'
        mit_score, triggered = evaluate_mitigating_factors(profile)
    elif prob > REJECTION_THRESHOLD:
        zone, decision = 'red', 'REJECT'
        mit_score, triggered = 0, []
    else:
        zone = 'amber'
        mit_score, triggered = evaluate_mitigating_factors(profile)
        if mit_score >= GREY_ZONE_MIN_SCORE:
            decision   = 'CONDITIONAL'
            conditions = [
                'Loan amount capped at 80% of requested principal',
                'Quarterly repayment review for first 12 months',
                'Direct debit mandate required',
            ]
        else:
            decision = 'REJECT'

    letter = generate_letter(name, decision, prob, triggered, conditions)
    return {
        'decision': decision, 'prob_default': prob, 'zone': zone,
        'mit_score': mit_score, 'triggered_factors': triggered,
        'conditions': conditions, 'letter': letter, 'model_name': BEST_MODEL_NAME,
    }

# ── STANDALONE TEST ───────────────────────────────────────────────
if __name__ == '__main__':
    cases = [
        ('BORDERLINE — grey zone expected', {
            'Name': 'Ahmed Al-Rashid', 'Age': 38, 'Income': 52000,
            'LoanAmount': 22000, 'CreditScore': 510, 'EmploymentLength': 7,
            'DebtToIncomeRatio': 0.51, 'OpenCreditLines': 4, 'LatePayments': 3,
            'HomeOwnership': 'RENT', 'LoanPurpose': 'debt_consolidation',
            'AppSessionTime': 6.2, 'UtilityPaymentLatency': 0,
            'SocialSentimentScore': 74, 'OverdraftFrequency': 1,
            'NumInquiries': 4, 'BankLoyaltyScore': 9,
            'SupportContactFrequency': 2, 'DeviceType': 'mobile',
            'TransactionConsistency': 80, 'NighttimeTransactionRatio': 0.10,
        }),
        ('STRONG — approve expected', {
            'Name': 'Sarah Chen', 'Age': 45, 'Income': 95000,
            'LoanAmount': 20000, 'CreditScore': 780, 'EmploymentLength': 15,
            'DebtToIncomeRatio': 0.18, 'OpenCreditLines': 6, 'LatePayments': 0,
            'HomeOwnership': 'OWN', 'LoanPurpose': 'home_improvement',
            'AppSessionTime': 12.0, 'UtilityPaymentLatency': 0,
            'SocialSentimentScore': 88, 'OverdraftFrequency': 0,
            'NumInquiries': 1, 'BankLoyaltyScore': 15,
            'SupportContactFrequency': 1, 'DeviceType': 'desktop',
            'TransactionConsistency': 92, 'NighttimeTransactionRatio': 0.05,
        }),
        ('HIGH RISK — reject expected', {
            'Name': 'John Doe', 'Age': 26, 'Income': 28000,
            'LoanAmount': 35000, 'CreditScore': 490, 'EmploymentLength': 1,
            'DebtToIncomeRatio': 0.82, 'OpenCreditLines': 11, 'LatePayments': 5,
            'HomeOwnership': 'RENT', 'LoanPurpose': 'other',
            'AppSessionTime': 2.0, 'UtilityPaymentLatency': 20,
            'SocialSentimentScore': 30, 'OverdraftFrequency': 7,
            'NumInquiries': 9, 'BankLoyaltyScore': 1,
            'SupportContactFrequency': 10, 'DeviceType': 'mobile',
            'TransactionConsistency': 35, 'NighttimeTransactionRatio': 0.50,
        }),
    ]

    for label, profile in cases:
        print(f"\n{'='*55}")
        print(f"TEST: {label}")
        print('='*55)
        r = evaluate_borrower(profile)
        print(f"Decision   : {r['decision']}")
        print(f"P(default) : {r['prob_default']:.1%}")
        print(f"Zone       : {r['zone'].upper()}")
        print(f"Mit. score : {r['mit_score']}")
        print(f"\n{r['letter']}")
