"""
app.py  —  AI Credit Scoring Agent
Streamlit UI for the loan decision engine.
Run: streamlit run app.py
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from agent import evaluate_borrower, MITIGATING_FACTORS, MODEL_RESULTS, BEST_MODEL_NAME

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Credit Scoring Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .stApp { font-family: 'Inter', sans-serif; }

    .decision-approve {
        background: linear-gradient(135deg, #DCFCE7, #BBF7D0);
        border-left: 6px solid #16A34A;
        padding: 1.2rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .decision-conditional {
        background: linear-gradient(135deg, #FEF9C3, #FDE68A);
        border-left: 6px solid #D97706;
        padding: 1.2rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }
    .decision-reject {
        background: linear-gradient(135deg, #FEE2E2, #FECACA);
        border-left: 6px solid #DC2626;
        padding: 1.2rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
    }

    .metric-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-label { font-size: 0.78rem; color: #64748B; margin-bottom: 4px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #1E293B; }

    .factor-hit {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        margin: 4px 0;
        font-size: 0.88rem;
        color: #166534;
    }
    .factor-miss {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        margin: 4px 0;
        font-size: 0.88rem;
        color: #94A3B8;
    }
    .zone-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #334155;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR — MODEL PERFORMANCE ───────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 AI Credit Scoring")
    st.markdown("*Hybrid behavioral + financial underwriting*")
    st.divider()

    if MODEL_RESULTS:
        st.markdown("### 📊 Model Performance")
        for name, r in MODEL_RESULTS.items():
            is_best = (name == BEST_MODEL_NAME)
            label = f"{'⭐ ' if is_best else ''}{name}"
            with st.expander(label, expanded=is_best):
                c1, c2 = st.columns(2)
                c1.metric("AUC-ROC", f"{r['AUC']:.3f}")
                c2.metric("FPR", f"{r['FPR']:.1%}")
                c1.metric("Profit", f"${r['Profit']:,.0f}")
                c2.metric("Accuracy", f"{r['Accuracy']:.1%}")
                if is_best:
                    st.success("Active model")
        st.divider()

    st.markdown("### ⚖️ Decision thresholds")
    st.markdown("""
    | Zone | P(default) | Action |
    |------|-----------|--------|
    | 🟢 Green | < 40% | Approve |
    | 🟡 Amber | 40–60% | Grey zone |
    | 🔴 Red | > 60% | Reject |
    """)
    st.divider()
    st.markdown("### 🎯 Grey zone logic")
    st.markdown("Mitigating score ≥ 4 → Conditional approval")
    for f in MITIGATING_FACTORS:
        st.markdown(f"- **+{f['points']}** {f['label'][:40]}...")

    st.divider()
    st.caption(f"Active model: **{BEST_MODEL_NAME}**")


# ── MAIN LAYOUT ───────────────────────────────────────────────────
st.title("🏦 AI Loan Decision Engine")
st.markdown("Enter a borrower profile below. The agent predicts default probability, applies grey-zone analysis, and generates a formal decision letter.")
st.divider()

# ── QUICK LOAD TEST CASES ─────────────────────────────────────────
st.markdown("#### 📋 Quick load a test case")
col_a, col_b, col_c, col_d = st.columns(4)

preset = None
if col_a.button("🟡 Borderline Ahmed", use_container_width=True):
    preset = 'borderline'
if col_b.button("🟢 Strong Sarah", use_container_width=True):
    preset = 'strong'
if col_c.button("🔴 High-risk John", use_container_width=True):
    preset = 'highrisk'
if col_d.button("🔄 Clear form", use_container_width=True):
    preset = 'clear'

PRESETS = {
    'borderline': {
        'name': 'Ahmed Al-Rashid', 'age': 38, 'income': 52000,
        'loan_amount': 18000, 'credit_score': 598, 'employment_len': 7,
        'dti': 0.42, 'open_lines': 4, 'late_payments': 2,
        'home_ownership': 'RENT', 'loan_purpose': 'debt_consolidation',
        'app_session': 6.2, 'utility_latency': 0, 'social_sentiment': 74.0,
        'overdraft_freq': 1, 'num_inquiries': 3, 'bank_loyalty': 9,
        'support_contacts': 2, 'device_type': 'mobile',
        'txn_consistency': 80.0, 'nighttime_ratio': 0.10,
    },
    'strong': {
        'name': 'Sarah Chen', 'age': 45, 'income': 95000,
        'loan_amount': 20000, 'credit_score': 780, 'employment_len': 15,
        'dti': 0.18, 'open_lines': 6, 'late_payments': 0,
        'home_ownership': 'OWN', 'loan_purpose': 'home_improvement',
        'app_session': 12.0, 'utility_latency': 0, 'social_sentiment': 88.0,
        'overdraft_freq': 0, 'num_inquiries': 1, 'bank_loyalty': 15,
        'support_contacts': 1, 'device_type': 'desktop',
        'txn_consistency': 92.0, 'nighttime_ratio': 0.05,
    },
    'highrisk': {
        'name': 'John Doe', 'age': 26, 'income': 28000,
        'loan_amount': 35000, 'credit_score': 490, 'employment_len': 1,
        'dti': 0.82, 'open_lines': 11, 'late_payments': 5,
        'home_ownership': 'RENT', 'loan_purpose': 'other',
        'app_session': 2.0, 'utility_latency': 20, 'social_sentiment': 30.0,
        'overdraft_freq': 7, 'num_inquiries': 9, 'bank_loyalty': 1,
        'support_contacts': 10, 'device_type': 'mobile',
        'txn_consistency': 35.0, 'nighttime_ratio': 0.50,
    },
    'clear': {
        'name': '', 'age': 35, 'income': 50000, 'loan_amount': 15000,
        'credit_score': 650, 'employment_len': 5, 'dti': 0.30,
        'open_lines': 5, 'late_payments': 0,
        'home_ownership': 'RENT', 'loan_purpose': 'debt_consolidation',
        'app_session': 8.0, 'utility_latency': 2, 'social_sentiment': 65.0,
        'overdraft_freq': 1, 'num_inquiries': 2, 'bank_loyalty': 3,
        'support_contacts': 2, 'device_type': 'mobile',
        'txn_consistency': 70.0, 'nighttime_ratio': 0.15,
    },
}

def sv(key, default):
    """Session value — use preset if just selected, else default."""
    if preset and preset != 'clear':
        return PRESETS[preset].get(key, default)
    elif preset == 'clear':
        return PRESETS['clear'].get(key, default)
    return default

st.divider()

# ── BORROWER FORM ─────────────────────────────────────────────────
with st.form("borrower_form"):
    st.markdown("### 👤 Borrower profile")

    # — Applicant name —
    name_val = PRESETS[preset][  'name'] if preset and preset in PRESETS else ""
    applicant_name = st.text_input("Applicant name", value=name_val, placeholder="Full name")

    st.markdown("#### 💰 Financial indicators")
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)

    age_v          = PRESETS[preset]['age']           if preset and preset in PRESETS else 35
    income_v       = PRESETS[preset]['income']        if preset and preset in PRESETS else 50000
    loan_v         = PRESETS[preset]['loan_amount']   if preset and preset in PRESETS else 15000
    cs_v           = PRESETS[preset]['credit_score']  if preset and preset in PRESETS else 650

    age          = r1c1.number_input("Age",           min_value=18, max_value=75, value=age_v)
    income       = r1c2.number_input("Income ($)",    min_value=10000, max_value=500000, step=1000, value=income_v)
    loan_amount  = r1c3.number_input("Loan amount ($)", min_value=1000, max_value=100000, step=500, value=loan_v)
    credit_score = r1c4.number_input("Credit score",  min_value=300, max_value=850, value=cs_v)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    el_v   = PRESETS[preset]['employment_len']  if preset and preset in PRESETS else 5
    dti_v  = PRESETS[preset]['dti']             if preset and preset in PRESETS else 0.30
    ol_v   = PRESETS[preset]['open_lines']      if preset and preset in PRESETS else 5
    lp_v   = PRESETS[preset]['late_payments']   if preset and preset in PRESETS else 0

    employment_len = r2c1.number_input("Employment length (yrs)", 0, 40, value=el_v)
    dti            = r2c2.number_input("Debt-to-income ratio",    0.0, 1.0, step=0.01, value=float(dti_v))
    open_lines     = r2c3.number_input("Open credit lines",       1, 20, value=ol_v)
    late_payments  = r2c4.number_input("Late payments (2yr)",     0, 10, value=lp_v)

    r3c1, r3c2 = st.columns(2)
    ho_opts = ['RENT', 'OWN', 'MORTGAGE']
    lp_opts = ['debt_consolidation','home_improvement','car','medical','education','other']
    ho_v    = PRESETS[preset]['home_ownership'] if preset and preset in PRESETS else 'RENT'
    lpur_v  = PRESETS[preset]['loan_purpose']   if preset and preset in PRESETS else 'debt_consolidation'
    home_ownership = r3c1.selectbox("Home ownership", ho_opts,  index=ho_opts.index(ho_v))
    loan_purpose   = r3c2.selectbox("Loan purpose",   lp_opts, index=lp_opts.index(lpur_v))

    st.markdown("#### 📱 Behavioral indicators")
    b1, b2, b3, b4, b5 = st.columns(5)

    as_v   = PRESETS[preset]['app_session']       if preset and preset in PRESETS else 8.0
    ul_v   = PRESETS[preset]['utility_latency']   if preset and preset in PRESETS else 2
    ss_v   = PRESETS[preset]['social_sentiment']  if preset and preset in PRESETS else 65.0
    of_v   = PRESETS[preset]['overdraft_freq']    if preset and preset in PRESETS else 1
    ni_v   = PRESETS[preset]['num_inquiries']     if preset and preset in PRESETS else 2

    app_session_time   = b1.number_input("App session (min/day)", 0.0, 30.0, value=float(as_v), step=0.5)
    utility_latency    = b2.number_input("Utility latency (days)", 0, 30, value=ul_v)
    social_sentiment   = b3.number_input("Social sentiment (0–100)", 0.0, 100.0, value=float(ss_v), step=0.5)
    overdraft_freq     = b4.number_input("Overdraft freq (12mo)", 0, 12, value=of_v)
    num_inquiries      = b5.number_input("Credit inquiries (6mo)", 0, 12, value=ni_v)

    b6, b7, b8, b9, b10 = st.columns(5)

    bl_v   = PRESETS[preset]['bank_loyalty']      if preset and preset in PRESETS else 3
    sc_v   = PRESETS[preset]['support_contacts']  if preset and preset in PRESETS else 2
    dt_v   = PRESETS[preset]['device_type']       if preset and preset in PRESETS else 'mobile'
    tc_v   = PRESETS[preset]['txn_consistency']   if preset and preset in PRESETS else 70.0
    nr_v   = PRESETS[preset]['nighttime_ratio']   if preset and preset in PRESETS else 0.15

    dt_opts = ['mobile', 'desktop']
    bank_loyalty         = b6.number_input("Bank loyalty (yrs)",     0, 25, value=bl_v)
    support_contacts     = b7.number_input("Support contacts",       0, 15, value=sc_v)
    device_type          = b8.selectbox("Device type",  dt_opts, index=dt_opts.index(dt_v))
    txn_consistency      = b9.number_input("Txn consistency (0–100)", 0.0, 100.0, value=float(tc_v), step=0.5)
    nighttime_txn_ratio  = b10.number_input("Nighttime txn ratio",   0.0, 1.0, value=float(nr_v), step=0.01)

    submitted = st.form_submit_button("⚡ Evaluate borrower", use_container_width=True, type="primary")


# ── EVALUATION & RESULTS ──────────────────────────────────────────
if submitted:
    profile = {
        'Name':                    applicant_name or 'Applicant',
        'Age':                     int(age),
        'Income':                  int(income),
        'LoanAmount':              int(loan_amount),
        'CreditScore':             int(credit_score),
        'EmploymentLength':        int(employment_len),
        'DebtToIncomeRatio':       float(dti),
        'OpenCreditLines':         int(open_lines),
        'LatePayments':            int(late_payments),
        'HomeOwnership':           home_ownership,
        'LoanPurpose':             loan_purpose,
        'AppSessionTime':          float(app_session_time),
        'UtilityPaymentLatency':   int(utility_latency),
        'SocialSentimentScore':    float(social_sentiment),
        'OverdraftFrequency':      int(overdraft_freq),
        'NumInquiries':            int(num_inquiries),
        'BankLoyaltyScore':        int(bank_loyalty),
        'SupportContactFrequency': int(support_contacts),
        'DeviceType':              device_type,
        'TransactionConsistency':  float(txn_consistency),
        'NighttimeTransactionRatio': float(nighttime_txn_ratio),
    }

    result = evaluate_borrower(profile)
    decision  = result['decision']
    prob      = result['prob_default']
    zone      = result['zone']
    mit_score = result['mit_score']
    triggered = result['triggered_factors']
    letter    = result['letter']

    st.divider()
    st.markdown("## 📋 Decision result")

    # ── DECISION BANNER ───────────────────────────────────────────
    zone_colors = {'green': '#16A34A', 'amber': '#D97706', 'red': '#DC2626'}
    zone_bg     = {'green': '#DCFCE7', 'amber': '#FEF9C3', 'red': '#FEE2E2'}
    zone_labels = {'green': '🟢 LOW RISK', 'amber': '🟡 GREY ZONE', 'red': '🔴 HIGH RISK'}
    decision_labels = {
        'APPROVE':     '✅ APPROVED',
        'CONDITIONAL': '⚠️ CONDITIONALLY APPROVED',
        'REJECT':      '❌ DECLINED',
    }
    css_class = {'APPROVE': 'decision-approve', 'CONDITIONAL': 'decision-conditional', 'REJECT': 'decision-reject'}

    st.markdown(f"""
    <div class="{css_class[decision]}">
        <div style="font-size:1.6rem; font-weight:800; color:{zone_colors[zone]}; margin-bottom:4px;">
            {decision_labels[decision]}
        </div>
        <div style="font-size:0.95rem; color:#475569;">
            {zone_labels[zone]} &nbsp;|&nbsp; 
            P(default): <strong>{prob:.1%}</strong> &nbsp;|&nbsp;
            Model: <strong>{result['model_name']}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── THREE COLUMNS: GAUGE / MITIGATING / FACTORS ───────────────
    col1, col2, col3 = st.columns([1.2, 1, 1.4])

    with col1:
        st.markdown('<div class="section-header">Risk gauge</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw=dict(polar=False))
        fig.patch.set_facecolor('white')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
        # gauge bar
        ax.barh(0.5, 1.0, height=0.35, color='#F1F5F9', left=0, linewidth=0)
        ax.barh(0.5, 0.40, height=0.35, color='#86EFAC', left=0, linewidth=0)
        ax.barh(0.5, 0.20, height=0.35, color='#FDE68A', left=0.40, linewidth=0)
        ax.barh(0.5, 0.40, height=0.35, color='#FCA5A5', left=0.60, linewidth=0)
        # needle
        needle_color = zone_colors[zone]
        ax.plot([prob, prob], [0.28, 0.72], color=needle_color, linewidth=3, solid_capstyle='round')
        ax.plot(prob, 0.5, 'o', color=needle_color, markersize=10)
        ax.text(prob, 0.82, f'{prob:.0%}', ha='center', va='bottom',
                fontsize=13, fontweight='bold', color=needle_color)
        ax.text(0.20, 0.05, 'Approve', ha='center', fontsize=8, color='#16A34A')
        ax.text(0.50, 0.05, 'Grey', ha='center', fontsize=8, color='#D97706')
        ax.text(0.80, 0.05, 'Reject', ha='center', fontsize=8, color='#DC2626')
        plt.tight_layout(pad=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown('<div class="section-header">Mitigating score</div>', unsafe_allow_html=True)
        max_score = sum(f['points'] for f in MITIGATING_FACTORS)
        pct = mit_score / max_score

        fig2, ax2 = plt.subplots(figsize=(3, 2.2))
        fig2.patch.set_facecolor('white')
        ax2.set_facecolor('white')
        bar_color = '#16A34A' if pct >= 0.4 else ('#D97706' if pct >= 0.2 else '#E2E8F0')
        ax2.barh(0, max_score, height=0.5, color='#F1F5F9', left=0)
        ax2.barh(0, mit_score, height=0.5, color=bar_color, left=0)
        ax2.set_xlim(0, max_score); ax2.set_ylim(-0.8, 0.8)
        ax2.axis('off')
        ax2.text(mit_score/2, 0, f'{mit_score}/{max_score}',
                 ha='center', va='center', fontsize=16, fontweight='bold', color='white' if pct>0.15 else '#94A3B8')
        threshold_label = 'Threshold for conditional: 4 pts'
        ax2.axvline(4, color='#D97706', linewidth=1.5, linestyle='--')
        ax2.text(4.1, -0.55, '≥4 = conditional', fontsize=7, color='#D97706')
        plt.tight_layout(pad=0.2)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        if zone == 'amber':
            if mit_score >= 4:
                st.success(f"Score {mit_score} — grey zone cleared ✓")
            else:
                st.error(f"Score {mit_score} — insufficient for conditional")
        elif zone == 'green':
            st.info("Below risk threshold — direct approval")
        else:
            st.warning("Above risk threshold — rejected")

    with col3:
        st.markdown('<div class="section-header">Mitigating factors</div>', unsafe_allow_html=True)
        triggered_labels = {t['label'] for t in triggered}
        for f in MITIGATING_FACTORS:
            hit = f['label'] in triggered_labels
            icon = '✅' if hit else '○'
            pts  = f'+{f["points"]}' if hit else f'+{f["points"]}'
            color = '#166534' if hit else '#94A3B8'
            bg    = '#F0FDF4' if hit else '#F8FAFC'
            border = '#86EFAC' if hit else '#E2E8F0'
            st.markdown(f"""
            <div style="background:{bg}; border:1px solid {border}; border-radius:8px;
                        padding:5px 10px; margin:3px 0; font-size:0.83rem; color:{color};">
                {icon} <strong>{pts}pts</strong> &nbsp; {f['label']}
            </div>""", unsafe_allow_html=True)

    # ── CONDITIONS IF CONDITIONAL ─────────────────────────────────
    if decision == 'CONDITIONAL' and result['conditions']:
        st.markdown("#### 📌 Approval conditions")
        for c in result['conditions']:
            st.markdown(f"- {c}")

    # ── EXPLANATION LETTER ────────────────────────────────────────
    st.divider()
    st.markdown("#### 📄 Formal decision letter")
    st.text_area("", letter, height=320, label_visibility="collapsed")
    st.download_button(
        "⬇️ Download letter (.txt)",
        data=letter,
        file_name=f"decision_{profile['Name'].replace(' ','_')}.txt",
        mime="text/plain",
    )
