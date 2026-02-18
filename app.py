
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="𝕮𝖗𝖊𝖉𝖎𝖙 𝕮𝖆𝖗𝖉 𝕱𝖗𝖆𝖚𝖉 𝕯𝖊𝖙𝖊𝖈𝖙𝖎𝖔𝖓",
    layout="wide"
)

# ================= GLOBAL CSS =================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top, #0f2027, #000000 65%);
    color: #e5e7eb;
}

/* -------- TITLES -------- */
h1 { color: #facc15; font-weight: 800; }
h2, h3 { color: #22c55e; }

/* -------- SECTION BOX -------- */
.section-box {
    background: rgba(17, 24, 39, 0.82);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 26px;
    box-shadow: 0 0 22px rgba(0,0,0,0.45);
}

/* -------- INPUTS -------- */
input[type="number"] {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 10px !important;
    border: 1px solid #1f2937 !important;
}

/* -------- BUTTONS -------- */
.stButton>button {
    width: 100%;
    border-radius: 16px;
    padding: 0.65em 1em;
    font-size: 16px;
    font-weight: 700;
    border: none;
    transition: all 0.2s ease;
}

/* Predict button */
button[kind="primary"] {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    color: black;
}
button[kind="primary"]:hover {
    transform: scale(1.02);
    box-shadow: 0 0 18px rgba(34,197,94,0.6);
}

/* -------- METRIC -------- */
[data-testid="stMetricValue"] {
    font-size: 30px;
    font-weight: 800;
}

/* -------- FOOTER -------- */
.footer {
    text-align: center;
    opacity: 0.6;
    margin-top: 40px;
}

/* =====================================================
   💲 FLOATING MONEY BACKGROUND
   ===================================================== */
.money-bg {
    position: fixed;
    inset: 0;
    overflow: hidden;
    z-index: 0;
    pointer-events: none;
}

.money-bg span {
    position: absolute;
    bottom: -10vh;
    font-size: 32px;
    font-weight: 700;
    color: rgba(34,197,94,0.6);
    text-shadow: 0 0 12px rgba(34,197,94,0.6);
    animation: floatMoney 18s linear infinite;
}

.money-bg span:nth-child(1)  { left: 5%;  animation-delay: 0s; }
.money-bg span:nth-child(2)  { left: 15%; animation-delay: 3s; font-size: 24px; }
.money-bg span:nth-child(3)  { left: 25%; animation-delay: 6s; }
.money-bg span:nth-child(4)  { left: 35%; animation-delay: 2s; font-size: 40px; }
.money-bg span:nth-child(5)  { left: 45%; animation-delay: 8s; }
.money-bg span:nth-child(6)  { left: 55%; animation-delay: 4s; font-size: 26px; }
.money-bg span:nth-child(7)  { left: 65%; animation-delay: 10s; }
.money-bg span:nth-child(8)  { left: 75%; animation-delay: 5s; font-size: 34px; }
.money-bg span:nth-child(9)  { left: 85%; animation-delay: 12s; }
.money-bg span:nth-child(10) { left: 95%; animation-delay: 7s; font-size: 22px; }

@keyframes floatMoney {
    0%   { transform: translateY(0) rotate(0deg); opacity: 0; }
    20%  { opacity: 0.6; }
    50%  { opacity: 0.9; }
    80%  { opacity: 0.4; }
    100% { transform: translateY(-120vh) rotate(360deg); opacity: 0; }
}

/* =====================================================
   ⚡ BEAM BUTTON EFFECT
   ===================================================== */
.beam-btn {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
}

.beam-btn::before,
.beam-btn::after {
    content: "";
    position: absolute;
    inset: -2px;
    border-radius: inherit;
    border: 2px solid transparent;
    opacity: 0;
}

.beam-active::before,
.beam-active::after {
    opacity: 1;
    animation-duration: 1.4s;
    animation-iteration-count: infinite;
    animation-timing-function: linear;
}

.beam-active::before { animation-name: beamCW; }
.beam-active::after  { animation-name: beamCCW; }

.beam-green::before,
.beam-green::after {
    border-image: linear-gradient(90deg, transparent, #22c55e, transparent) 1;
}

.beam-red::before,
.beam-red::after {
    border-image: linear-gradient(90deg, transparent, #ef4444, transparent) 1;
}

@keyframes beamCW {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes beamCCW {
    from { transform: rotate(360deg); }
    to   { transform: rotate(0deg); }
}
</style>
""", unsafe_allow_html=True)

# ================= MONEY BACKGROUND =================
st.markdown("""
<div class="money-bg">
  <span>$</span><span>$</span><span>$</span><span>$</span><span>$</span>
  <span>$</span><span>$</span><span>$</span><span>$</span><span>$</span>
</div>
""", unsafe_allow_html=True)

# ================= PATHS =================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_fraud_model.pkl"
DATA_PATH = BASE_DIR/"notebooks"/"data" / "creditcard.csv"

# ================= LOAD MODEL & DATA =================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

# ================= SESSION STATE =================
if "input_row" not in st.session_state:
    st.session_state.input_row = None
if "safe_active" not in st.session_state:
    st.session_state.safe_active = False
if "fraud_active" not in st.session_state:
    st.session_state.fraud_active = False

# ================= TITLE =================
st.title("💳 Credit Card Fraud Detection")
st.write("Predict the probability that a transaction is fraudulent.")

# ================= QUICK DEMO =================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("⚡ Quick Demo Transactions")

c1, c2 = st.columns(2)

# SAFE BUTTON
safe_cls = "beam-btn beam-green"
if st.session_state.safe_active:
    safe_cls += " beam-active"

st.markdown(f'<div class="{safe_cls}">', unsafe_allow_html=True)
if st.button("🟢 Load Sample NORMAL Transaction"):
    st.session_state.input_row = (
        df[df["Class"] == 0]
        .sample(1, random_state=42)
        .drop(columns=["Class"])
        .iloc[0]
    )
    st.session_state.safe_active = True
    st.session_state.fraud_active = False
st.markdown('</div>', unsafe_allow_html=True)

# FRAUD BUTTON
fraud_cls = "beam-btn beam-red"
if st.session_state.fraud_active:
    fraud_cls += " beam-active"

st.markdown(f'<div class="{fraud_cls}">', unsafe_allow_html=True)
if st.button("🔴 Load Sample FRAUD Transaction"):
    st.session_state.input_row = (
        df[df["Class"] == 1]
        .sample(1, random_state=42)
        .drop(columns=["Class"])
        .iloc[0]
    )
    st.session_state.fraud_active = True
    st.session_state.safe_active = False
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= INPUT HELPERS =================
def get_val(col, default=0.0):
    if st.session_state.input_row is not None:
        return float(st.session_state.input_row[col])
    return default

# ================= INPUT UI =================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.subheader("📊 Transaction Features")

col1, col2 = st.columns(2)
with col1:
    time_val = st.number_input("Time", value=get_val("Time"), step=1.0)
with col2:
    amount = st.number_input("Amount", value=get_val("Amount"), step=10.0)

st.markdown("---")

feature_values = {}
cols = st.columns(4)
for i in range(1, 29):
    with cols[(i - 1) % 4]:
        feature_values[f"V{i}"] = st.number_input(
            f"V{i}", value=get_val(f"V{i}"), step=0.01
        )

st.markdown('</div>', unsafe_allow_html=True)

# ================= PREDICTION =================
input_df = pd.DataFrame([{"Time": time_val, **feature_values, "Amount": amount}])

st.markdown('<div class="section-box">', unsafe_allow_html=True)
if st.button("💰 Predict Fraud Risk", type="primary"):
    prob = model.predict_proba(input_df)[0][1]
    st.metric("Fraud Probability", f"{prob:.4f}")
    if prob > 0.5:
        st.error("⚠️ High Risk Transaction — Likely FRAUD")
    else:
        st.success("✅ Low Risk Transaction — Likely GENUINE")
st.markdown('</div>', unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
💼 Credit Card Fraud Detection • XGBoost • Streamlit • Fintech Demo
</div>
""", unsafe_allow_html=True)

