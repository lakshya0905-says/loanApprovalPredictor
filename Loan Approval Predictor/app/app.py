"""
=============================================================================
  🏦 Loan Approval Prediction System — Streamlit Application
=============================================================================
  Author      : Minor Project — Data Science & ML
  Dataset     : Dream Housing Finance Loan Dataset (614 records)
  Best Model  : Logistic Regression  (saved as best_model.pkl)
  Scaler      : StandardScaler       (saved as scaler.pkl)

  EXACT PREPROCESSING PIPELINE (sourced from notebook):
  -------------------------------------------------------
  1. Feature Engineering:
       TotalIncome = ApplicantIncome + CoapplicantIncome
       EMI_Burden  = LoanAmount / TotalIncome

  2. Label Encoding (LabelEncoder — alphabetical fit_transform):
       Gender        → Female=0, Male=1
       Married       → No=0, Yes=1
       Dependents    → 0=0, 1=1, 2=2, 3+=3
       Education     → Graduate=0, Not Graduate=1
       Self_Employed → No=0, Yes=1
       Property_Area → Rural=0, Semiurban=1, Urban=2

  3. Feature Order (exact, from model.feature_names_in_):
       ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'TotalIncome', 'EMI_Burden']

  4. Scaled columns (exact, from scaler.feature_names_in_):
       ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
        'Loan_Amount_Term', 'TotalIncome', 'EMI_Burden']

  Target: 1 = Approved (Y), 0 = Rejected (N)
=============================================================================
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CUSTOM CSS — Professional banking theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ---- Global font & background ---- */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* ---- Header banner ---- */
        .bank-header {
            background: linear-gradient(135deg, #1a237e 0%, #283593 60%, #3949ab 100%);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(26,35,126,0.3);
        }
        .bank-header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            margin: 0 0 0.3rem 0;
            letter-spacing: -0.5px;
        }
        .bank-header p {
            font-size: 1.05rem;
            margin: 0;
            opacity: 0.88;
        }

        /* ---- Section cards ---- */
        .section-card {
            background: #ffffff;
            border: 1px solid #e3e8f0;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #1a237e;
            border-bottom: 2px solid #e8eaf6;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }

        /* ---- Result boxes ---- */
        .result-approved {
            background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
            border: 2px solid #2e7d32;
            border-radius: 12px;
            padding: 1.8rem;
            text-align: center;
        }
        .result-rejected {
            background: linear-gradient(135deg, #ffebee, #ffcdd2);
            border: 2px solid #c62828;
            border-radius: 12px;
            padding: 1.8rem;
            text-align: center;
        }
        .result-icon { font-size: 3.5rem; }
        .result-text-approved {
            font-size: 1.9rem;
            font-weight: 700;
            color: #1b5e20;
            margin: 0.3rem 0;
        }
        .result-text-rejected {
            font-size: 1.9rem;
            font-weight: 700;
            color: #b71c1c;
            margin: 0.3rem 0;
        }
        .result-sub {
            font-size: 1rem;
            color: #555;
            margin-top: 0.4rem;
        }

        /* ---- Probability bar ---- */
        .prob-bar-wrap {
            background: #e0e0e0;
            border-radius: 20px;
            height: 18px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        .prob-bar-fill-approved {
            background: linear-gradient(90deg, #43a047, #2e7d32);
            height: 100%;
            border-radius: 20px;
        }
        .prob-bar-fill-rejected {
            background: linear-gradient(90deg, #e53935, #b71c1c);
            height: 100%;
            border-radius: 20px;
        }

        /* ---- Feature summary chips ---- */
        .chip {
            display: inline-block;
            background: #e8eaf6;
            color: #283593;
            border-radius: 20px;
            padding: 0.25rem 0.75rem;
            font-size: 0.82rem;
            font-weight: 600;
            margin: 0.2rem;
        }

        /* ---- Engineered feature box ---- */
        .eng-box {
            background: #f3f4f9;
            border-left: 4px solid #3949ab;
            border-radius: 6px;
            padding: 0.9rem 1.2rem;
            margin: 0.4rem 0;
        }
        .eng-label { font-size: 0.82rem; color: #666; margin-bottom: 0.15rem; }
        .eng-value { font-size: 1.4rem; font-weight: 700; color: #1a237e; }

        /* ---- Sidebar styling ---- */
        [data-testid="stSidebar"] {
            background: #1a237e;
        }
        [data-testid="stSidebar"] * {
            color: #e8eaf6 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #3949ab !important;
        }

        /* ---- Predict button ---- */
        div.stButton > button {
            background: linear-gradient(135deg, #1a237e, #3949ab);
            color: white;
            font-size: 1.05rem;
            font-weight: 600;
            padding: 0.7rem 2.5rem;
            border: none;
            border-radius: 8px;
            width: 100%;
            cursor: pointer;
            transition: all 0.2s ease;
            letter-spacing: 0.5px;
        }
        div.stButton > button:hover {
            background: linear-gradient(135deg, #283593, #5c6bc0);
            box-shadow: 0 4px 15px rgba(26,35,126,0.4);
            transform: translateY(-1px);
        }

        /* ---- Metric cards ---- */
        .metric-mini {
            background: #f8f9ff;
            border: 1px solid #c5cae9;
            border-radius: 8px;
            padding: 0.7rem 1rem;
            text-align: center;
        }
        .metric-mini-label { font-size: 0.78rem; color: #5c6bc0; font-weight: 600; }
        .metric-mini-value { font-size: 1.3rem; font-weight: 700; color: #1a237e; }

        /* ---- Warning / info notes ---- */
        .note-box {
            background: #fff8e1;
            border-left: 4px solid #ffa000;
            border-radius: 6px;
            padding: 0.7rem 1rem;
            font-size: 0.88rem;
            color: #5d4037;
        }
        .hide-streamlit-brand footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# CONSTANTS — Exact encoding from LabelEncoder(fit_transform) alphabetical sort
# ---------------------------------------------------------------------------
# LabelEncoder assigns codes alphabetically.
ENCODE_MAP = {
    "Gender":        {"Male": 1, "Female": 0},
    "Married":       {"Yes": 1, "No": 0},
    "Dependents":    {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education":     {"Graduate": 0, "Not Graduate": 1},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2},
}

# Exact feature order as stored in model.feature_names_in_
FEATURE_ORDER = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area",
    "TotalIncome", "EMI_Burden",
]

# Exact columns scaled — as stored in scaler.feature_names_in_
SCALE_COLS = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "TotalIncome", "EMI_Burden",
]


# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """
    Load best_model.pkl and scaler.pkl from the models/ directory.
    Returns (model, scaler) or raises an informative error.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path  = os.path.join(base_dir, "..", "models", "best_model.pkl")
    scaler_path = os.path.join(base_dir, "..", "models", "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            "Please ensure 'best_model.pkl' is in the 'models/' directory."
        )
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler file not found at: {scaler_path}\n"
            "Please ensure 'scaler.pkl' is in the 'models/' directory."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


# ---------------------------------------------------------------------------
# PREPROCESSING PIPELINE
# ---------------------------------------------------------------------------
def feature_engineering(applicant_income: float, coapplicant_income: float,
                         loan_amount: float) -> tuple[float, float]:
    """
    Recreate the exact engineered features from the notebook.
      TotalIncome = ApplicantIncome + CoapplicantIncome
      EMI_Burden  = LoanAmount / TotalIncome
    """
    total_income = applicant_income + coapplicant_income
    # Guard against division by zero for edge cases
    emi_burden = loan_amount / total_income if total_income > 0 else 0.0
    return total_income, emi_burden


def encode_inputs(raw_inputs: dict) -> dict:
    """
    Apply exact LabelEncoder mapping from the notebook (alphabetical sort).
    Only categorical fields are encoded; numerical fields pass through.
    """
    encoded = raw_inputs.copy()
    for col, mapping in ENCODE_MAP.items():
        if col in encoded:
            encoded[col] = mapping[encoded[col]]
    return encoded


def build_feature_row(encoded: dict) -> pd.DataFrame:
    """
    Assemble a single-row DataFrame in the exact feature order
    the model was trained on (model.feature_names_in_).
    """
    row = {col: [encoded[col]] for col in FEATURE_ORDER}
    return pd.DataFrame(row)


def scale_features(df_row: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Apply StandardScaler to exactly the columns that were scaled during training.
    All other columns remain unchanged.
    """
    df_scaled = df_row.copy()
    df_scaled[SCALE_COLS] = scaler.transform(df_scaled[SCALE_COLS])
    return df_scaled


def preprocess(raw_inputs: dict, scaler) -> pd.DataFrame:
    """
    Full preprocessing pipeline matching the notebook exactly:
      1. Feature engineering (TotalIncome, EMI_Burden)
      2. Encode categorical features
      3. Build ordered feature row
      4. Scale numerical features
    """
    encoded = encode_inputs(raw_inputs)
    df_row  = build_feature_row(encoded)
    df_final = scale_features(df_row, scaler)
    return df_final


def predict(model, df_final: pd.DataFrame) -> tuple[int, float, float]:
    """
    Run prediction and return (class, prob_approved, prob_rejected).
    model.classes_ = [0, 1]  →  index 1 = Approved
    """
    pred_class = int(model.predict(df_final)[0])
    proba      = model.predict_proba(df_final)[0]   # [P(rejected), P(approved)]
    prob_approved = float(proba[1])
    prob_rejected = float(proba[0])
    return pred_class, prob_approved, prob_rejected


# ---------------------------------------------------------------------------
# UI HELPERS
# ---------------------------------------------------------------------------
def render_result(pred_class: int, prob_approved: float, prob_rejected: float):
    """Render the approval/rejection result card with probability bar."""
    if pred_class == 1:
        st.markdown(
            f"""
            <div class="result-approved">
                <div class="result-icon">✅</div>
                <div class="result-text-approved">Loan Approved</div>
                <div class="result-sub">
                    The applicant is likely to be approved based on the provided information.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bar_class = "prob-bar-fill-approved"
    else:
        st.markdown(
            f"""
            <div class="result-rejected">
                <div class="result-icon">❌</div>
                <div class="result-text-rejected">Loan Rejected</div>
                <div class="result-sub">
                    The application does not meet the approval criteria based on the provided information.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bar_class = "prob-bar-fill-rejected"

    # Probability meters
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="metric-mini">
                <div class="metric-mini-label">✅ APPROVAL PROBABILITY</div>
                <div class="metric-mini-value">{prob_approved * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bar_pct = int(prob_approved * 100)
        st.markdown(
            f"""
            <div class="prob-bar-wrap">
                <div class="{bar_class}" style="width:{bar_pct}%"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-mini">
                <div class="metric-mini-label">❌ REJECTION PROBABILITY</div>
                <div class="metric-mini-value">{prob_rejected * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        bar_pct2 = int(prob_rejected * 100)
        st.markdown(
            f"""
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill-rejected" style="width:{bar_pct2}%"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_engineered_features(total_income: float, emi_burden: float):
    """Display the computed engineered features before prediction."""
    st.markdown("#### 🔧 Computed Feature Engineering")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            <div class="eng-box">
                <div class="eng-label">TotalIncome  =  ApplicantIncome + CoapplicantIncome</div>
                <div class="eng-value">₹ {total_income:,.0f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="eng-box">
                <div class="eng-label">EMI_Burden  =  LoanAmount ÷ TotalIncome</div>
                <div class="eng-value">{emi_burden:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_explanation(pred_class: int, raw_inputs: dict,
                                   total_income: float, emi_burden: float,
                                   prob_approved: float):
    """Provide a plain-language explanation of the prediction factors."""
    credit = raw_inputs["Credit_History"]
    edu    = raw_inputs["Education"]
    area   = raw_inputs["Property_Area"]

    factors_positive = []
    factors_negative = []

    if credit == "Good (1)":
        factors_positive.append("✅ Good credit history — strongest positive signal")
    else:
        factors_negative.append("❌ Poor credit history — strongest negative signal")

    if edu == "Graduate":
        factors_positive.append("✅ Graduate education — associated with higher approval rates")
    else:
        factors_negative.append("⚠️ Non-graduate education — marginally lower approval rate")

    if area == "Semiurban":
        factors_positive.append("✅ Semiurban property area — highest approval rate in dataset")
    elif area == "Urban":
        factors_positive.append("✅ Urban property area — good approval rate")
    else:
        factors_negative.append("⚠️ Rural property area — slightly lower approval rate")

    if raw_inputs["Married"] == "Yes":
        factors_positive.append("✅ Married applicant — positively correlated with approval")

    if emi_burden < 0.03:
        factors_positive.append(f"✅ Low EMI burden ({emi_burden:.4f}) — affordable loan amount")
    elif emi_burden > 0.06:
        factors_negative.append(f"⚠️ High EMI burden ({emi_burden:.4f}) — loan amount is high relative to income")

    if total_income > 5000:
        factors_positive.append(f"✅ Combined income ₹{total_income:,.0f} — above dataset average")
    else:
        factors_negative.append(f"⚠️ Combined income ₹{total_income:,.0f} — below dataset average (₹7,025)")

    with st.expander("📋 Prediction Explanation", expanded=True):
        st.markdown(
            f"**Decision Confidence:** {prob_approved * 100:.1f}% approval probability "
            f"({'High' if prob_approved > 0.75 else 'Moderate' if prob_approved > 0.5 else 'Low'} confidence)"
        )
        st.markdown("---")
        if factors_positive:
            st.markdown("**Factors supporting approval:**")
            for f in factors_positive:
                st.markdown(f"&nbsp;&nbsp;{f}")
        if factors_negative:
            st.markdown("**Factors working against approval:**")
            for f in factors_negative:
                st.markdown(f"&nbsp;&nbsp;{f}")

        st.markdown("---")
        st.markdown(
            "<div class='note-box'>⚠️ <b>Disclaimer:</b> This prediction is generated by a machine "
            "learning model trained on historical data. It is intended for educational purposes only "
            "and should not be used as the sole basis for any financial decision.</div>",
            unsafe_allow_html=True,
        )


def render_sidebar():
    """Render the sidebar with project information."""
    with st.sidebar:
        st.markdown("## 🏦 Project Info")
        st.markdown("---")

        st.markdown("**Model**")
        st.markdown("Logistic Regression")
        st.markdown("*(max_iter=1000, random_state=42)*")
        st.markdown("---")

        st.markdown("**Dataset**")
        st.markdown("Dream Housing Finance  \nLoan Dataset")
        st.markdown("614 training records  \n367 test records")
        st.markdown("---")

        st.markdown("**Features Used (13)**")
        features_display = [
            "Gender", "Married", "Dependents",
            "Education", "Self Employed",
            "Applicant Income", "CoApplicant Income",
            "Loan Amount", "Loan Amount Term",
            "Credit History", "Property Area",
            "🔧 Total Income *", "🔧 EMI Burden *",
        ]
        for feat in features_display:
            st.markdown(f"• {feat}")
        st.markdown("*🔧 = Engineered feature*")
        st.markdown("---")

        st.markdown("**Encoding**")
        st.markdown(
            "Label Encoding (alphabetical)  \n"
            "• Gender: M=1, F=0  \n"
            "• Married: Y=1, N=0  \n"
            "• Education: Grad=0  \n"
            "• Property: R=0, S=1, U=2"
        )
        st.markdown("---")

        st.markdown("**Scaling**")
        st.markdown("StandardScaler on 6 numerical columns")
        st.markdown("---")

        st.markdown("**Target**")
        st.markdown("1 = Approved (Y)  \n0 = Rejected (N)")


# ---------------------------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------------------------
def main():
    # ---- Sidebar ----
    render_sidebar()

    # ---- Header ----
    st.markdown(
        """
        <div class="bank-header">
            <h1>🏦 Loan Approval Prediction System</h1>
            <p>Predict whether a loan application is likely to be approved based on applicant information.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Load model artifacts ----
    try:
        model, scaler = load_artifacts()
    except FileNotFoundError as e:
        st.error(f"**Model Loading Error:**\n\n{e}")
        st.stop()

    # ---- About Project Expander ----
    with st.expander("ℹ️ About This Project", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                """
                **Project Overview**

                This application uses a **Logistic Regression** model trained on the
                Dream Housing Finance loan dataset to predict loan approval decisions.

                **Workflow:**
                1. User inputs applicant details via the form below.
                2. Two features are engineered automatically:
                   - `TotalIncome` = ApplicantIncome + CoapplicantIncome
                   - `EMI_Burden` = LoanAmount ÷ TotalIncome
                3. Categorical features are label-encoded using the same mapping
                   applied during training.
                4. Numerical features are scaled with a fitted StandardScaler.
                5. The model outputs an approval decision and confidence probability.
                """
            )
        with col_b:
            st.markdown(
                """
                **Key Findings from the Notebook:**

                - **Credit History** is the most important predictor (corr ≈ 0.54 with target)
                - **Semiurban** property areas have the highest approval rate
                - **Graduates** are approved more frequently than non-graduates
                - **EMI_Burden** and **TotalIncome** (engineered) ranked in top 5 features
                - Dataset: 68.7% approved, 31.3% rejected (moderately imbalanced)

                **Model Performance (test set, 20% holdout):**
                - Best model selected by highest accuracy across
                  Logistic Regression, Decision Tree, and Random Forest.
                """
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ====================================================================
    # INPUT FORM
    # ====================================================================
    st.markdown(
        "<div class='section-title' style='font-size:1.15rem; font-weight:700; "
        "color:#1a237e; margin-bottom:1rem;'>📝 Applicant Information</div>",
        unsafe_allow_html=True,
    )

    # ---- Row 1: Personal Information ----
    st.markdown(
        "<div class='section-card'>"
        "<div class='section-title'>👤 Personal Details</div>",
        unsafe_allow_html=True,
    )
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            help="Applicant's gender",
        )
    with r1c2:
        married = st.selectbox(
            "Marital Status",
            options=["Yes", "No"],
            help="Is the applicant married?",
        )
    with r1c3:
        dependents = st.selectbox(
            "Dependents",
            options=["0", "1", "2", "3+"],
            help="Number of dependents",
        )
    with r1c4:
        education = st.selectbox(
            "Education",
            options=["Graduate", "Not Graduate"],
            help="Highest education level",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Row 2: Employment & Property ----
    st.markdown(
        "<div class='section-card'>"
        "<div class='section-title'>💼 Employment & Property</div>",
        unsafe_allow_html=True,
    )
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        self_employed = st.selectbox(
            "Self Employed",
            options=["No", "Yes"],
            help="Is the applicant self-employed?",
        )
    with r2c2:
        property_area = st.selectbox(
            "Property Area",
            options=["Semiurban", "Urban", "Rural"],
            help="Location of the property",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Row 3: Financial Information ----
    st.markdown(
        "<div class='section-card'>"
        "<div class='section-title'>💰 Financial Details</div>",
        unsafe_allow_html=True,
    )
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        applicant_income = st.number_input(
            "Applicant Income (₹ / month)",
            min_value=0,
            max_value=200_000,
            value=5_000,
            step=500,
            help="Monthly income of the primary applicant. Dataset range: ₹150 – ₹81,000",
        )
    with r3c2:
        coapplicant_income = st.number_input(
            "Co-Applicant Income (₹ / month)",
            min_value=0,
            max_value=100_000,
            value=0,
            step=500,
            help="Monthly income of the co-applicant (0 if none). Dataset range: ₹0 – ₹41,667",
        )
    with r3c3:
        loan_amount = st.number_input(
            "Loan Amount (₹ thousands)",
            min_value=1,
            max_value=1_000,
            value=128,
            step=1,
            help="Requested loan amount in thousands. Dataset range: ₹9K – ₹700K",
        )

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        loan_amount_term = st.number_input(
            "Loan Amount Term (months)",
            min_value=12,
            max_value=480,
            value=360,
            step=12,
            help="Repayment term in months. Most common: 360 months (30 years)",
        )
    with r4c2:
        credit_history = st.selectbox(
            "Credit History",
            options=["Good (1)", "Bad (0)"],
            help="1 = Good credit history (repaid debts), 0 = Poor credit history. "
                 "This is the MOST IMPORTANT feature for loan approval.",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ====================================================================
    # PREDICT BUTTON
    # ====================================================================
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict Loan Status", use_container_width=True)

    # ====================================================================
    # PREDICTION LOGIC
    # ====================================================================
    if predict_clicked:
        # ---- Input validation ----
        errors = []
        if applicant_income == 0 and coapplicant_income == 0:
            errors.append("Both Applicant Income and Co-Applicant Income are 0. "
                          "At least one applicant must have an income.")
        if loan_amount <= 0:
            errors.append("Loan Amount must be greater than 0.")

        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")
            st.stop()

        # ---- Feature engineering ----
        total_income, emi_burden = feature_engineering(
            applicant_income, coapplicant_income, loan_amount
        )

        # ---- Encode credit history (strip label, keep numeric value) ----
        credit_history_val = 1.0 if credit_history == "Good (1)" else 0.0

        # ---- Collect raw inputs (categorical as strings for encode_inputs) ----
        raw_inputs = {
            "Gender":            gender,
            "Married":           married,
            "Dependents":        dependents,
            "Education":         education,
            "Self_Employed":     self_employed,
            "ApplicantIncome":   float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount":        float(loan_amount),
            "Loan_Amount_Term":  float(loan_amount_term),
            "Credit_History":    credit_history,    # display string kept for explanation
            "Property_Area":     property_area,
            "TotalIncome":       total_income,
            "EMI_Burden":        emi_burden,
        }

        # ---- Build encoded dict for preprocessing ----
        encoded_inputs = {
            "Gender":            ENCODE_MAP["Gender"][gender],
            "Married":           ENCODE_MAP["Married"][married],
            "Dependents":        ENCODE_MAP["Dependents"][dependents],
            "Education":         ENCODE_MAP["Education"][education],
            "Self_Employed":     ENCODE_MAP["Self_Employed"][self_employed],
            "ApplicantIncome":   float(applicant_income),
            "CoapplicantIncome": float(coapplicant_income),
            "LoanAmount":        float(loan_amount),
            "Loan_Amount_Term":  float(loan_amount_term),
            "Credit_History":    credit_history_val,
            "Property_Area":     ENCODE_MAP["Property_Area"][property_area],
            "TotalIncome":       total_income,
            "EMI_Burden":        emi_burden,
        }

        # ---- Build ordered DataFrame & scale ----
        try:
            df_row   = build_feature_row(encoded_inputs)
            df_final = scale_features(df_row, scaler)
        except Exception as ex:
            st.error(f"Preprocessing error: {ex}")
            st.stop()

        # ---- Predict ----
        try:
            pred_class, prob_approved, prob_rejected = predict(model, df_final)
        except Exception as ex:
            st.error(f"Prediction error: {ex}")
            st.stop()

        # ====================================================================
        # RESULTS DISPLAY
        # ====================================================================
        st.markdown("---")
        st.markdown(
            "<div class='section-title' style='font-size:1.15rem; font-weight:700; "
            "color:#1a237e; margin-bottom:1rem;'>📊 Prediction Results</div>",
            unsafe_allow_html=True,
        )

        # Engineered features display
        render_engineered_features(total_income, emi_burden)
        st.markdown("<br>", unsafe_allow_html=True)

        # Main result card
        render_result(pred_class, prob_approved, prob_rejected)
        st.markdown("<br>", unsafe_allow_html=True)

        # Input summary
        with st.expander("📄 Input Summary", expanded=False):
            summary_data = {
                "Field": [
                    "Gender", "Married", "Dependents", "Education",
                    "Self Employed", "Property Area", "Credit History",
                    "Applicant Income", "Co-Applicant Income",
                    "Loan Amount", "Loan Amount Term",
                    "— Total Income (computed)", "— EMI Burden (computed)",
                ],
                "Value": [
                    gender, married, dependents, education,
                    self_employed, property_area, credit_history,
                    f"₹ {applicant_income:,}", f"₹ {coapplicant_income:,}",
                    f"₹ {loan_amount}K", f"{loan_amount_term} months",
                    f"₹ {total_income:,.0f}", f"{emi_burden:.6f}",
                ],
                "Encoded Value": [
                    encoded_inputs["Gender"], encoded_inputs["Married"],
                    encoded_inputs["Dependents"], encoded_inputs["Education"],
                    encoded_inputs["Self_Employed"], encoded_inputs["Property_Area"],
                    encoded_inputs["Credit_History"],
                    "→ scaled", "→ scaled",
                    "→ scaled", "→ scaled",
                    "→ scaled", "→ scaled",
                ],
            }
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # Prediction explanation
        render_prediction_explanation(
            pred_class, raw_inputs, total_income, emi_burden, prob_approved
        )

    else:
        # Placeholder shown before first prediction
        st.markdown(
            """
            <div style="text-align:center; padding:2.5rem; color:#9e9e9e; border:2px dashed #e0e0e0;
                        border-radius:12px; margin-top:1rem;">
                <div style="font-size:2.5rem;">🔍</div>
                <div style="font-size:1.1rem; margin-top:0.5rem;">
                    Fill in the applicant details above and click
                    <b style="color:#3949ab;">Predict Loan Status</b> to see the result.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
