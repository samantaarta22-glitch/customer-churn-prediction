import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Telco Churn AI - SMOTE",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
/* Main app */
.stApp {
    background: #0E1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0F172A 55%, #0B1120 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

/* Sidebar navigation */
section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 8px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 10px 12px;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(59,130,246,0.12);
    border-color: rgba(96,165,250,0.35);
    transform: translateX(2px);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(90deg, rgba(37,99,235,0.28), rgba(37,99,235,0.08));
    border-color: rgba(96,165,250,0.55);
}

/* Titles */
h1 {
    color: #F8FAFC !important;
    font-weight: 800 !important;
}

h2 {
    color: #F8FAFC !important;
}

h3 {
    color: #E2E8F0 !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #161B22;
    border: 1px solid #30363D;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #60A5FA;
    font-weight: 700;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: 1px solid rgba(96,165,250,0.25);
    padding: 12px;
    font-weight: 700;
    background: #2563EB;
    color: white;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #3B82F6;
    transform: translateY(-1px);
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 8px;
}

/* Divider */
hr {
    border-color: #30363D;
}

/* Info / alerts */
div[data-testid="stAlert"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    df = pd.read_excel("TelcoChurn.xlsx")
    return df


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("churn_smote_model.pkl")


try:
    df = load_data()
    model = load_model()
except Exception as e:
    st.error("Gagal memuat dataset atau model.")
    st.code(str(e))
    st.stop()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def create_tenure_group(tenure):
    if tenure <= 12:
        return "New"
    elif tenure <= 24:
        return "Early"
    elif tenure <= 48:
        return "Mid"
    return "Loyal"


service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["TenureGroup"] = df["tenure"].apply(create_tenure_group)

df["TotalServices"] = (
    df[service_columns] == "Yes"
).sum(axis=1)

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["AverageMonthlyCharges"] = (
    df["TotalCharges"] /
    df["tenure"].replace(0, 1)
)

df["IsLongTermContract"] = (
    df["Contract"] != "Month-to-month"
).astype(int)


# =========================================================
# SIDEBAR HEADER
# =========================================================

st.sidebar.markdown("""
<div style="
    padding: 8px 4px 18px 4px;
">
    <div style="
        display:flex;
        align-items:center;
        gap:12px;
    ">
        <div style="
            width:44px;
            height:44px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:rgba(37,99,235,0.18);
            border:1px solid rgba(96,165,250,0.25);
            border-radius:12px;
            font-size:24px;
        ">🤖</div>

        <div>
            <div style="
                font-size:20px;
                font-weight:800;
                color:#F8FAFC;
                line-height:1.1;
            ">Telco Churn</div>

            <div style="
                font-size:12px;
                color:#60A5FA;
                font-weight:700;
                margin-top:4px;
            ">AI ANALYTICS</div>
        </div>
    </div>

    <div style="
        color:#94A3B8;
        font-size:12px;
        line-height:1.6;
        margin-top:16px;
    ">
        Customer churn prediction menggunakan
        <b style="color:#CBD5E1;">Logistic Regression + SMOTE</b>.
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# NAVIGATION
# =========================================================

st.sidebar.markdown("""
<div style="
    color:#64748B;
    font-size:10px;
    font-weight:800;
    letter-spacing:1.4px;
    margin:4px 0 8px 2px;
">
    NAVIGATION
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Churn Analytics",
        "🔮 Customer Prediction"
    ],
    label_visibility="collapsed"
)


# =========================================================
# MODEL CARD
# =========================================================

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="
    color:#64748B;
    font-size:10px;
    font-weight:800;
    letter-spacing:1.4px;
    margin:4px 0 10px 2px;
">
    MACHINE LEARNING
</div>

<div style="
    background:rgba(255,255,255,0.035);
    border:1px solid rgba(255,255,255,0.07);
    border-radius:12px;
    padding:14px;
">
    <div style="
        color:#F8FAFC;
        font-size:14px;
        font-weight:700;
    ">
        Logistic Regression
    </div>

    <div style="
        color:#60A5FA;
        font-size:11px;
        font-weight:700;
        margin-top:4px;
    ">
        + SMOTE
    </div>

    <div style="
        display:flex;
        align-items:center;
        gap:8px;
        color:#CBD5E1;
        font-size:11px;
        margin-top:12px;
    ">
        <span style="
            width:8px;
            height:8px;
            background:#22C55E;
            border-radius:50%;
            box-shadow:0 0 8px rgba(34,197,94,0.7);
        "></span>
        Model Active
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="
    margin-top:12px;
    padding:12px;
    background:rgba(37,99,235,0.08);
    border:1px solid rgba(37,99,235,0.15);
    border-radius:10px;
">
    <div style="color:#94A3B8;font-size:10px;">
        ROC-AUC
    </div>
    <div style="
        color:#60A5FA;
        font-size:20px;
        font-weight:800;
        margin-top:2px;
    ">
        83.98%
    </div>
    <div style="color:#64748B;font-size:10px;">
        Recall: 79.14%
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="
    margin-top:16px;
    padding:12px;
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:10px;
">
    <div style="color:#64748B;font-size:10px;">
        BUILT WITH
    </div>

    <div style="
        color:#CBD5E1;
        font-size:11px;
        font-weight:600;
        margin-top:5px;
        line-height:1.7;
    ">
        Python • Pandas • Scikit-learn<br>
        imbalanced-learn • Streamlit
    </div>

    <div style="
        color:#475569;
        font-size:10px;
        margin-top:7px;
    ">
        Data Science Portfolio
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.title("🤖 Telco Churn AI")

    st.markdown("""
    ### Customer Churn Prediction with SMOTE

    Machine Learning application untuk membantu perusahaan
    mengidentifikasi customer yang berpotensi melakukan churn.

    Model menggunakan **Logistic Regression + SMOTE** untuk
    meningkatkan kemampuan mendeteksi customer churn.
    """)

    st.divider()

    total_customers = len(df)

    churn_customers = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churn_customers / total_customers
    ) * 100

    avg_tenure = df["tenure"].mean()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Churn Customers",
            f"{churn_customers:,}"
        )

    with col3:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    with col4:
        st.metric(
            "Avg Tenure",
            f"{avg_tenure:.1f} months"
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        churn_distribution = (
            df["Churn"]
            .value_counts()
            .reset_index()
        )

        churn_distribution.columns = [
            "Churn",
            "Customers"
        ]

        fig = px.pie(
            churn_distribution,
            names="Churn",
            values="Customers",
            hole=0.55,
            title="Customer Churn Distribution"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:
        contract_churn = (
            df.groupby("Contract")["Churn"]
            .apply(
                lambda x: (x == "Yes").mean() * 100
            )
            .reset_index(name="ChurnRate")
        )

        fig = px.bar(
            contract_churn,
            x="Contract",
            y="ChurnRate",
            text_auto=".1f",
            title="Churn Rate by Contract"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    st.subheader("🎯 Model Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", "73.81%")

    with col2:
        st.metric("Precision", "50.43%")

    with col3:
        st.metric("Recall", "79.14%")

    with col4:
        st.metric("F1-score", "61.60%")

    with col5:
        st.metric("ROC-AUC", "83.98%")

    st.info(
        "SMOTE meningkatkan kemampuan model dalam mendeteksi "
        "customer yang benar-benar churn. Recall mencapai **79.14%**."
    )


# =========================================================
# PAGE 2 — ANALYTICS
# =========================================================

elif page == "📊 Churn Analytics":

    st.title("📊 Churn Analytics")

    st.markdown("""
    Explore customer behavior dan faktor-faktor yang
    berkaitan dengan churn.
    """)

    st.divider()

    st.subheader("📈 Churn Rate by Tenure Group")

    tenure_churn = (
        df.groupby("TenureGroup")["Churn"]
        .apply(
            lambda x: (x == "Yes").mean() * 100
        )
        .reset_index(name="ChurnRate")
    )

    tenure_order = [
        "New",
        "Early",
        "Mid",
        "Loyal"
    ]

    tenure_churn["TenureGroup"] = pd.Categorical(
        tenure_churn["TenureGroup"],
        categories=tenure_order,
        ordered=True
    )

    tenure_churn = tenure_churn.sort_values(
        "TenureGroup"
    )

    fig = px.bar(
        tenure_churn,
        x="TenureGroup",
        y="ChurnRate",
        text_auto=".1f",
        title="Churn Rate by Tenure Group"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("💰 Monthly Charges vs Churn")

    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges Distribution"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("💳 Churn Rate by Payment Method")

    payment_churn = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(
            lambda x: (x == "Yes").mean() * 100
        )
        .reset_index(name="ChurnRate")
        .sort_values("ChurnRate")
    )

    fig = px.bar(
        payment_churn,
        x="ChurnRate",
        y="PaymentMethod",
        orientation="h",
        text_auto=".1f",
        title="Churn Rate by Payment Method"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("🛠️ Churn Rate by Number of Services")

    services_churn = (
        df.groupby("TotalServices")["Churn"]
        .apply(
            lambda x: (x == "Yes").mean() * 100
        )
        .reset_index(name="ChurnRate")
    )

    fig = px.line(
        services_churn,
        x="TotalServices",
        y="ChurnRate",
        markers=True,
        title="Churn Rate by Number of Services"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PAGE 3 — PREDICTION
# =========================================================

elif page == "🔮 Customer Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.markdown("""
    Masukkan data customer untuk mendapatkan estimasi
    probabilitas churn.
    """)

    st.divider()

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

    with col2:
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=float(monthly_charges * tenure),
            step=10.0
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    st.subheader("🛠️ Customer Services")

    col1, col2, col3 = st.columns(3)

    with col1:
        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No phone service",
                "No",
                "Yes"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col2:
        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with col3:
        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    # Feature engineering
    tenure_group = create_tenure_group(tenure)

    service_values = [
        phone_service,
        multiple_lines,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies
    ]

    total_services = sum(
        value == "Yes"
        for value in service_values
    )

    if tenure > 0:
        average_monthly_charges = total_charges / tenure
    else:
        average_monthly_charges = monthly_charges

    is_long_term_contract = int(
        contract != "Month-to-month"
    )

    st.divider()

    predict_button = st.button(
        "🤖 Analyze Customer Risk",
        use_container_width=True
    )

    if predict_button:

        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "TenureGroup": tenure_group,
            "TotalServices": total_services,
            "AverageMonthlyCharges": average_monthly_charges,
            "IsLongTermContract": is_long_term_contract
        }])

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        except Exception as e:
            st.error("Prediksi gagal. Pastikan model cocok dengan fitur input.")
            st.code(str(e))
            st.stop()

        if probability >= 0.70:
            risk_level = "HIGH"
        elif probability >= 0.40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        st.divider()

        st.subheader("🎯 Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            if prediction == 1:
                st.error("⚠️ CUSTOMER LIKELY TO CHURN")
            else:
                st.success("✅ CUSTOMER LIKELY TO STAY")

        with col2:
            st.metric(
                "Churn Probability",
                f"{probability * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Risk Level",
                risk_level
            )

        st.write("Churn Probability")
        st.progress(float(probability))

        st.subheader("💡 Business Recommendation")

        if risk_level == "HIGH":
            st.error("""
            **High Risk Customer**

            Customer memiliki probabilitas churn yang tinggi.

            **Recommended actions:**
            - Jalankan retention campaign.
            - Berikan personalized promotion.
            - Tawarkan incentive untuk kontrak jangka panjang.
            - Tingkatkan customer engagement.
            """)

        elif risk_level == "MEDIUM":
            st.warning("""
            **Medium Risk Customer**

            Customer perlu mendapatkan monitoring.

            **Recommended actions:**
            - Monitor customer secara berkala.
            - Tingkatkan engagement.
            - Berikan penawaran yang relevan.
            """)

        else:
            st.success("""
            **Low Risk Customer**

            Customer memiliki risiko churn yang relatif rendah.

            **Recommended actions:**
            - Pertahankan kualitas layanan.
            - Pertahankan customer engagement.
            - Pertimbangkan loyalty program.
            """)

        st.subheader("📋 Customer Profile")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Gender:** {gender}")
            st.write(f"**Tenure:** {tenure} months")
            st.write(f"**Contract:** {contract}")
            st.write(f"**Monthly Charges:** ${monthly_charges:.2f}")

        with col2:
            st.write(f"**Payment Method:** {payment_method}")
            st.write(f"**Total Services:** {total_services}")
            st.write(f"**Tenure Group:** {tenure_group}")
            st.write(
                f"**Long-term Contract:** "
                f"{'Yes' if is_long_term_contract else 'No'}"
            )
