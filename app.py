import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📊",
    layout="wide"
)


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
    return joblib.load("churn_model.pkl")


df = load_data()
model = load_model()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

# ---------------------------------------------------------
# 1. Tenure Group
# ---------------------------------------------------------

def create_tenure_group(tenure):

    if tenure <= 12:
        return "New"

    elif tenure <= 24:
        return "Early"

    elif tenure <= 48:
        return "Mid"

    else:
        return "Loyal"


df["TenureGroup"] = df["tenure"].apply(
    create_tenure_group
)


# ---------------------------------------------------------
# 2. Total Services
# ---------------------------------------------------------

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


df["TotalServices"] = (
    df[service_columns] == "Yes"
).sum(axis=1)


# ---------------------------------------------------------
# 3. Total Charges
# ---------------------------------------------------------

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# ---------------------------------------------------------
# 4. Average Monthly Charges
# ---------------------------------------------------------

df["AverageMonthlyCharges"] = (
    df["TotalCharges"] /
    df["tenure"].replace(0, 1)
)


# ---------------------------------------------------------
# 5. Long Term Contract
# ---------------------------------------------------------

df["IsLongTermContract"] = (
    df["Contract"] != "Month-to-month"
).astype(int)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Telco Churn Analytics")

st.sidebar.markdown(
    """
    ### Customer Churn Prediction

    Machine Learning application untuk:

    - Analisis customer churn
    - Business insights
    - Prediksi risiko churn
    - Customer retention
    """
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Churn Analytics",
        "🔮 Customer Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Built with Python • Pandas • Scikit-learn • Streamlit"
)


# =========================================================
# PAGE 1 — OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.title("📊 Telco Customer Churn Analytics")

    st.markdown(
        """
        ### Customer Churn Prediction System

        Aplikasi ini menggunakan **Machine Learning**
        untuk menganalisis perilaku customer dan
        memprediksi kemungkinan customer melakukan churn.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # KPI
    # -----------------------------------------------------

    total_customers = len(df)

    churn_customers = (
        df["Churn"] == "Yes"
    ).sum()

    churn_rate = (
        churn_customers /
        total_customers
    ) * 100

    avg_monthly_charges = (
        df["MonthlyCharges"].mean()
    )

    avg_tenure = (
        df["tenure"].mean()
    )

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
            "Avg Monthly Charges",
            f"${avg_monthly_charges:.2f}"
        )

    st.divider()

    # -----------------------------------------------------
    # CHARTS
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # Churn Distribution
    # -----------------------------------------------------

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
            hole=0.5,
            title="Customer Churn Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------------------
    # Contract Churn
    # -----------------------------------------------------

    with col2:

        contract_churn = (
            df.groupby("Contract")["Churn"]
            .apply(
                lambda x:
                (x == "Yes").mean() * 100
            )
            .reset_index(
                name="ChurnRate"
            )
        )

        fig = px.bar(
            contract_churn,
            x="Contract",
            y="ChurnRate",
            text_auto=".1f",
            title="Churn Rate by Contract"
        )

        fig.update_yaxes(
            title="Churn Rate (%)"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # -----------------------------------------------------
    # BUSINESS INSIGHTS
    # -----------------------------------------------------

    st.subheader("💡 Key Business Insights")

    tenure_churn = (
        df.groupby("TenureGroup")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
    )

    highest_tenure_group = (
        tenure_churn.idxmax()
    )

    highest_tenure_rate = (
        tenure_churn.max()
    )

    payment_churn = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
    )

    highest_payment = (
        payment_churn.idxmax()
    )

    highest_payment_rate = (
        payment_churn.max()
    )

    st.info(
        f"""
        **Customer Tenure**

        Kelompok **{highest_tenure_group}**
        memiliki churn rate tertinggi,
        yaitu sekitar **{highest_tenure_rate:.2f}%**.

        **Payment Method**

        Metode pembayaran **{highest_payment}**
        memiliki churn rate tertinggi,
        yaitu sekitar **{highest_payment_rate:.2f}%**.

        **Average Tenure**

        Rata-rata customer telah menggunakan layanan
        selama sekitar **{avg_tenure:.1f} bulan**.
        """
    )


# =========================================================
# PAGE 2 — CHURN ANALYTICS
# =========================================================

elif page == "📊 Churn Analytics":

    st.title("📊 Churn Analytics")

    st.markdown(
        """
        Halaman ini digunakan untuk mengeksplorasi
        faktor-faktor yang berkaitan dengan customer churn.
        """
    )

    st.divider()

    # -----------------------------------------------------
    # TENURE GROUP
    # -----------------------------------------------------

    st.subheader(
        "📈 Churn Rate by Tenure Group"
    )

    tenure_churn = (
        df.groupby("TenureGroup")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(
            name="ChurnRate"
        )
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

    tenure_churn = (
        tenure_churn
        .sort_values("TenureGroup")
    )

    fig = px.bar(
        tenure_churn,
        x="TenureGroup",
        y="ChurnRate",
        text_auto=".1f",
        title="Customer Churn Rate by Tenure Group"
    )

    fig.update_yaxes(
        title="Churn Rate (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # MONTHLY CHARGES
    # -----------------------------------------------------

    st.subheader(
        "💰 Monthly Charges vs Churn"
    )

    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges Distribution by Churn"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # PAYMENT METHOD
    # -----------------------------------------------------

    st.subheader(
        "💳 Churn Rate by Payment Method"
    )

    payment_churn = (
        df.groupby("PaymentMethod")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(
            name="ChurnRate"
        )
        .sort_values(
            "ChurnRate",
            ascending=True
        )
    )

    fig = px.bar(
        payment_churn,
        x="ChurnRate",
        y="PaymentMethod",
        orientation="h",
        text_auto=".1f",
        title="Churn Rate by Payment Method"
    )

    fig.update_xaxes(
        title="Churn Rate (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # TOTAL SERVICES
    # -----------------------------------------------------

    st.subheader(
        "🛠️ Churn Rate by Number of Services"
    )

    services_churn = (
        df.groupby("TotalServices")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(
            name="ChurnRate"
        )
    )

    fig = px.line(
        services_churn,
        x="TotalServices",
        y="ChurnRate",
        markers=True,
        title="Churn Rate by Number of Services"
    )

    fig.update_xaxes(
        title="Total Services"
    )

    fig.update_yaxes(
        title="Churn Rate (%)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # -----------------------------------------------------
    # MONTHLY CHARGES HISTOGRAM
    # -----------------------------------------------------

    st.subheader(
        "💵 Monthly Charges Distribution"
    )

    fig = px.histogram(
        df,
        x="MonthlyCharges",
        color="Churn",
        nbins=40,
        title="Monthly Charges Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PAGE 3 — CUSTOMER PREDICTION
# =========================================================

elif page == "🔮 Customer Prediction":

    st.title(
        "🔮 Customer Churn Prediction"
    )

    st.markdown(
        """
        Masukkan informasi customer untuk
        memprediksi kemungkinan customer melakukan churn.
        """
    )

    st.divider()

    # =====================================================
    # CUSTOMER INFORMATION
    # =====================================================

    st.subheader(
        "👤 Customer Information"
    )

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------------------
    # COLUMN 1
    # -----------------------------------------------------

    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            [
                "Yes",
                "No"
            ]
        )

        dependents = st.selectbox(
            "Dependents",
            [
                "Yes",
                "No"
            ]
        )

    # -----------------------------------------------------
    # COLUMN 2
    # -----------------------------------------------------

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
            value=float(
                monthly_charges * tenure
            ),
            step=10.0
        )

    # -----------------------------------------------------
    # COLUMN 3
    # -----------------------------------------------------

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
            [
                "Yes",
                "No"
            ]
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

    # =====================================================
    # SERVICES
    # =====================================================

    st.subheader(
        "🛠️ Customer Services"
    )

    col1, col2, col3 = st.columns(3)

    # -----------------------------------------------------
    # COLUMN 1
    # -----------------------------------------------------

    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            [
                "No",
                "Yes"
            ]
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

    # -----------------------------------------------------
    # COLUMN 2
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # COLUMN 3
    # -----------------------------------------------------

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

    # =====================================================
    # FEATURE ENGINEERING FOR CUSTOMER INPUT
    # =====================================================

    if tenure <= 12:

        tenure_group = "New"

    elif tenure <= 24:

        tenure_group = "Early"

    elif tenure <= 48:

        tenure_group = "Mid"

    else:

        tenure_group = "Loyal"


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

        average_monthly_charges = (
            total_charges / tenure
        )

    else:

        average_monthly_charges = (
            monthly_charges
        )


    is_long_term_contract = int(
        contract != "Month-to-month"
    )


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    st.divider()

    predict_button = st.button(
        "🔮 Predict Customer Churn",
        use_container_width=True
    )


    if predict_button:

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame([{

            "gender": gender,

            "SeniorCitizen":
                senior_citizen,

            "Partner":
                partner,

            "Dependents":
                dependents,

            "tenure":
                tenure,

            "PhoneService":
                phone_service,

            "MultipleLines":
                multiple_lines,

            "InternetService":
                internet_service,

            "OnlineSecurity":
                online_security,

            "OnlineBackup":
                online_backup,

            "DeviceProtection":
                device_protection,

            "TechSupport":
                tech_support,

            "StreamingTV":
                streaming_tv,

            "StreamingMovies":
                streaming_movies,

            "Contract":
                contract,

            "PaperlessBilling":
                paperless_billing,

            "PaymentMethod":
                payment_method,

            "MonthlyCharges":
                monthly_charges,

            "TotalCharges":
                total_charges,

            "TenureGroup":
                tenure_group,

            "TotalServices":
                total_services,

            "AverageMonthlyCharges":
                average_monthly_charges,

            "IsLongTermContract":
                is_long_term_contract
        }])


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        probability = model.predict_proba(
            input_data
        )[0][1]


        # =================================================
        # DETERMINE RISK
        # =================================================

        if probability >= 0.70:

            risk_level = "HIGH"

        elif probability >= 0.40:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader(
            "🎯 Prediction Result"
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            if prediction == 1:

                st.error(
                    "⚠️ CUSTOMER LIKELY TO CHURN"
                )

            else:

                st.success(
                    "✅ CUSTOMER LIKELY TO STAY"
                )


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


        # -------------------------------------------------
        # PROBABILITY BAR
        # -------------------------------------------------

        st.write(
            "Churn Probability"
        )

        st.progress(
            float(probability)
        )


        # =================================================
        # BUSINESS RECOMMENDATION
        # =================================================

        st.subheader(
            "💡 Business Recommendation"
        )


        if risk_level == "HIGH":

            st.error(
                """
                **High Risk Customer**

                Customer memiliki kemungkinan churn yang tinggi.

                **Recommended Actions:**

                - Jalankan retention campaign.
                - Berikan personalized promotion.
                - Tawarkan discount atau benefit khusus.
                - Tawarkan upgrade ke kontrak jangka panjang.
                - Tingkatkan customer engagement.
                """
            )


        elif risk_level == "MEDIUM":

            st.warning(
                """
                **Medium Risk Customer**

                Customer memiliki risiko churn yang perlu dimonitor.

                **Recommended Actions:**

                - Monitor customer secara berkala.
                - Tingkatkan engagement.
                - Berikan penawaran yang relevan.
                - Evaluasi pengalaman customer.
                """
            )


        else:

            st.success(
                """
                **Low Risk Customer**

                Customer memiliki risiko churn yang relatif rendah.

                **Recommended Actions:**

                - Pertahankan kualitas layanan.
                - Pertahankan customer engagement.
                - Pertimbangkan loyalty program.
                """
            )


        # =================================================
        # CUSTOMER PROFILE
        # =================================================

        st.subheader(
            "📋 Customer Profile"
        )

        col1, col2 = st.columns(2)


        with col1:

            st.write(
                f"**Gender:** {gender}"
            )

            st.write(
                f"**Tenure:** {tenure} months"
            )

            st.write(
                f"**Contract:** {contract}"
            )

            st.write(
                f"**Monthly Charges:** "
                f"${monthly_charges:.2f}"
            )


        with col2:

            st.write(
                f"**Payment Method:** "
                f"{payment_method}"
            )

            st.write(
                f"**Total Services:** "
                f"{total_services}"
            )

            st.write(
                f"**Tenure Group:** "
                f"{tenure_group}"
            )

            st.write(
                f"**Long-term Contract:** "
                f"{'Yes' if is_long_term_contract else 'No'}"
            )


# =========================================================
# FOOTER
# =========================================================



st.sidebar.divider()

st.sidebar.caption(
    "Telco Customer Churn Prediction • Data Science Portfolio"
)