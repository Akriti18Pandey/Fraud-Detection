import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Enterprise Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PROFESSIONAL BANK BACKGROUND ----------------
st.markdown("""
<style>
.stApp {
    background: url("https://images.unsplash.com/photo-1554224155-6726b3ff858f")
                no-repeat center center fixed;
    background-size: cover;
    font-family: 'Segoe UI', sans-serif;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(255,255,255,0.90);
    z-index: 0;
}

.block-container {
    position: relative;
    z-index: 1;
}

.header {
    background: linear-gradient(135deg,#1d4ed8,#1e3a8a);
    padding: 22px;
    border-radius: 14px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.fraud {
    background: #dc2626;
    padding: 25px;
    border-radius: 14px;
    color: white;
    font-size: 28px;
    text-align: center;
    font-weight: bold;
}

.safe {
    background: #16a34a;
    padding: 25px;
    border-radius: 14px;
    color: white;
    font-size: 28px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:

    st.title("🔐 Secure Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "Akriti" and password == "@kritiii":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# ---------------- LOAD DATA ----------------
df = pd.read_csv("https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset?resource=download")
df = df.drop(['nameOrig', 'nameDest'], axis=1)

df['type'] = df['type'].map({
    'PAYMENT': 0,
    'TRANSFER': 1,
    'CASH_OUT': 2,
    'DEBIT': 3,
    'CASH_IN': 4
})

X = df.drop('isFraud', axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🛡️ Enterprise Panel")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "💳 Predict Fraud", "📜 History", "📈 Analytics"]
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
<h2>AI Fraud Detection Platform</h2>
<p>Enterprise Financial Security System</p>
</div>
""", unsafe_allow_html=True)

# ================= DASHBOARD =================
if page == "📊 Dashboard":

    st.subheader("System Overview")

    m1, m2, m3 = st.columns(3)
    m1.metric("Dataset Size", f"{len(df):,}")
    m2.metric("Model", "Random Forest")
    m3.metric("Accuracy", f"{accuracy:.2%}")

    st.markdown("""
    This platform detects suspicious financial transactions using
    machine learning models in real time.
    """)

# ================= PREDICTION =================
elif page == "💳 Predict Fraud":

    st.subheader("Transaction Details")

    c1, c2 = st.columns(2)

    with c1:
        type_name = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
        )
        amount = st.number_input("Amount", 0.0, value=1000.0)
        oldbalanceOrg = st.number_input("Old Balance (Origin)", value=0.0)

    with c2:
        newbalanceOrig = st.number_input("New Balance (Origin)", value=0.0)
        oldbalanceDest = st.number_input("Old Balance (Destination)", value=0.0)
        newbalanceDest = st.number_input("New Balance (Destination)", value=0.0)

    if st.button("Analyze Transaction"):

        type_map = {
            "PAYMENT": 0,
            "TRANSFER": 1,
            "CASH_OUT": 2,
            "DEBIT": 3,
            "CASH_IN": 4
        }

        new_data = [[
            type_map[type_name],
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest
        ]]

        prediction = model.predict(new_data)[0]
        prob = model.predict_proba(new_data)[0][1]

        st.subheader("Analysis Result")

        st.progress(float(prob))

        if prediction == 1:
            st.markdown('<div class="fraud">🚨 FRAUD DETECTED</div>', unsafe_allow_html=True)
            result = "Fraud"
        else:
            st.markdown('<div class="safe">✅ Legitimate</div>', unsafe_allow_html=True)
            result = "Legitimate"

        # Save to history
        st.session_state.history.append({
            "Type": type_name,
            "Amount": amount,
            "Result": result,
            "Risk": prob
        })

# ================= HISTORY =================
elif page == "📜 History":

    st.subheader("Transaction History")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No transactions analyzed yet.")

# ================= ANALYTICS =================
elif page == "📈 Analytics":

    st.subheader("Fraud Analytics")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        fraud_count = history_df["Result"].value_counts()

        fig, ax = plt.subplots()
        ax.pie(fraud_count, labels=fraud_count.index, autopct="%1.1f%%")
        ax.set_title("Fraud vs Legitimate")

        st.pyplot(fig)

    else:

        st.info("No data available for analytics.")
