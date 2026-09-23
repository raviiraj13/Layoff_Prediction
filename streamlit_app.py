import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Tech Layoffs Prediction", layout="wide")
st.title("📉 Tech Layoffs Prediction")

# ---------- Load & prep data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("tech_layoffs_til_2025.csv")
    df["Laid_Off"] = pd.to_numeric(df["Laid_Off"], errors="coerce")
    df["Company_Size_before_Layoffs"] = pd.to_numeric(df["Company_Size_before_Layoffs"], errors="coerce")
    df = df.dropna(subset=["Laid_Off", "Company_Size_before_Layoffs", "Industry", "Country", "Stage", "Year"])
    return df

df = load_data()

industry_enc = LabelEncoder().fit(df["Industry"])
country_enc = LabelEncoder().fit(df["Country"])
stage_enc = LabelEncoder().fit(df["Stage"])

df["Industry_enc"] = industry_enc.transform(df["Industry"])
df["Country_enc"] = country_enc.transform(df["Country"])
df["Stage_enc"] = stage_enc.transform(df["Stage"])

FEATURES = ["Company_Size_before_Layoffs", "Industry_enc", "Country_enc", "Stage_enc", "Year"]
X = df[FEATURES]

# Shared input form for company details
def company_input_form(key_prefix):
    f1, f2 = st.columns(2)
    with f1:
        size = st.number_input("Company size before layoffs", min_value=1, value=500, key=f"{key_prefix}_size")
        industry = st.selectbox("Industry", sorted(df["Industry"].unique()), key=f"{key_prefix}_ind")
        year = st.number_input("Year", min_value=2020, max_value=2027, value=2026, key=f"{key_prefix}_year")
    with f2:
        country = st.selectbox("Country", sorted(df["Country"].unique()), key=f"{key_prefix}_country")
        stage = st.selectbox("Stage", sorted(df["Stage"].unique()), key=f"{key_prefix}_stage")
    row = pd.DataFrame([{
        "Company_Size_before_Layoffs": size,
        "Industry_enc": industry_enc.transform([industry])[0],
        "Country_enc": country_enc.transform([country])[0],
        "Stage_enc": stage_enc.transform([stage])[0],
        "Year": year,
    }])
    return row, size

# ============================================================
# SECTION 1: Layoff size prediction (regression)
# ============================================================
st.header("1️⃣ Layoff Size Prediction")
st.caption("Predicts the number of employees a company might lay off. Model: Random Forest Regressor.")

with st.form("size_form"):
    row_size, in_size = company_input_form("size")
    submitted_size = st.form_submit_button("Predict layoff size")

if submitted_size:
    rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_reg.fit(X, df["Laid_Off"])
    pred_layoffs = max(0, rf_reg.predict(row_size)[0])
    pred_pct = min(100, pred_layoffs / in_size * 100)

    r1, r2 = st.columns(2)
    r1.metric("Predicted employees laid off", f"{int(pred_layoffs):,}")
    r2.metric("Predicted % of workforce", f"{pred_pct:.1f}%")

st.divider()

# ============================================================
# SECTION 2: Severity prediction (classification, choice of algorithm)
# ============================================================
st.header("2️⃣ Severity Prediction")
st.caption("Classifies expected layoffs into Low / Medium / High severity (terciles of historical layoff counts).")

df["Severity"] = pd.qcut(df["Laid_Off"], q=3, labels=["Low", "Medium", "High"])

ALGORITHMS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
}

algo_name = st.selectbox("Choose algorithm", list(ALGORITHMS.keys()))

with st.form("severity_form"):
    row_sev, _ = company_input_form("sev")
    submitted_sev = st.form_submit_button("Predict severity")

if submitted_sev:
    clf = ALGORITHMS[algo_name]
    clf.fit(X, df["Severity"])
    pred_class = clf.predict(row_sev)[0]
    proba = clf.predict_proba(row_sev)[0]
    classes = clf.classes_

    r1, r2 = st.columns(2)
    r1.metric("Predicted severity", pred_class)
    r2.metric("Confidence", f"{max(proba) * 100:.1f}%")

    proba_df = pd.DataFrame({"Severity": classes, "Probability": proba})
    fig = px.bar(proba_df, x="Severity", y="Probability")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Model used: {algo_name}")

st.caption("Trained on historical data — simplified estimates, not real forecasts.")
