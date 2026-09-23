import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, confusion_matrix

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
st.caption("Predicts the number of employees a company might lay off.")

REGRESSORS = {
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
}

reg_algo_name = st.selectbox("Choose regression algorithm", list(REGRESSORS.keys()))

with st.form("size_form"):
    row_size, in_size = company_input_form("size")
    submitted_size = st.form_submit_button("Predict layoff size")

if submitted_size:
    y_reg = df["Laid_Off"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    reg = REGRESSORS[reg_algo_name]
    reg.fit(X_train, y_train)
    r2 = r2_score(y_test, reg.predict(X_test))

    reg.fit(X, y_reg)  # refit on all data for the actual prediction
    pred_layoffs = max(0, reg.predict(row_size)[0])
    pred_pct = min(100, pred_layoffs / in_size * 100)

    r1, r2_col, r3 = st.columns(3)
    r1.metric("Predicted employees laid off", f"{int(pred_layoffs):,}")
    r2_col.metric("Predicted % of workforce", f"{pred_pct:.1f}%")
    r3.metric("R² score (test set)", f"{r2:.2f}")
    st.caption(f"Model used: {reg_algo_name}. R² measures how well the model explains variance in "
               "held-out data — 1.0 is a perfect fit, 0 means no better than predicting the average, "
               "negative means worse than that.")

st.divider()

# ============================================================
# SECTION 2: Severity prediction (classification, choice of algorithm)
# ============================================================
st.header("2️⃣ Layoff Scale Prediction")
st.caption("Classifies expected layoffs as Small / Moderate / Large (terciles of historical layoff counts).")

df["Layoff_Scale"] = pd.qcut(df["Laid_Off"], q=3, labels=["Small", "Moderate", "Large"])

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
    submitted_sev = st.form_submit_button("Predict layoff scale")

if submitted_sev:
    y_clf = df["Layoff_Scale"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    clf = ALGORITHMS[algo_name]
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)

    clf.fit(X, y_clf)  # refit on all data for the actual prediction
    pred_class = clf.predict(row_sev)[0]
    st.metric("Predicted layoff scale", pred_class)

    st.subheader("Confusion Matrix")
    labels = ["Small", "Moderate", "Large"]
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual: {l}" for l in labels], columns=[f"Predicted: {l}" for l in labels])
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
    st.plotly_chart(fig, use_container_width=True)
    accuracy = (y_pred_test == y_test).mean()
    st.caption(f"Model used: {algo_name}. Test accuracy: {accuracy:.1%}. Rows are actual layoff scale, "
               "columns are what the model predicted — the diagonal is correct predictions.")
    st.caption(f"Model used: {algo_name}")


st.caption("Trained on historical data — simplified estimates, not real forecasts.")

st.header("LayOff Dataset")
st.dataframe(df)
