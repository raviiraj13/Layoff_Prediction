import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Tech Layoffs Dashboard", layout="wide")

# ---------- Load data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("tech_layoffs_til_2025.csv")
    df["Date_layoffs"] = pd.to_datetime(df["Date_layoffs"], errors="coerce")
    df["Laid_Off"] = pd.to_numeric(df["Laid_Off"], errors="coerce")
    return df

df = load_data()

st.title("📉 Tech Layoffs Dashboard (2020–2025)")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
years = sorted(df["Year"].dropna().unique())
year_range = st.sidebar.slider("Year range", int(min(years)), int(max(years)),
                                (int(min(years)), int(max(years))))

industries = sorted(df["Industry"].dropna().unique())
selected_industries = st.sidebar.multiselect("Industry", industries, default=industries)

filtered = df[
    (df["Year"] >= year_range[0]) & (df["Year"] <= year_range[1]) &
    (df["Industry"].isin(selected_industries))
]

# ---------- KPIs ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Layoffs", f"{int(filtered['Laid_Off'].sum()):,}")
col2.metric("Companies Affected", filtered["Company"].nunique())
col3.metric("Countries", filtered["Country"].nunique())
col4.metric("Avg Layoff %", f"{filtered['Percentage'].mean():.1f}%")

st.divider()

# ---------- Charts ----------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Layoffs by Year")
    yearly = filtered.groupby("Year")["Laid_Off"].sum().reset_index()
    fig = px.bar(yearly, x="Year", y="Laid_Off")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Top 10 Industries")
    ind = filtered.groupby("Industry")["Laid_Off"].sum().nlargest(10).reset_index()
    fig2 = px.bar(ind, x="Laid_Off", y="Industry", orientation="h")
    st.plotly_chart(fig2, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.subheader("Top 10 Companies")
    comp = filtered.groupby("Company")["Laid_Off"].sum().nlargest(10).reset_index()
    fig3 = px.bar(comp, x="Laid_Off", y="Company", orientation="h")
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    st.subheader("Layoffs Over Time")
    timeline = filtered.groupby("Date_layoffs")["Laid_Off"].sum().reset_index()
    fig4 = px.line(timeline, x="Date_layoffs", y="Laid_Off")
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ---------- Simple Prediction ----------
st.subheader("🔮 Predict Next Year's Total Layoffs")

yearly_totals = df.groupby("Year")["Laid_Off"].sum().reset_index().dropna()
X = yearly_totals[["Year"]]
y = yearly_totals["Laid_Off"]

model = LinearRegression()
model.fit(X, y)

next_year = int(yearly_totals["Year"].max()) + 1
prediction = model.predict([[next_year]])[0]

pred_col1, pred_col2 = st.columns([1, 2])
with pred_col1:
    st.metric(f"Predicted layoffs in {next_year}", f"{int(max(prediction, 0)):,}")
    st.caption("Simple linear regression on yearly totals. For learning purposes only — not a real forecast.")

with pred_col2:
    plot_df = yearly_totals.copy()
    plot_df["Type"] = "Actual"
    pred_row = pd.DataFrame({"Year": [next_year], "Laid_Off": [max(prediction, 0)], "Type": ["Predicted"]})
    plot_df = pd.concat([plot_df, pred_row])
    fig5 = px.line(plot_df, x="Year", y="Laid_Off", color="Type", markers=True)
    st.plotly_chart(fig5, use_container_width=True)

st.divider()

# ---------- Prediction Model: estimate layoffs for a company ----------
st.subheader("🧮 Predict Layoffs for a Company")

model_choice = st.radio(
    "Choose a model",
    ["Random Forest (predicts number laid off)", "Logistic Regression (predicts severity class)"],
    horizontal=True,
)

model_df = df.dropna(subset=["Laid_Off", "Company_Size_before_Layoffs", "Industry", "Country", "Stage"]).copy()
model_df["Company_Size_before_Layoffs"] = pd.to_numeric(
    model_df["Company_Size_before_Layoffs"], errors="coerce"
)
model_df = model_df.dropna(subset=["Company_Size_before_Layoffs"])

# Encode categorical features
industry_enc = LabelEncoder().fit(model_df["Industry"])
country_enc = LabelEncoder().fit(model_df["Country"])
stage_enc = LabelEncoder().fit(model_df["Stage"])

model_df["Industry_enc"] = industry_enc.transform(model_df["Industry"])
model_df["Country_enc"] = country_enc.transform(model_df["Country"])
model_df["Stage_enc"] = stage_enc.transform(model_df["Stage"])

features = ["Company_Size_before_Layoffs", "Industry_enc", "Country_enc", "Stage_enc", "Year"]
X_model = model_df[features]

# Severity classes (low/medium/high) from tercile split of Laid_Off, used by Logistic Regression
model_df["Severity"] = pd.qcut(model_df["Laid_Off"], q=3, labels=["Low", "Medium", "High"])

with st.form("predict_form"):
    st.write("Enter company details:")
    f1, f2 = st.columns(2)
    with f1:
        in_size = st.number_input("Company size before layoffs", min_value=1, value=500)
        in_industry = st.selectbox("Industry", sorted(model_df["Industry"].unique()))
        in_year = st.number_input("Year", min_value=2020, max_value=2027, value=2026)
    with f2:
        in_country = st.selectbox("Country", sorted(model_df["Country"].unique()))
        in_stage = st.selectbox("Stage", sorted(model_df["Stage"].unique()))
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([{
        "Company_Size_before_Layoffs": in_size,
        "Industry_enc": industry_enc.transform([in_industry])[0],
        "Country_enc": country_enc.transform([in_country])[0],
        "Stage_enc": stage_enc.transform([in_stage])[0],
        "Year": in_year,
    }])

    if model_choice.startswith("Random Forest"):
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X_model, model_df["Laid_Off"])
        pred_layoffs = rf.predict(row)[0]
        pred_pct = min(100, max(0, pred_layoffs / in_size * 100))
        r1, r2 = st.columns(2)
        r1.metric("Predicted employees laid off", f"{int(pred_layoffs):,}")
        r2.metric("Predicted % of workforce", f"{pred_pct:.1f}%")
        st.caption("Model: RandomForestRegressor on company size, industry, country, stage, and year.")
    else:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_model, model_df["Severity"])
        pred_class = clf.predict(row)[0]
        proba = clf.predict_proba(row)[0]
        classes = clf.classes_
        r1, r2 = st.columns(2)
        r1.metric("Predicted severity", pred_class)
        r2.metric("Confidence", f"{max(proba) * 100:.1f}%")
        proba_df = pd.DataFrame({"Severity": classes, "Probability": proba})
        fig6 = px.bar(proba_df, x="Severity", y="Probability")
        st.plotly_chart(fig6, use_container_width=True)
        st.caption("Model: LogisticRegression classifying layoff severity (Low/Medium/High tercile of "
                   "historical Laid_Off counts) from company size, industry, country, stage, and year.")

    st.caption("Trained on historical data — a simplified estimate, not a real forecast.")

st.divider()
st.subheader("Raw Data")
st.dataframe(filtered, use_container_width=True)
