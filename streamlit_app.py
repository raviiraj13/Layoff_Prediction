import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

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
st.subheader("Raw Data")
st.dataframe(filtered, use_container_width=True)
