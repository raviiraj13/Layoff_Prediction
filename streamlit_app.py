"""
Layoff Intelligence & Prediction Platform
==========================================
A Streamlit dashboard for exploring the tech layoffs dataset and predicting
which layoff events are likely to be "high impact" (>= 20% of workforce cut).

Run:
    pip install streamlit pandas numpy scikit-learn plotly xgboost
    streamlit run streamlit_app.py

Expects `tech_layoffs_til_2025.csv` in the same folder (or upload it in the sidebar).
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_PATH = "tech_layoffs_til_2025.csv"
HIGH_IMPACT_THRESHOLD = 20  # % of workforce

st.set_page_config(
    page_title="Layoff Intelligence & Prediction Platform",
    page_icon="📉",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    for c in ["Company_Size_before_Layoffs", "Company_Size_after_layoffs"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")

    df["Date_layoffs"] = pd.to_datetime(df["Date_layoffs"], errors="coerce")
    df["Month"] = df["Date_layoffs"].dt.month
    df["MonthName"] = df["Date_layoffs"].dt.strftime("%b")
    df["YearMonth"] = df["Date_layoffs"].dt.to_period("M").astype(str)

    for c in ["Industry", "Country", "Stage", "Continent", "Region"]:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()

    df["Laid_Off"] = pd.to_numeric(df["Laid_Off"], errors="coerce")
    df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce")
    df["Money_Raised_in__mil"] = pd.to_numeric(df["Money_Raised_in__mil"], errors="coerce")

    def size_bucket(x):
        if pd.isna(x):
            return "Unknown"
        if x < 50:
            return "0-50"
        if x < 200:
            return "50-200"
        if x < 1000:
            return "200-1000"
        if x < 5000:
            return "1000-5000"
        return "5000+"

    df["Size_Bucket"] = df["Company_Size_before_Layoffs"].apply(size_bucket)
    return df


# ---------------------------------------------------------------------------
# ML: train (cached) so it only runs once per session
# ---------------------------------------------------------------------------

FEATURES_NUM = ["Company_Size_before_Layoffs", "Money_Raised_in__mil", "Year", "Month"]
FEATURES_CAT = ["Industry", "Country", "Stage", "Continent"]


@st.cache_resource
def train_models(df: pd.DataFrame):
    ml_df = df.dropna(subset=["Percentage"]).copy()
    ml_df["High_Impact"] = (ml_df["Percentage"] >= HIGH_IMPACT_THRESHOLD).astype(int)

    for c in FEATURES_NUM:
        ml_df[c] = ml_df[c].fillna(ml_df[c].median())

    X = ml_df[FEATURES_NUM + FEATURES_CAT]
    y = ml_df["High_Impact"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced", random_state=42
        ),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=42
        )

    fitted, rows, roc_data = {}, [], {}
    for name, model in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", model)])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "F1 Score": f1_score(y_test, pred),
            "ROC AUC": roc_auc_score(y_test, proba),
        })
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, proba))

    results_df = pd.DataFrame(rows).sort_values("ROC AUC", ascending=False).reset_index(drop=True)
    best_name = results_df.iloc[0]["Model"]

    # Feature importance for the best model
    best_pipe = fitted[best_name]
    clf = best_pipe.named_steps["clf"]
    prep = best_pipe.named_steps["prep"]
    cat_names = prep.named_transformers_["cat"].get_feature_names_out(FEATURES_CAT)
    all_names = FEATURES_NUM + list(cat_names)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
    else:
        importances = np.zeros(len(all_names))

    fi_df = pd.DataFrame({"Feature": all_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(15)

    return {
        "fitted": fitted,
        "results": results_df,
        "roc_data": roc_data,
        "best_name": best_name,
        "feature_importance": fi_df,
    }


# ---------------------------------------------------------------------------
# Sidebar: data source + filters
# ---------------------------------------------------------------------------

st.sidebar.title("📉 Layoff Intelligence")
uploaded = st.sidebar.file_uploader("Upload layoffs CSV", type="csv")

try:
    df_raw = load_data(uploaded if uploaded is not None else DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DATA_PATH}`. Upload the CSV using the sidebar, "
        "or place it next to this script."
    )
    st.stop()

st.sidebar.markdown("### Filters")

years = sorted(df_raw["Year"].dropna().unique().tolist())
industries = sorted(df_raw["Industry"].unique().tolist())
countries = sorted(df_raw["Country"].unique().tolist())
stages = sorted(df_raw["Stage"].unique().tolist())

sel_years = st.sidebar.multiselect("Year", years, default=years)
sel_industries = st.sidebar.multiselect("Industry", industries, default=[])
sel_countries = st.sidebar.multiselect("Country", countries, default=[])
sel_stages = st.sidebar.multiselect("Funding Stage", stages, default=[])

df = df_raw[df_raw["Year"].isin(sel_years)]
if sel_industries:
    df = df[df["Industry"].isin(sel_industries)]
if sel_countries:
    df = df[df["Country"].isin(sel_countries)]
if sel_stages:
    df = df[df["Stage"].isin(sel_stages)]

if df.empty:
    st.warning("No rows match the current filters. Adjust the filters in the sidebar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(df):,} of {len(df_raw):,} events shown")

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------

st.title("Layoff Intelligence & Prediction Platform")
st.caption("Explore tech-industry layoff trends and predict high-impact events.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Layoff Events", f"{len(df):,}")
k2.metric("People Laid Off", f"{int(df['Laid_Off'].sum(skipna=True)):,}")
k3.metric("Companies", f"{df['Company'].nunique():,}")
avg_pct = df["Percentage"].mean(skipna=True)
k4.metric("Avg % of Workforce Cut", f"{avg_pct:.1f}%" if pd.notna(avg_pct) else "—")

st.markdown("---")

tabs = st.tabs([
    "📈 Trends", "🏭 Industry", "🌍 Country", "💰 Stage & Size",
    "📊 Distributions", "🔎 Company Search", "🤖 ML Prediction", "📋 Data Explorer",
])

# ---------------------------------------------------------------------------
# Tab: Trends over time
# ---------------------------------------------------------------------------

with tabs[0]:
    st.subheader("Layoffs Over Time")

    yearly = df.groupby("Year").agg(Events=("Company", "count"),
                                     People=("Laid_Off", "sum")).reset_index()

    fig = go.Figure()
    fig.add_bar(x=yearly["Year"], y=yearly["People"], name="People laid off",
                marker_color="#4C72B0", yaxis="y1")
    fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["Events"], name="Events",
                              mode="lines+markers", marker_color="#DD8452", yaxis="y2"))
    fig.update_layout(
        yaxis=dict(title="People laid off"),
        yaxis2=dict(title="Events", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly Trend")
    monthly = df.groupby("YearMonth").agg(Events=("Company", "count"),
                                           People=("Laid_Off", "sum")).reset_index()
    fig2 = px.line(monthly, x="YearMonth", y="People", markers=True)
    fig2.update_layout(xaxis_title="Month", yaxis_title="People laid off",
                        xaxis_tickangle=-90)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top Companies Affected")
    top_n = st.slider("Number of companies", 5, 30, 15)
    top_companies = (df.groupby("Company")
                      .agg(Events=("Company", "count"), Total_Laid_Off=("Laid_Off", "sum"))
                      .sort_values("Total_Laid_Off", ascending=False).head(top_n).reset_index())
    fig3 = px.bar(top_companies.sort_values("Total_Laid_Off"),
                  x="Total_Laid_Off", y="Company", orientation="h",
                  color="Total_Laid_Off", color_continuous_scale="Blues")
    fig3.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab: Industry
# ---------------------------------------------------------------------------

with tabs[1]:
    st.subheader("Layoffs by Industry")
    industry = (df.groupby("Industry")
                .agg(Events=("Company", "count"), People=("Laid_Off", "sum"),
                     Avg_Pct=("Percentage", "mean"))
                .sort_values("People", ascending=False).reset_index())

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(industry.head(15).sort_values("People"), x="People", y="Industry",
                     orientation="h", color="People", color_continuous_scale="Blues")
        fig.update_layout(coloraxis_showscale=False, title="Top Industries by People Laid Off")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(industry.head(10), names="Industry", values="Events", hole=0.45,
                     title="Share of Events by Industry (Top 10)")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(industry, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab: Country
# ---------------------------------------------------------------------------

with tabs[2]:
    st.subheader("Layoffs by Country")
    country = (df.groupby("Country")
               .agg(Events=("Company", "count"), People=("Laid_Off", "sum"))
               .sort_values("People", ascending=False).reset_index())

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(country.head(15).sort_values("People"), x="People", y="Country",
                     orientation="h", color="People", color_continuous_scale="Greens")
        fig.update_layout(coloraxis_showscale=False, title="Top Countries by People Laid Off")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        map_df = df.dropna(subset=["latitude", "longitude"])
        fig = px.scatter_geo(map_df, lat="latitude", lon="longitude", size="Laid_Off",
                              hover_name="Company", color="Continent", opacity=0.6)
        fig.update_layout(title="Layoff Locations (bubble size = people laid off)")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(country, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab: Stage & Size
# ---------------------------------------------------------------------------

with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Funding-Stage Analysis")
        stage = (df.groupby("Stage")
                 .agg(Events=("Company", "count"), People=("Laid_Off", "sum"),
                      Avg_Pct=("Percentage", "mean"))
                 .sort_values("People", ascending=False).reset_index())
        fig = px.bar(stage.sort_values("People"), x="People", y="Stage",
                     orientation="h", color="Avg_Pct", color_continuous_scale="Reds",
                     labels={"Avg_Pct": "Avg % cut"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(stage, use_container_width=True, hide_index=True)

    with c2:
        st.subheader("Company-Size Analysis")
        size_order = ["0-50", "50-200", "200-1000", "1000-5000", "5000+", "Unknown"]
        size_stats = (df.groupby("Size_Bucket")
                      .agg(Events=("Company", "count"), Avg_Pct=("Percentage", "mean"))
                      .reindex(size_order).dropna(how="all").reset_index())
        fig = px.bar(size_stats, x="Size_Bucket", y="Avg_Pct",
                     labels={"Avg_Pct": "Avg % of workforce cut"},
                     color="Avg_Pct", color_continuous_scale="Purples")
        fig.update_layout(coloraxis_showscale=False,
                           title="Avg % Laid Off by Pre-Layoff Company Size")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(size_stats, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Tab: Distributions
# ---------------------------------------------------------------------------

with tabs[4]:
    st.subheader("Layoff Percentage Distribution")
    fig = px.histogram(df.dropna(subset=["Percentage"]), x="Percentage", nbins=30,
                        color_discrete_sequence=["#4C72B0"])
    fig.update_layout(xaxis_title="% of workforce laid off", yaxis_title="Events")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By Industry (box plot)")
        top_ind = df["Industry"].value_counts().head(8).index
        fig = px.box(df[df["Industry"].isin(top_ind)], x="Industry", y="Percentage")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("People Laid Off Distribution")
        fig = px.histogram(df.dropna(subset=["Laid_Off"]), x="Laid_Off", nbins=40,
                            color_discrete_sequence=["#DD8452"])
        fig.update_xaxes(range=[0, df["Laid_Off"].quantile(0.95)])
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab: Company search
# ---------------------------------------------------------------------------

with tabs[5]:
    st.subheader("Company Search & History")
    query = st.text_input("Search for a company", "")
    if query:
        sub = df_raw[df_raw["Company"].str.contains(query, case=False, na=False)]
        if sub.empty:
            st.info("No matches. Try a shorter or different spelling.")
        else:
            companies_found = sorted(sub["Company"].unique())
            picked = st.selectbox("Matches", companies_found)
            hist = sub[sub["Company"] == picked].sort_values("Date_layoffs")

            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Layoff Events", len(hist))
            hc2.metric("Total People Laid Off", f"{int(hist['Laid_Off'].sum(skipna=True)):,}")
            avg = hist["Percentage"].mean()
            hc3.metric("Avg % Workforce Cut", f"{avg:.1f}%" if pd.notna(avg) else "—")

            st.dataframe(
                hist[["Date_layoffs", "Laid_Off", "Percentage",
                      "Company_Size_before_Layoffs", "Company_Size_after_layoffs",
                      "Industry", "Stage", "Country"]],
                use_container_width=True, hide_index=True,
            )

            if len(hist) > 1:
                fig = px.line(hist, x="Date_layoffs", y="Laid_Off", markers=True,
                              title=f"{picked} — Layoffs Over Time")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Type a company name above to see its full layoff history.")

# ---------------------------------------------------------------------------
# Tab: ML Prediction
# ---------------------------------------------------------------------------

with tabs[6]:
    st.subheader("High-Impact Layoff Prediction")
    st.caption(
        f"A layoff event is labeled **high impact** when it cuts "
        f"**{HIGH_IMPACT_THRESHOLD}%+** of the company's workforce. "
        "Models are trained once per session on the full (unfiltered) dataset."
    )

    with st.spinner("Training models..."):
        ml = train_models(df_raw)

    st.markdown("### Model Comparison")
    st.dataframe(
        ml["results"].style.format({"Accuracy": "{:.2%}", "F1 Score": "{:.2f}", "ROC AUC": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )
    if not HAS_XGB:
        st.info("Install `xgboost` (`pip install xgboost`) to include it in the comparison.")

    fig = go.Figure()
    for name, (fpr, tpr, auc) in ml["roc_data"].items():
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(dash="dash", color="gray"), showlegend=False))
    fig.update_layout(title="ROC Curve — Model Comparison",
                       xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"### Feature Importance — {ml['best_name']} (best model)")
    fig = px.bar(ml["feature_importance"].sort_values("Importance"),
                 x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Try a Prediction")
    st.caption("Estimate the odds a hypothetical layoff event would be high-impact.")
    p1, p2, p3 = st.columns(3)
    with p1:
        in_industry = st.selectbox("Industry", sorted(df_raw["Industry"].unique()))
        in_country = st.selectbox("Country", sorted(df_raw["Country"].unique()))
    with p2:
        in_stage = st.selectbox("Funding Stage", sorted(df_raw["Stage"].unique()))
        in_continent = st.selectbox("Continent", sorted(df_raw["Continent"].unique()))
    with p3:
        in_size = st.number_input("Company size before layoffs", min_value=1, value=500)
        in_money = st.number_input("Money raised ($M)", min_value=0.0, value=50.0)

    model_choice = st.selectbox("Model to use", list(ml["fitted"].keys()),
                                 index=list(ml["fitted"].keys()).index(ml["best_name"]))
    if st.button("Predict"):
        row = pd.DataFrame([{
            "Company_Size_before_Layoffs": in_size,
            "Money_Raised_in__mil": in_money,
            "Year": 2026,
            "Month": 1,
            "Industry": in_industry,
            "Country": in_country,
            "Stage": in_stage,
            "Continent": in_continent,
      
