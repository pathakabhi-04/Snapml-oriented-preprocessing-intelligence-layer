import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils import load_dataset, run_drift_check, load_contract

st.set_page_config(page_title="SnapML Observability", layout="wide")

st.title("🔍 SnapML Preprocessing Observability")

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "System Overview",
        "Dataset Explorer",
        "Preprocessing Pipeline",
        "Drift Monitoring",
        "Experiment Lab",
    ],
)

df = load_dataset()
contract = load_contract()

# =====================================================
# SYSTEM OVERVIEW
# =====================================================

if page == "System Overview":

    st.header("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Numeric Features", len(contract["data_schema"]["num_indices"]))
    col4.metric("Categorical Features", len(contract["data_schema"]["cat_indices"]))

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), width="stretch")

# =====================================================
# DATASET EXPLORER
# =====================================================

elif page == "Dataset Explorer":

    st.header("Dataset Explorer")

    column = st.selectbox("Select Feature", df.columns)

    if df[column].dtype == "object":

        fig = px.histogram(df, x=column, title=f"Distribution of {column}")
        st.plotly_chart(fig, use_container_width=True)

    else:

        fig = px.histogram(df, x=column, nbins=40, title=f"Distribution of {column}")
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(df[[column]].describe(), width="stretch")

# =====================================================
# PREPROCESSING PIPELINE
# =====================================================

elif page == "Preprocessing Pipeline":

    st.header("SnapML Preprocessing Contract")

    num_cols = contract["data_schema"]["num_indices"]
    cat_cols = contract["data_schema"]["cat_indices"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Numeric Feature Indices")
        st.write(num_cols)

    with col2:
        st.subheader("Categorical Feature Indices")
        st.write(cat_cols)

    st.divider()

    st.subheader("Transformers")

    for name, steps in contract["transformers"].items():

        if not steps:
            continue

        step = steps[0]

        st.markdown(f"### {step['type']}")

        st.write("Columns:", step["columns"])

        if step["type"] == "Normalizer":
            st.info("Applies L2 normalization to numeric features.")

        elif step["type"] == "OneHotEncoder":

            st.info(
                "Encodes categorical features with fixed training categories."
            )

            categories = step["data"]["categories"]

            for i, cat_list in enumerate(categories):
                st.write(f"Feature {step['columns'][i]} categories:", cat_list)

# =====================================================
# DRIFT MONITORING
# =====================================================

elif page == "Drift Monitoring":

    st.header("Drift Monitoring")

    if st.button("Run Drift Detection"):

        with st.spinner("Running drift detection..."):

            report = run_drift_check(df)

        metrics = report["metrics"]

        metrics_df = pd.DataFrame(
            metrics.items(), columns=["Metric", "Value"]
        )

        st.subheader("Metrics")
        st.dataframe(metrics_df, width="stretch")

        fig = px.bar(
            metrics_df,
            x="Metric",
            y="Value",
            title="Drift Metrics",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Alerts")

        if report["alerts"]:
            for alert in report["alerts"]:
                st.error(alert)
        else:
            st.success("No alerts triggered.")

# =====================================================
# EXPERIMENT LAB
# =====================================================

elif page == "Experiment Lab":

    st.header("Experiment Lab")

    experiment = st.selectbox(
        "Select Experiment",
        [
            "Baseline",
            "Numeric Scale Drift",
            "Unseen Category Drift",
            "Column Shuffle Drift",
        ],
    )

    if st.button("Run Experiment"):

        df_exp = df.copy()

        # -------------------------------
        # Numeric Scale Drift
        # -------------------------------

        if experiment == "Numeric Scale Drift":

            num_cols = df_exp.select_dtypes(include="number").columns
            df_exp[num_cols] = df_exp[num_cols] * 50

        # -------------------------------
        # Unseen Category Drift
        # -------------------------------

        elif experiment == "Unseen Category Drift":

            cat_cols = df_exp.select_dtypes(include="object").columns

            # introduce drift in ~30% of rows
            mask = df_exp.sample(frac=0.3).index

            for col in cat_cols:
                df_exp.loc[mask, col] = "UNKNOWN_NEW_CATEGORY"

        # -------------------------------
        # Column Shuffle Drift
        # -------------------------------

        elif experiment == "Column Shuffle Drift":

            df_exp = df_exp.sample(frac=1, axis=1)

        with st.spinner("Running experiment..."):

            report = run_drift_check(df_exp)

        metrics_df = pd.DataFrame(
            report["metrics"].items(),
            columns=["Metric", "Value"],
        )

        st.subheader("Metrics")
        st.dataframe(metrics_df, width="stretch")

        fig = px.bar(
            metrics_df,
            x="Metric",
            y="Value",
            title="Experiment Metrics",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Alerts")

        if report["alerts"]:
            for alert in report["alerts"]:
                st.error(alert)
        else:
            st.success("No alerts triggered.")