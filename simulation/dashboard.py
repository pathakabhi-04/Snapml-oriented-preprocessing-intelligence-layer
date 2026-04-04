import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.run_live_pipeline import run_pipeline

st.set_page_config(layout="wide")
st.title("🚀 ML Observability Live Demo")

# -----------------------------
# Session State
# -----------------------------
if "results" not in st.session_state:
    st.session_state.results = []

if "running" not in st.session_state:
    st.session_state.running = False

if "generator" not in st.session_state:
    st.session_state.generator = None

if "summary" not in st.session_state:  # ✅ FIX 1: initialize summary
    st.session_state.summary = None

# -----------------------------
# Controls
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start Simulation"):
        st.session_state.results = []
        st.session_state.summary = None  # ✅ reset summary on restart
        st.session_state.generator = run_pipeline()
        st.session_state.running = True

with col2:
    if st.button("⏹ Stop"):
        st.session_state.running = False
        st.session_state.generator = None

# -----------------------------
# Run ONE STEP
# -----------------------------
if st.session_state.running and st.session_state.generator:

    try:
        result = next(st.session_state.generator)

        # ✅ HANDLE SUMMARY SEPARATELY
        if isinstance(result, dict) and result.get("type") == "summary":
            st.session_state.summary = result
        else:
            st.session_state.results.append(result)

    except StopIteration:
        st.session_state.running = False

# -----------------------------
# UI Rendering
# -----------------------------
if st.session_state.results:

    df = pd.DataFrame(st.session_state.results)
    latest = st.session_state.results[-1]

    # -----------------------------
    # Latest Batch
    # -----------------------------
    st.subheader("📡 Latest Batch")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🚨 Drift")
        st.write(f"Scenario: {latest['scenario']}")
        st.write(f"Detected: {latest['drift_detected']}")

    with col2:
        st.markdown("### 📉 Metrics")
        if latest["accuracy"] is not None:
            st.write(f"Accuracy: {latest['accuracy']:.4f}")
        else:
            st.warning("Waiting for labels...")

    # -----------------------------
    # Logs
    # -----------------------------
    st.subheader("🧾 Logs")
    for log in latest["logs"][-10:]:
        st.text(log)

    # -----------------------------
    # Live Metrics
    # -----------------------------
    TP = sum(1 for r in st.session_state.results if r["drift_detected"] and r["scenario"] != "none")
    FP = sum(1 for r in st.session_state.results if r["drift_detected"] and r["scenario"] == "none")
    FN = sum(1 for r in st.session_state.results if not r["drift_detected"] and r["scenario"] != "none")

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0

    # 🔥 NEW: Drift Rate
    drift_rate = sum(r["drift_detected"] for r in st.session_state.results) / len(st.session_state.results)

    st.subheader("📊 Live Metrics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision:.2f}")
    c2.metric("Recall", f"{recall:.2f}")
    c3.metric("Batches", len(st.session_state.results))
    c4.metric("Drift Rate", f"{drift_rate:.2%}")

    # -----------------------------
    # 🚨 System Alerts (REAL-TIME FIXED)
    # -----------------------------
    st.markdown("## 🚨 System Alerts")

    issues = []

    # Use LIVE metrics (not summary)
    current_accuracy = latest["accuracy"]
    drift_detected = latest["drift_detected"]

    # Compute rolling drift rate (last 10 batches)
    recent = st.session_state.results[-10:]
    drift_rate_recent = sum(r["drift_detected"] for r in recent) / len(recent)

    # Conditions
    if current_accuracy is not None and current_accuracy < 0.75:
        issues.append(("error", "Model performance degrading"))

    if drift_rate_recent > 0.5:
        issues.append(("warning", "Frequent drift detected"))

    if drift_detected:
        issues.append(("info", "Drift detected in latest batch"))

    # Display
    if not issues:
        st.success("✅ System operating normally")
    else:
        for level, msg in issues:
            if level == "error":
                st.error(f"🚨 {msg}")
            elif level == "warning":
                st.warning(f"⚠️ {msg}")
            else:
                st.info(f"ℹ️ {msg}")

    st.markdown("---")

    # -----------------------------
    # Charts
    # -----------------------------
    st.subheader("📈 Trends")

    chart_df = df.copy()

    # -----------------------------
    # ⏱ Convert timestamps (REQUIRED)
    # -----------------------------
    chart_df["drift_time"] = pd.to_datetime(chart_df["drift_time"], errors="coerce")
    chart_df["metrics_time"] = pd.to_datetime(chart_df["metrics_time"], errors="coerce")

    # Set index if available
    if "batch_id" in chart_df.columns:
        chart_df = chart_df.set_index("batch_id")

    # -----------------------------
    # PREPROCESS (ONLY ONCE ✅)
    # -----------------------------
    chart_df["drift_flag"] = chart_df["drift_detected"].astype(int)
    chart_df["accuracy"] = chart_df["accuracy"].astype(float)

    # -----------------------------
    # ✅ ADD THIS HERE (NEW LOGIC)
    # -----------------------------
    drift_window = 3
    drift_indices = chart_df[chart_df["drift_flag"] == 1].index.tolist()

    first_drift = None

    for i in range(len(drift_indices) - drift_window + 1):
        window_slice = drift_indices[i:i+drift_window]

        # ✅ STRICT: consecutive drift (strong signal)
        if all(window_slice[j+1] - window_slice[j] == 1 for j in range(drift_window - 1)):
            first_drift = window_slice[0]
            break

    # -----------------------------
    # Plotly Chart (FINAL PRO VERSION)
    # -----------------------------
    fig = go.Figure()

    # Accuracy line
    fig.add_trace(go.Scatter(
        x=chart_df.index,
        y=chart_df["accuracy"],
        mode='lines',
        name='Accuracy',
        line=dict(width=3),
        hovertemplate="Batch: %{x}<br>Accuracy: %{y:.3f}<extra></extra>"
    ))

    # -----------------------------
    # ✅ Drift markers (DECOUPLED FROM ACCURACY)
    # -----------------------------
    max_acc = chart_df["accuracy"].max() if chart_df["accuracy"].notnull().any() else 1.0

    drift_points = chart_df[chart_df["drift_flag"] == 1]

    # Place drift markers at constant top band
    fig.add_trace(go.Scatter(
        x=drift_points.index,
        y=[max_acc * 1.02] * len(drift_points),
        mode='markers',
        name='Drift Detected',
        marker=dict(size=6, color='red'),
        text=drift_points["scenario"],
        hovertemplate="Batch: %{x}<br>Drift: %{text}<extra></extra>"
    ))


    # 🔥 Drift zones (MERGED — PRO VERSION)
    merged_zones = []
    current_start = None
    last_drift_idx = None
    gap_threshold = 4  # 🔥 increase for smoother regions

    for i, row in chart_df.iterrows():

        if row["drift_flag"] == 1:
            if current_start is None:
                current_start = i

            last_drift_idx = i

        else:
            if current_start is not None:
                if last_drift_idx is not None and (i - last_drift_idx) <= gap_threshold:
                    continue
                else:
                    merged_zones.append((current_start, last_drift_idx))
                    current_start = None
                    last_drift_idx = None

    # Handle last zone
    if current_start is not None:
        merged_zones.append((current_start, last_drift_idx))

    # Plot zones
    for start, end in merged_zones:
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="red",
            opacity=0.03,
            line_width=0
        )
    # -----------------------------
    # Threshold line
    # -----------------------------
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="white",
        line_width=2,
        annotation_text="Target Accuracy (SLA)",
        annotation_position="top left"
    )

    # -----------------------------
    # 🔥 Degradation Start Marker
    # -----------------------------
    threshold = 0.8
    degradation_point = None

    window = 3  # 👈 require 3 consecutive drops

    values = chart_df["accuracy"].values
    indices = chart_df.index

    for i in range(len(values) - window + 1):

        # Skip if any value is NaN
        if any(pd.isna(v) for v in values[i:i+window]):
            continue

        # Check sustained degradation
        if all(v < threshold for v in values[i:i+window]):
            degradation_point = indices[i]
            break

    if degradation_point is not None:
        fig.add_vline(
            x=degradation_point,
            line_width=3,
            line_dash="dash",
            line_color="yellow"
        )

        fig.add_annotation(
            x=degradation_point,
            y=threshold,
            text="Accuracy drops below SLA",
            showarrow=True,
            arrowhead=2,
            font=dict(color="yellow")
        )

    # -----------------------------
    # 🔥 Highlight ONLY degraded region
    # -----------------------------
    if degradation_point is not None:

        fig.add_vrect(
            x0=degradation_point,
            x1=chart_df.index[-1],
            fillcolor="red",
            opacity=0.08,
            line_width=0
        )


    # -----------------------------
    # ⏱ Lead Time (NUMERIC)
    # -----------------------------
    lead_time = None

    if first_drift is not None and degradation_point is not None:

        drift_time = chart_df.loc[first_drift]["drift_time"]
        degrade_time = chart_df.loc[degradation_point]["metrics_time"]

        if pd.notnull(drift_time) and pd.notnull(degrade_time):
            lead_time = (degrade_time - drift_time).total_seconds()


    # Draw lead time region
    if first_drift is not None and degradation_point is not None and first_drift < degradation_point:

        fig.add_vrect(
            x0=first_drift,
            x1=degradation_point,
            fillcolor="yellow",
            opacity=0.08,
            line_width=0
        )

        fig.add_annotation(
            x=(first_drift + degradation_point) / 2,
            y=0.82,  # slightly above threshold line (0.8)
            text="Detection Lead Time",
            showarrow=False,
            font=dict(color="yellow")
        )


    # -----------------------------
    # Layout polish
    # -----------------------------
    fig.update_layout(
        xaxis_title="Batch",
        yaxis_title="Accuracy",
        height=420,
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # 🚀 Key Insight (CORE STORY)
    # -----------------------------
    if (
        first_drift is not None 
        and degradation_point is not None 
        and first_drift < degradation_point
    ):

        lead_batches = degradation_point - first_drift

        # Use system-level avg lead time (more meaningful than per-event ~0 sec)
        avg_lt = None
        if st.session_state.summary:
            avg_lt = st.session_state.summary["lead_time"]["avg"]

        if avg_lt is not None and avg_lt > 0:
            st.success(
                f"🚀 Early Warning Active → Drift detected {lead_batches} batches earlier "
                f"(~{avg_lt:.1f} sec lead time before performance drop)"
            )
        else:
            st.success(
                f"🚀 Early Warning Active → Drift detected {lead_batches} batches before performance drop"
            )


    # -----------------------------
    # 🔍 Insight (NEW)
    # -----------------------------
    if st.session_state.summary:

        drift_rate = sum(r["drift_detected"] for r in st.session_state.results) / len(st.session_state.results)
        accuracy = st.session_state.summary["drift_detection"]["accuracy"]

        if drift_rate > 0.6 and accuracy < 0.75:
            st.warning("⚠️ High drift + falling accuracy → Model likely degrading")

    # -----------------------------
    # Final Summary
    # -----------------------------
    if st.session_state.summary:

        summary = st.session_state.summary

        st.subheader("📊 Final Evaluation")

        # -----------------------------
        # 📊 Confusion Matrix (GRID FORMAT)
        # -----------------------------
        st.markdown("### 📊 Confusion Matrix")

        TP = summary["drift_detection"]["TP"]
        FP = summary["drift_detection"]["FP"]
        FN = summary["drift_detection"]["FN"]
        TN = summary["drift_detection"]["TN"]

        # Header
        h1, h2, h3 = st.columns([2, 2, 2])
        h2.markdown("**Predicted Drift**")
        h3.markdown("**Predicted No Drift**")

        # Row 1
        r1c1, r1c2, r1c3 = st.columns([2, 2, 2])
        r1c1.markdown("**Actual Drift**")
        r1c2.metric("TP", TP)
        r1c3.metric("FN", FN)

        # Row 2
        r2c1, r2c2, r2c3 = st.columns([2, 2, 2])
        r2c1.markdown("**Actual No Drift**")
        r2c2.metric("FP", FP)
        r2c3.metric("TN", TN)

        # -----------------------------
        # Performance Metrics
        # -----------------------------
        st.markdown("### 📈 Performance")

        st.caption("Recall → how many drift events are correctly detected")
        st.caption("Precision → how often drift alerts are correct (low false alarms)")

        p1, p2, p3, p4 = st.columns(4)

        p1.metric("Precision", f"{summary['drift_detection']['precision']:.2f}")
        p2.metric("Recall", f"{summary['drift_detection']['recall']:.2f}")
        p3.metric("Accuracy", f"{summary['drift_detection']['accuracy']:.2f}")
        p4.metric("Coverage", f"{summary['coverage']['rate']:.2f}")

        # -----------------------------
        # System Metrics
        # -----------------------------
        st.markdown("### ⚙️ System")

        s1, s2 = st.columns(2)

        avg_lt = summary["lead_time"]["avg"]

        s1.metric(
            "Avg Lead Time",
            f"{avg_lt:.2f} sec" if avg_lt is not None else "N/A"
        )

        available = summary['metrics_availability']['available']
        total = summary['metrics_availability']['total']

        percentage = (available / total) if total > 0 else 0

        s2.metric(
            "Metrics Ready",
            f"{available}/{total} ({percentage:.0%})"
        )

    # -----------------------------
    # 🧪 Scenario Breakdown
    # -----------------------------
    if st.session_state.summary:

        st.markdown("---")
        st.subheader("🧪 Scenario Breakdown")

        scenario_df = pd.DataFrame.from_dict(
            st.session_state.summary["scenario_stats"],
            orient="index"
        )

        # Add detection rate safely
        scenario_df["rate"] = scenario_df.apply(
            lambda row: row["detected"] / row["total"] if row["total"] > 0 else 0,
            axis=1
        )

        # 🔥 PRO: sort by performance
        scenario_df = scenario_df.sort_values("rate", ascending=False)

        # Optional: nicer formatting
        scenario_df["rate"] = scenario_df["rate"].round(2)

        st.dataframe(scenario_df, use_container_width=True)
    # -----------------------------
    # Table
    # -----------------------------
    st.subheader("📄 Recent Results")
    st.dataframe(df.tail(10), use_container_width=True)

# -----------------------------
# Auto-refresh
# -----------------------------
if st.session_state.running:
    st.rerun()