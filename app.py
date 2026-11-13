from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Offshore Wind Turbine Predictive Maintenance",
    page_icon="🌬️",
    layout="wide",
)

OUT_DIR = Path("outputs")
if not (OUT_DIR / "metrics_summary.csv").exists():
    st.error("Missing 'outputs/' folder. Please run `python train.py` first.")
    st.stop()


@st.cache_data
def load_data():
    fleet = pd.read_csv(OUT_DIR / "fleet_250.csv")
    metrics = pd.read_csv(OUT_DIR / "metrics_summary.csv")
    comp = pd.read_csv(OUT_DIR / "component_metrics.csv")
    farm_b = pd.read_csv(OUT_DIR / "farm_b_predictions.csv")
    farm_a = pd.read_csv(OUT_DIR / "farm_a_predictions.csv")
    return fleet, metrics, comp, farm_b, farm_a


fleet_df, metrics_df, comp_df, farm_b_df, farm_a_df = load_data()


def recommend_action(status: str) -> str:
    if status == "High Risk":
        return "Schedule Bearing Replacement / Crane Dispatch"
    if status == "Warning":
        return "Inspect Bearing & Oil Temperature Trend"
    return "Continue Normal Monitoring"


fleet_df["Recommended Action"] = fleet_df["Status"].map(recommend_action)

# ---------------- HEADER & KPIs ----------------
st.title("Offshore Wind Turbine Predictive Maintenance")
st.caption("30-Day Main Bearing Failure Forecasting & SCADA Anomaly Detection (CARE to Compare Dataset)")

n_total = len(fleet_df)
n_high = int((fleet_df["Status"] == "High Risk").sum())
n_warn = int((fleet_df["Status"] == "Warning").sum())
bearing_auc = comp_df.loc[comp_df["Component"] == "Main/Rotor Bearing", "ROC-AUC (RF)"].iloc[0]
gearbox_auc = comp_df.loc[comp_df["Component"] == "Gearbox", "ROC-AUC (RF)"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Offshore Fleet Monitored", f"{n_total} Turbines")
c2.metric("High-Risk Turbines (30d)", f"{n_high}", delta="Action Required", delta_color="inverse")
c3.metric("Warning / Watchlist", f"{n_warn}")
c4.metric("Main Bearing ROC-AUC", f"{bearing_auc:.3f}", delta=f"Farm A Gearbox: {gearbox_auc:.3f}")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Fleet Overview (250 Turbines)",
    "2. Main Bearing Early Warning",
    "3. Model & False-Alarm Analysis",
    "4. Project Summary",
])

# === TAB 1: FLEET ===
with tab1:
    st.subheader("250-Turbine Offshore Fleet Health Monitor")
    st.info(
        "Note: All ML metrics are evaluated strictly on real held-out CARE to Compare events. "
        "The 250-turbine fleet view scales validated Wind Farm B turbine risk states across 250 offshore IDs for fleet-level prioritization."
    )

    f_col1, f_col2 = st.columns([1, 2])
    status_filter = f_col1.multiselect(
        "Filter by Status",
        ["High Risk", "Warning", "Normal"],
        default=["High Risk", "Warning", "Normal"],
    )
    search_id = f_col2.text_input("Search Turbine ID (e.g., WT-001)", "").strip().upper()

    filtered = fleet_df[fleet_df["Status"].isin(status_filter)]
    if search_id:
        filtered = filtered[filtered["Turbine ID"].str.contains(search_id)]

    left, right = st.columns([1, 2])
    with left:
        counts = fleet_df["Status"].value_counts().reindex(["High Risk", "Warning", "Normal"], fill_value=0)
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.bar(counts.index, counts.values, color=["#d62728", "#ff7f0e", "#2ca02c"])
        ax.set_ylabel("Number of Turbines")
        ax.set_title("Fleet Risk Distribution")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.write(f"Showing **{len(filtered)}** turbines (sorted by highest 30-day risk):")
        st.dataframe(
            filtered.sort_values("30-Day Failure Risk (%)", ascending=False),
            hide_index=True,
        )

# === TAB 2: EARLY WARNING ===
with tab2:
    st.subheader("Real Wind Farm B Events: 30-Day Early Warning Timeline")
    event_meta = (
        farm_b_df[["event_id", "asset_id", "component", "event_label"]]
        .drop_duplicates()
        .sort_values(["event_label", "component", "event_id"])
    )
    event_labels = {
        int(r["event_id"]): (
            f"Event {int(r['event_id'])} (Turbine {int(r['asset_id'])}) — "
            f"{r['component'] if r['event_label']=='anomaly' else 'Normal Healthy Operation'}"
        )
        for _, r in event_meta.iterrows()
    }

    default_idx = list(event_labels.keys()).index(77) if 77 in event_labels else 0
    selected_eid = st.selectbox(
        "Select a real Wind Farm B turbine event:",
        options=list(event_labels.keys()),
        index=default_idx,
        format_func=lambda k: event_labels[k],
    )

    ev_df = farm_b_df[farm_b_df["event_id"] == selected_eid].sort_values("id").copy()
    ev_df = ev_df[ev_df["days_to_failure"] >= -45].copy()
    ev_df["rf_smooth"] = ev_df["rf_prob"].rolling(24, min_periods=6).mean()
    ev_df["iso_smooth"] = ev_df["iso_score"].rolling(24, min_periods=6).mean()

    thresh_1pct = float(np.quantile(farm_b_df.loc[farm_b_df["y"] == 0, "rf_prob"], 0.99))
    alerts = ev_df[ev_df["rf_smooth"] >= thresh_1pct]

    m1, m2, m3 = st.columns(3)
    m1.metric("Peak 30-Day Risk (24h avg)", f"{ev_df['rf_smooth'].max():.1%}")
    m2.metric("Peak Anomaly Score", f"{ev_df['iso_smooth'].max():.1%}")
    if not alerts.empty and ev_df["event_label"].iloc[0] == "anomaly":
        lead_days = -float(alerts["days_to_failure"].min())
        m3.metric("First Alert Lead Time", f"{lead_days:.1f} days before failure")
    elif ev_df["event_label"].iloc[0] == "normal":
        m3.metric("False Alarms Triggered", f"{len(alerts)} hours")
    else:
        m3.metric("First Alert Lead Time", "No alert above threshold")

    col_a, col_b = st.columns(2)
    with col_a:
        fig1, ax1 = plt.subplots(figsize=(6.5, 4))
        ax1.plot(ev_df["days_to_failure"], ev_df["rf_smooth"], lw=2.2, label="Random Forest 30d Risk")
        ax1.plot(ev_df["days_to_failure"], ev_df["iso_smooth"], lw=1.8, ls=":", label="Isolation Forest Anomaly")
        ax1.axhline(thresh_1pct, color="orange", ls="--", label=f"1% False-Alarm Threshold ({thresh_1pct:.2f})")
        if ev_df["event_label"].iloc[0] == "anomaly":
            ax1.axvspan(-30, 0, color="red", alpha=0.1, label="30-Day Failure Window")
        ax1.set(xlabel="Days Relative to End/Failure", ylabel="Score (0–1)", title="Model Risk & Anomaly Scores", ylim=(0, 1))
        ax1.legend(fontsize=8)
        st.pyplot(fig1)
        plt.close(fig1)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(6.5, 4))
        ax2.plot(ev_df["days_to_failure"], ev_df["sensor_51_avg_zmean"], lw=2, label="Rotor Bearing Temp 1 (z-score)")
        ax2.plot(ev_df["days_to_failure"], ev_df["sensor_52_avg_zmean"], lw=2, label="Rotor Bearing Temp 2 (z-score)")
        ax2.axhline(0, color="gray", lw=1)
        if ev_df["event_label"].iloc[0] == "anomaly":
            ax2.axvspan(-30, 0, color="red", alpha=0.1, label="30-Day Failure Window")
        ax2.set(xlabel="Days Relative to End/Failure", ylabel="Deviation from Normal (z-score)", title="Rotor Bearing Temperature Signal")
        ax2.legend(fontsize=8)
        st.pyplot(fig2)
        plt.close(fig2)

# === TAB 3: MODEL & FALSE-ALARM ANALYSIS ===
with tab3:
    st.subheader("Model Comparison (5-Fold Stratified Group Cross-Validation)")
    st.dataframe(metrics_df, hide_index=True)

    st.subheader("Detection Accuracy by Component")
    st.dataframe(comp_df, hide_index=True)

    st.subheader("False-Alarm vs. Early-Detection Tradeoff (Wind Farm B)")
    user_thresh = st.slider("Alert Probability Threshold", min_value=0.10, max_value=0.90, value=0.40, step=0.05)
    y_true = farm_b_df["y"].to_numpy()
    preds = (farm_b_df["rf_prob"].to_numpy() >= user_thresh).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    t1, t2, t3 = st.columns(3)
    t1.metric("Precision (True Alerts)", f"{tp / max(tp + fp, 1):.1%}")
    t2.metric("Recall (Pre-Failure Hours Caught)", f"{tp / max(tp + fn, 1):.1%}")
    t3.metric("False Alarm Rate (Normal Hours)", f"{fp / max(fp + tn, 1):.2%}")

    p1, p2 = st.columns(2)
    with p1:
        st.image(str(OUT_DIR / "01_pr_curve_farm_b.png"))
    with p2:
        st.image(str(OUT_DIR / "02_main_bearing_degradation.png"))

# === TAB 4: SUMMARY ===
with tab4:
    st.subheader("Framework Overview & Key Takeaways")
    st.markdown(
        f"""
        **1. Problem Solved:** Traditional fixed SCADA thresholds miss gradual main-bearing degradation under changing wind and load conditions.
        By converting every sensor into **turbine-specific z-scores** (deviation from that turbine's healthy baseline), **24-hour rolling means**, and **24-hour trends**, our models normalize for operating and environmental differences.

        **2. Two-Stage ML Architecture:**
        - **Isolation Forest (Unsupervised):** Trained exclusively on healthy baseline operation (`y = 0`) to detect unusual multivariate sensor patterns without needing failure labels.
        - **Random Forest Classifier (Supervised 30-Day Horizon):** Estimates failure probability within the next **30 days**, validated using **5-fold Stratified Group Cross-Validation** (`event_id` grouping) so no failure event leaks between train and test folds.

        **3. Validated Findings & Honest Limitations:**
        - On **Wind Farm B**, Random Forest achieves **ROC-AUC = {bearing_auc:.3f}** on confirmed Main/Rotor Bearing failures (3 events), whereas Transformer overheating events score lower (`0.462`), pulling down the combined Farm B average.
        - On **Wind Farm A** (30-day horizon), Random Forest reaches **ROC-AUC = {gearbox_auc:.3f}** on Gearbox failures, confirming that mechanical bearing/gearbox faults produce strong thermal precursors 30 days ahead, while sudden electrical faults (e.g., generator) are harder to anticipate early.
        """
    )