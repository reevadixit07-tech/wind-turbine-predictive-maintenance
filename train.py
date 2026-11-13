from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Save charts without popping up blocking windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------- SETTINGS ----------------
DATA_ROOT = Path("data/CARE_To_Compare")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

HORIZON_DAYS = 30
ROWS_PER_DAY = 144     # 10-min intervals -> 144 rows/day
HOURLY_STEP = 6        # Keep 1 row/hour to keep RAM low and training fast
ROLLING_HOURS = 24     # 24-hour smoothing window


def identify_component(description: str) -> str:
    text = str(description).lower()
    if "main bearing" in text or "rotorbearing" in text or "rotor bearing" in text:
        return "Main/Rotor Bearing"
    if "gearbox" in text:
        return "Gearbox"
    if "generator" in text:
        return "Generator"
    if "transformer" in text:
        return "Transformer"
    if "hydraulic" in text:
        return "Hydraulic"
    return "Other"


def load_and_prepare_farm(farm_name: str, use_event_end: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    farm_dir = DATA_ROOT / farm_name
    events = pd.read_csv(farm_dir / "event_info.csv", sep=";")
    events["component"] = events["event_description"].map(identify_component)
    event_by_id = events.set_index("event_id")

    data_files = sorted((farm_dir / "datasets").glob("*.csv"), key=lambda f: int(f.stem))
    header = pd.read_csv(data_files[0], sep=";", nrows=0).columns
    sensor_cols = [c for c in header if c.endswith("_avg")]
    meta_cols = ["time_stamp", "asset_id", "id", "train_test", "status_type_id"]

    prepared = []
    for file in data_files:
        eid = int(file.stem)
        ev = event_by_id.loc[eid]
        raw = pd.read_csv(file, sep=";", usecols=meta_cols + sensor_cols).sort_values("id")

        healthy = raw.loc[raw["train_test"] == "train", sensor_cols]
        if healthy.empty:
            healthy = raw[sensor_cols]
        med = healthy.median()
        std = healthy.std().replace(0, np.nan)

        hourly = raw.iloc[::HOURLY_STEP].reset_index(drop=True)
        z = (hourly[sensor_cols] - med) / std

        feats = pd.concat(
            [
                z.add_suffix("_z"),
                z.rolling(ROLLING_HOURS, min_periods=12).mean().add_suffix("_zmean"),
                (z - z.shift(ROLLING_HOURS)).add_suffix("_trend"),
            ],
            axis=1,
        ).astype("float32")

        # Labeling: 30-day warning window before failure
        if ev["event_label"] == "anomaly":
            failure_id = ev["event_end_id"] if use_event_end else ev["event_start_id"]
            warn_start = failure_id - HORIZON_DAYS * ROWS_PER_DAY
            if use_event_end:
                warn_start = max(warn_start, ev["event_start_id"])
            target = (hourly["id"] >= warn_start) & (hourly["id"] <= failure_id)
            # Exclude earlier active anomaly rows (>30d before failure) so they aren't mislabeled healthy
            drop_early_anomaly = (hourly["id"] >= ev["event_start_id"]) & (hourly["id"] < warn_start)
        else:
            failure_id = hourly["id"].max()
            target = pd.Series(False, index=hourly.index)
            drop_early_anomaly = pd.Series(False, index=hourly.index)

        feats["y"] = target.astype("int8").to_numpy()
        feats["event_id"] = eid
        feats["asset_id"] = hourly["asset_id"].to_numpy()
        feats["id"] = hourly["id"].to_numpy()
        feats["time_stamp"] = hourly["time_stamp"].to_numpy()
        feats["train_test"] = hourly["train_test"].to_numpy()
        feats["component"] = ev["component"]
        feats["event_label"] = ev["event_label"]
        feats["days_to_failure"] = (hourly["id"].to_numpy() - failure_id) / ROWS_PER_DAY

        operating = hourly["status_type_id"].isin([0, 1, 2]).to_numpy()
        warmup = np.arange(len(hourly)) < ROLLING_HOURS
        keep = operating & (~warmup) & (~drop_early_anomaly.to_numpy())

        cleaned = feats.loc[keep].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        prepared.append(cleaned)

    df = pd.concat(prepared, ignore_index=True)
    feat_cols = [c for c in df.columns if c.endswith(("_z", "_zmean", "_trend"))]
    return df, events, feat_cols


def evaluate_farm(df: pd.DataFrame, feat_cols: list[str], tag: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = df[feat_cols]
    y = df["y"]
    groups = df["event_id"]

    cv = StratifiedGroupKFold(n_splits=5)

    # 1. Logistic Regression (Baseline)
    log_pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    log_prob = cross_val_predict(log_pipe, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]

    # 2. Random Forest (30-Day Failure Predictor)
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_prob = cross_val_predict(rf, X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]

    # 3. Isolation Forest (Unsupervised Anomaly Detection on healthy rows only)
    iso_score = np.zeros(len(df), dtype="float32")
    for tr_idx, va_idx in cv.split(X, y, groups):
        healthy_tr = tr_idx[(y.iloc[tr_idx] == 0).to_numpy()]
        iso = IsolationForest(n_estimators=150, contamination=0.02, random_state=42, n_jobs=-1)
        iso.fit(X.iloc[healthy_tr])
        raw_tr = -iso.score_samples(X.iloc[healthy_tr])
        raw_va = -iso.score_samples(X.iloc[va_idx])
        lo, hi = np.percentile(raw_tr, [5, 99])
        iso_score[va_idx] = np.clip((raw_va - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    df = df.copy()
    df["rf_prob"] = rf_prob
    df["log_prob"] = log_prob
    df["iso_score"] = iso_score

    # Overall metrics + False Alarm Rate at 99th percentile of healthy hours
    rows = []
    for name, scores in [
        ("Random Baseline", np.full(len(y), y.mean())),
        ("Isolation Forest (Unsupervised)", iso_score),
        ("Logistic Regression", log_prob),
        ("Random Forest (30-Day)", rf_prob),
    ]:
        if name == "Random Baseline":
            pr, roc, fpr_1, recall_1 = y.mean(), 0.5, 0.01, 0.01
        else:
            pr = average_precision_score(y, scores)
            roc = roc_auc_score(y, scores)
            thresh = np.quantile(scores[y == 0], 0.99)
            fpr_1 = float((scores[y == 0] >= thresh).mean())
            recall_1 = float((scores[y == 1] >= thresh).mean())
        rows.append({
            "Farm": tag,
            "Model": name,
            "PR-AUC": round(pr, 3),
            "ROC-AUC": round(roc, 3),
            "Recall @ 1% False Alarm": round(recall_1, 3),
            "False Alarm Rate": round(fpr_1, 3),
        })
    metrics_df = pd.DataFrame(rows)

    # Component breakdown using Random Forest
    comp_rows = []
    for comp in sorted(df.loc[df["y"] == 1, "component"].unique()):
        comp_y = ((df["y"] == 1) & (df["component"] == comp)).astype(int)
        n_events = df.loc[comp_y == 1, "event_id"].nunique()
        comp_rows.append({
            "Farm": tag,
            "Component": comp,
            "Failure Events": n_events,
            "ROC-AUC (RF)": round(roc_auc_score(comp_y, rf_prob), 3),
            "PR-AUC (RF)": round(average_precision_score(comp_y, rf_prob), 3),
            "ROC-AUC (IsoForest)": round(roc_auc_score(comp_y, iso_score), 3),
        })
    comp_df = pd.DataFrame(comp_rows)

    return df, metrics_df, comp_df


def save_plots(df_b: pd.DataFrame, df_a: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Precision-Recall Curve (Wind Farm B - Main Bearing Focus)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, label in [
        ("rf_prob", "Random Forest (30-Day)"),
        ("log_prob", "Logistic Regression"),
        ("iso_score", "Isolation Forest"),
    ]:
        p, r, _ = precision_recall_curve(df_b["y"], df_b[col])
        ap = average_precision_score(df_b["y"], df_b[col])
        ax.plot(r, p, lw=2, label=f"{label} (PR-AUC={ap:.3f})")
    ax.axhline(df_b["y"].mean(), color="gray", ls="--", label="Baseline")
    ax.set(xlabel="Recall", ylabel="Precision", title="Wind Farm B: 30-Day Failure Prediction PR Curve", ylim=(0, 1))
    ax.legend()
    plt.savefig(OUT_DIR / "01_pr_curve_farm_b.png", bbox_inches="tight", dpi=130)
    plt.close()

    # 2. Rotor Bearing Thermal Degradation before Main Bearing Failure (Farm B)
    bearing_ev = df_b[(df_b["y"] == 1) & (df_b["component"] == "Main/Rotor Bearing")]["event_id"].unique()
    sub = df_b[df_b["event_id"].isin(bearing_ev) & (df_b["days_to_failure"] >= -45) & (df_b["days_to_failure"] <= 0)].copy()
    sub["day"] = sub["days_to_failure"].round().astype(int)
    sub["rotor_temp_z"] = sub[["sensor_51_avg_zmean", "sensor_52_avg_zmean"]].mean(axis=1)
    curve = sub.groupby("day")["rotor_temp_z"].mean()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(curve.index, curve.values, lw=2.5, color="#d62728")
    ax.axvspan(-30, 0, color="red", alpha=0.1, label="30-Day Failure Window")
    ax.axhline(0, color="gray", lw=1)
    ax.set(
        xlabel="Days Before Main Bearing Failure / Standstill",
        ylabel="Rotor Bearing Temp (24h z-score)",
        title="Wind Farm B: Rotor Bearing Temperature Rises Before Failure",
    )
    ax.legend()
    plt.savefig(OUT_DIR / "02_main_bearing_degradation.png", bbox_inches="tight", dpi=130)
    plt.close()

    # 3. Early Warning Example (Worst Main Bearing Failure in Farm B)
    ex_id = int(bearing_ev[0])
    for eid in bearing_ev:
        if eid == 77:
            ex_id = 77
    ex = df_b[(df_b["event_id"] == ex_id) & (df_b["days_to_failure"] >= -45) & (df_b["days_to_failure"] <= 0)].sort_values("id").copy()
    ex["smoothed_rf"] = ex["rf_prob"].rolling(24, min_periods=6).mean()
    ex["smoothed_iso"] = ex["iso_score"].rolling(24, min_periods=6).mean()
    thresh = np.quantile(df_b.loc[df_b["y"] == 0, "rf_prob"], 0.99)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ex["days_to_failure"], ex["smoothed_rf"], lw=2.2, label="Random Forest 30-Day Risk (24h mean)")
    ax.plot(ex["days_to_failure"], ex["smoothed_iso"], lw=1.8, ls=":", label="Isolation Forest Anomaly Score")
    ax.axhline(thresh, color="orange", ls="--", label="1% False-Alarm Alert Threshold")
    ax.axvspan(-30, 0, color="red", alpha=0.1, label="30-Day Warning Horizon")
    ax.set(xlabel="Days Before Failure", ylabel="Risk / Anomaly Score", title=f"Early Warning Timeline (Main Bearing Event {ex_id})", ylim=(0, 1))
    ax.legend()
    plt.savefig(OUT_DIR / "03_early_warning_timeline.png", bbox_inches="tight", dpi=130)
    plt.close()


def build_fleet_250(df_b: pd.DataFrame) -> pd.DataFrame:
    """Create a 250-turbine offshore fleet monitoring snapshot grounded in real Farm B predictions."""
    rng = np.random.default_rng(42)
    latest_per_event = []
    for eid, grp in df_b.groupby("event_id"):
        grp = grp.sort_values("id")
        if grp["event_label"].iloc[0] == "anomaly":
            # Take snapshots from the 30-day pre-failure window
            sub = grp[grp["y"] == 1]
            sample_row = sub.iloc[-min(len(sub), 24)] if not sub.empty else grp.iloc[-1]
        else:
            sample_row = grp.iloc[-1]
        latest_per_event.append(sample_row)
    pool = pd.DataFrame(latest_per_event)

    normal_pool = pool[pool["event_label"] == "normal"]
    bearing_pool = pool[pool["component"] == "Main/Rotor Bearing"]
    other_anom_pool = pool[(pool["event_label"] == "anomaly") & (pool["component"] != "Main/Rotor Bearing")]

    fleet_rows = []
    for i in range(1, 251):
        tid = f"WT-{i:03d}"
        if i <= 8:
            src = bearing_pool.iloc[i % len(bearing_pool)]
            tier = "High Risk (Bearing)"
        elif i <= 22:
            src = other_anom_pool.iloc[i % len(other_anom_pool)] if not other_anom_pool.empty else bearing_pool.iloc[0]
            tier = "Warning"
        else:
            src = normal_pool.iloc[i % len(normal_pool)]
            tier = "Normal"

        jitter = float(rng.normal(0, 0.025))
        rf_p = float(np.clip(src["rf_prob"] + jitter, 0.01, 0.99))
        iso_p = float(np.clip(src["iso_score"] + jitter, 0.01, 0.99))
        rotor_z = float(np.mean([src.get("sensor_51_avg_zmean", 0.0), src.get("sensor_52_avg_zmean", 0.0)]) + rng.normal(0, 0.12))

        if rf_p >= 0.55 or (iso_p >= 0.70 and rotor_z > 0.5):
            status = "High Risk"
        elif rf_p >= 0.30 or iso_p >= 0.45:
            status = "Warning"
        else:
            status = "Normal"

        fleet_rows.append({
            "Turbine ID": tid,
            "Source CARE Event": int(src["event_id"]),
            "Component Focus": src["component"] if src["event_label"] == "anomaly" else "Healthy",
            "Rotor Bearing Temp (z-score)": round(rotor_z, 2),
            "Isolation Forest Anomaly": round(iso_p * 100, 1),
            "30-Day Failure Risk (%)": round(rf_p * 100, 1),
            "Status": status,
        })
    return pd.DataFrame(fleet_rows)


def main():
    print("1/4 Loading Wind Farm B (Main/Rotor Bearing Focus)...")
    df_b, _, feat_b = load_and_prepare_farm("Wind Farm B", use_event_end=True)
    print(f"    Farm B rows: {len(df_b):,} | Positive (30d): {int(df_b['y'].sum()):,} ({df_b['y'].mean():.2%})")

    print("2/4 Training & evaluating models on Wind Farm B...")
    df_b, metrics_b, comp_b = evaluate_farm(df_b, feat_b, "Wind Farm B (Main Bearing)")

    print("3/4 Loading & evaluating Wind Farm A (Multi-Component 30d Benchmark)...")
    df_a, _, feat_a = load_and_prepare_farm("Wind Farm A", use_event_end=False)
    print(f"    Farm A rows: {len(df_a):,} | Positive (30d): {int(df_a['y'].sum()):,} ({df_a['y'].mean():.2%})")
    df_a, metrics_a, comp_a = evaluate_farm(df_a, feat_a, "Wind Farm A (Multi-Component)")

    metrics_all = pd.concat([metrics_b, metrics_a], ignore_index=True)
    comp_all = pd.concat([comp_b, comp_a], ignore_index=True)

    print("\n=== MODEL COMPARISON (5-Fold Stratified Group CV) ===")
    print(metrics_all.to_string(index=False))
    print("\n=== PERFORMANCE BY COMPONENT ===")
    print(comp_all.to_string(index=False))

    print("\n4/4 Saving outputs, charts, and 250-turbine fleet snapshot...")
    metrics_all.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    comp_all.to_csv(OUT_DIR / "component_metrics.csv", index=False)

    # Save compact dashboard datasets (only metadata + predictions + key bearing sensors)
    keep_b = [
        "event_id", "asset_id", "id", "time_stamp", "component", "event_label",
        "days_to_failure", "y", "rf_prob", "log_prob", "iso_score",
        "sensor_51_avg_zmean", "sensor_52_avg_zmean", "sensor_34_avg_zmean",
    ]
    df_b[[c for c in keep_b if c in df_b.columns]].to_csv(OUT_DIR / "farm_b_predictions.csv", index=False)

    keep_a = [
        "event_id", "asset_id", "id", "time_stamp", "component", "event_label",
        "days_to_failure", "y", "rf_prob", "log_prob", "iso_score",
        "sensor_11_avg_zmean", "sensor_13_avg_zmean", "sensor_14_avg_zmean",
    ]
    df_a[[c for c in keep_a if c in df_a.columns]].to_csv(OUT_DIR / "farm_a_predictions.csv", index=False)

    fleet_df = build_fleet_250(df_b)
    fleet_df.to_csv(OUT_DIR / "fleet_250.csv", index=False)

    # Fit and save final Wind Farm B Random Forest model
    final_rf = RandomForestClassifier(
        n_estimators=150, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(df_b[feat_b], df_b["y"])
    joblib.dump(final_rf, OUT_DIR / "rf_farm_b_30d.joblib")

    save_plots(df_b, df_a)
    print(f"\nDone! Everything saved to '{OUT_DIR}/'.")


if __name__ == "__main__":
    main()