#!/usr/bin/env python3
"""Chronic disease encounter sequence analysis pipeline.

This script implements four stages:
1. Sequence construction
2. Feature engineering
3. Clustering
4. Cluster profiling and reporting

It is designed for the integrated encounter master table produced earlier in
this workspace. The default input is:
    outputs/chronic_visit_master_2022_2025.csv

Key implementation notes:
- Uses pandas/sklearn/matplotlib/seaborn as requested.
- Reads only the columns needed for analysis to reduce memory pressure.
- Detects duplicate encounter rows and reports them before deduplicating.
- Copies itself into the output directory for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
CACHE_DIR = REPO_ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
KEYWORDS = [
    "endocrin",
    "cardiol",
    "internal med",
    "nephrol",
    "diabet",
    "cardiovasc",
    "heart",
    "hypertens",
]
HIGH_RISK_DEPARTMENTS = {"ED", "ICU", "OR", "LND"}
DATE_REFERENCE_YEAR = 2025
DPI = 150

EVENT_COLS = [
    "PatientDurableKey",
    "EncounterKey",
    "Date",
    "IsEdVisit",
    "IsHospitalAdmission",
    "IsInpatientAdmission",
    "DepartmentType",
    "DepartmentSpecialty",
    "PatientBirthYearBin",
    "SexAssignedAtBirth",
    "FirstRace",
    "OmbEthnicity",
    "VitalStatus",
    "SmokingStatus",
    "MyChartStatus",
    "MaritalStatus",
    "SDOH_Financial_Strain",
    "SDOH_Food_Insecurity",
    "SDOH_Housing_Instability",
    "SDOH_Transportation_Barrier",
    "SDOH_Depression_Flag",
    "SDOH_Social_Isolation",
    "SDOH_Stress_Flag",
    "SDOH_Alcohol_Use",
    "SDOH_Utilities_Risk",
    "SDOH_Any_Risk",
    "SDOH_Domain_Count",
]

PROFILE_COLS = [
    "PatientBirthYearBin",
    "SexAssignedAtBirth",
    "FirstRace",
    "OmbEthnicity",
    "VitalStatus",
    "SmokingStatus",
    "MyChartStatus",
    "MaritalStatus",
]

SDOH_DOMAIN_COLS = [
    "SDOH_Financial_Strain",
    "SDOH_Food_Insecurity",
    "SDOH_Housing_Instability",
    "SDOH_Transportation_Barrier",
    "SDOH_Depression_Flag",
    "SDOH_Social_Isolation",
    "SDOH_Stress_Flag",
    "SDOH_Alcohol_Use",
    "SDOH_Utilities_Risk",
]

SDOH_ALL_COLS = SDOH_DOMAIN_COLS + ["SDOH_Any_Risk", "SDOH_Domain_Count"]

FEATURE_COLS = [
    "total_encounters",
    "ed_hospital_ratio",
    "mean_gap_days",
    "std_gap_days",
    "max_gap_days",
    "silence_to_ed_prob",
]


def stage(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def default_input_csv(repo_root: Path) -> Path:
    candidates = [
        repo_root / "outputs" / "integrated_master.csv",
        repo_root / "integrated_master.csv",
        repo_root / "outputs" / "chronic_visit_master_2022_2025.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate an input CSV. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run the chronic disease analysis pipeline.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=default_input_csv(repo_root),
        help="Integrated master CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "outputs",
        help="Directory to save all outputs.",
    )
    parser.add_argument(
        "--max-silhouette-sample",
        type=int,
        default=10000,
        help="Maximum sample size used for silhouette score calculation.",
    )
    return parser.parse_args()


def read_master_table(input_csv: Path) -> pd.DataFrame:
    dtype_map = {
        "PatientDurableKey": "int64",
        "EncounterKey": "int64",
        "Date": "string",
        "IsEdVisit": "Int8",
        "IsHospitalAdmission": "Int8",
        "IsInpatientAdmission": "Int8",
        "DepartmentType": "category",
        "DepartmentSpecialty": "category",
        "PatientBirthYearBin": "Float32",
        "SexAssignedAtBirth": "category",
        "FirstRace": "category",
        "OmbEthnicity": "category",
        "VitalStatus": "category",
        "SmokingStatus": "category",
        "MyChartStatus": "category",
        "MaritalStatus": "category",
        "SDOH_Financial_Strain": "Int8",
        "SDOH_Food_Insecurity": "Int8",
        "SDOH_Housing_Instability": "Int8",
        "SDOH_Transportation_Barrier": "Int8",
        "SDOH_Depression_Flag": "Int8",
        "SDOH_Social_Isolation": "Int8",
        "SDOH_Stress_Flag": "Int8",
        "SDOH_Alcohol_Use": "Int8",
        "SDOH_Utilities_Risk": "Int8",
        "SDOH_Any_Risk": "Int8",
        "SDOH_Domain_Count": "Int8",
    }
    df = pd.read_csv(
        input_csv,
        usecols=EVENT_COLS,
        dtype=dtype_map,
        encoding="utf-8-sig",
        low_memory=False,
    )
    return df


def print_basic_diagnostics(df: pd.DataFrame) -> None:
    stage("Input Diagnostics")
    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")
    duplicate_encounters = int(df["EncounterKey"].duplicated(keep=False).sum())
    duplicate_keys = int(df["EncounterKey"].duplicated().sum())
    print(f"Duplicate EncounterKey rows detected: {duplicate_encounters:,}")
    print(f"Duplicate EncounterKey rows to drop: {duplicate_keys:,}")

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%y", errors="coerce")
    invalid_dates = int(df["Date"].isna().sum())
    print(f"Invalid Date values coerced to NaT: {invalid_dates:,}")

    if duplicate_keys > 0:
        print(
            "Data quality note: EncounterKey is not unique in the source CSV. "
            "The pipeline will deduplicate to one row per EncounterKey before sequence construction."
        )


def deduplicate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["PatientDurableKey", "Date", "EncounterKey"], kind="mergesort")
    df = df.drop_duplicates(subset=["EncounterKey"], keep="first").reset_index(drop=True)
    print(f"Rows after EncounterKey deduplication: {len(df):,}")

    dept_type = df["DepartmentType"].astype("string").str.upper()
    specialty = df["DepartmentSpecialty"].astype("string")
    specialty_pattern = "|".join(KEYWORDS)

    high_risk = (
        df["IsEdVisit"].fillna(0).astype("int8").eq(1)
        | df["IsHospitalAdmission"].fillna(0).astype("int8").eq(1)
        | df["IsInpatientAdmission"].fillna(0).astype("int8").eq(1)
        | dept_type.isin(HIGH_RISK_DEPARTMENTS)
    )
    specialist = specialty.str.contains(specialty_pattern, case=False, na=False)
    df["event_code"] = np.select(
        [high_risk, specialist],
        [3, 2],
        default=1,
    ).astype("int8")

    df["gap_days"] = (
        df.groupby("PatientDurableKey", sort=False)["Date"]
        .diff()
        .dt.days.astype("float32")
    )
    df["silence_before"] = df["gap_days"].gt(90).fillna(False)
    return df


def first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def build_patient_level_outputs(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    stage("Stage 1: Sequence Construction")
    grouped = df.groupby("PatientDurableKey", sort=False, observed=True)

    sequence_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    profile_rows: List[Dict[str, object]] = []
    conflict_counts = {col: 0 for col in PROFILE_COLS}

    for patient_id, group in grouped:
        group = group.sort_values(["Date", "EncounterKey"], kind="mergesort")
        codes = group["event_code"].to_numpy(dtype=np.int8, copy=False)
        silence_flags = group["silence_before"].to_numpy(dtype=bool, copy=False)

        sequence: List[int] = []
        for silence_flag, code in zip(silence_flags, codes):
            if silence_flag:
                sequence.append(0)
            sequence.append(int(code))
        sequence_rows.append(
            {
                "PatientDurableKey": int(patient_id),
                "sequence": json.dumps(sequence, ensure_ascii=False),
                "sequence_length": len(sequence),
            }
        )

        gaps = group["gap_days"].dropna().to_numpy(dtype=np.float32, copy=False)
        total_encounters = int(len(group))
        ed_hospital_count = int((codes == 3).sum())
        silence_count = int(silence_flags.sum())
        silence_to_ed_count = int(((codes == 3) & silence_flags).sum())

        feature_rows.append(
            {
                "PatientDurableKey": int(patient_id),
                "total_encounters": total_encounters,
                "ed_hospital_ratio": ed_hospital_count / total_encounters if total_encounters else 0.0,
                "mean_gap_days": float(gaps.mean()) if len(gaps) else 0.0,
                "std_gap_days": float(gaps.std(ddof=0)) if len(gaps) > 1 else 0.0,
                "max_gap_days": float(gaps.max()) if len(gaps) else 0.0,
                "silence_to_ed_prob": silence_to_ed_count / silence_count if silence_count else 0.0,
            }
        )

        profile_record: Dict[str, object] = {"PatientDurableKey": int(patient_id)}
        for col in PROFILE_COLS:
            values = group[col].dropna().unique()
            if len(values) > 1:
                conflict_counts[col] += 1
            profile_record[col] = values[0] if len(values) else np.nan

        for col in SDOH_ALL_COLS:
            profile_record[col] = int(group[col].fillna(0).astype("int16").max())
        profile_rows.append(profile_record)

    sequences = pd.DataFrame(sequence_rows)
    features = pd.DataFrame(feature_rows)
    profiles = pd.DataFrame(profile_rows)

    seq_lengths = sequences["sequence_length"]
    seq_stats = {
        "min": int(seq_lengths.min()),
        "p25": float(seq_lengths.quantile(0.25)),
        "median": float(seq_lengths.median()),
        "mean": float(seq_lengths.mean()),
        "p75": float(seq_lengths.quantile(0.75)),
        "max": int(seq_lengths.max()),
    }
    print("Sequence length stats:")
    print(pd.Series(seq_stats).to_string())

    code_distribution = {
        0: int(df["silence_before"].sum()),
        1: int((df["event_code"] == 1).sum()),
        2: int((df["event_code"] == 2).sum()),
        3: int((df["event_code"] == 3).sum()),
    }
    print("Overall code distribution:")
    print(pd.Series(code_distribution).rename("count").to_string())

    total_profile_conflicts = sum(conflict_counts.values())
    if total_profile_conflicts:
        print("Profile consistency warnings (patients with >1 non-null value):")
        print(pd.Series(conflict_counts).to_string())
    else:
        print("Profile consistency check: no multi-valued patient profile columns detected.")

    sdoh_positive_rate = float(profiles["SDOH_Any_Risk"].mean())
    print(f"Patient-level any-SDOH-positive rate (max across encounters): {sdoh_positive_rate:.4f}")
    if sdoh_positive_rate < 0.05:
        print(
            "Data quality note: SDOH positives are sparse. In the current source table, "
            "many zero values likely encode either true negatives or no SDOH data."
        )

    return sequences, features, profiles, code_distribution


def save_sequence_outputs(sequences: pd.DataFrame, output_dir: Path) -> Path:
    sequence_path = output_dir / "patient_sequences.csv"
    sequences[["PatientDurableKey", "sequence"]].to_csv(sequence_path, index=False, encoding="utf-8-sig")
    return sequence_path


def save_feature_outputs(features: pd.DataFrame, output_dir: Path) -> Path:
    feature_path = output_dir / "patient_features.csv"
    features.to_csv(feature_path, index=False, encoding="utf-8-sig")
    return feature_path


def plot_feature_correlation(features: pd.DataFrame, output_dir: Path) -> Path:
    stage("Stage 2: Feature Engineering")
    print(features[FEATURE_COLS].describe().to_string())

    corr = features[FEATURE_COLS].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    out_path = output_dir / "feature_correlation_heatmap.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    return out_path


def choose_best_k(ks: List[int], inertias: List[float], silhouettes: List[float]) -> Tuple[int, int, int]:
    if len(ks) >= 3:
        curvature = np.abs(np.diff(inertias, 2))
        elbow_k = ks[int(np.argmax(curvature)) + 1]
    else:
        elbow_k = ks[0]
    silhouette_k = ks[int(np.nanargmax(silhouettes))]

    if elbow_k == silhouette_k:
        return elbow_k, elbow_k, silhouette_k
    return 4 if 4 in ks else silhouette_k, elbow_k, silhouette_k


def run_clustering(
    features: pd.DataFrame,
    output_dir: Path,
    max_silhouette_sample: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage("Stage 3: Clustering")
    X = features[FEATURE_COLS].to_numpy(dtype=np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ks = list(range(2, 9))
    inertias: List[float] = []
    silhouettes: List[float] = []

    sample_n = min(max_silhouette_sample, len(features))
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = np.sort(rng.choice(len(features), size=sample_n, replace=False))
    X_sample = X_scaled[sample_idx]
    print(f"Silhouette calculation sample size: {sample_n:,}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*divide by zero encountered in matmul.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*overflow encountered in matmul.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*invalid value encountered in matmul.*", category=RuntimeWarning)
        for k in ks:
            model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
            labels = model.fit_predict(X_scaled)
            inertias.append(float(model.inertia_))
            sample_labels = labels[sample_idx]
            silhouettes.append(float(silhouette_score(X_sample, sample_labels)))

    best_k, elbow_k, silhouette_k = choose_best_k(ks, inertias, silhouettes)
    print(f"Elbow candidate K: {elbow_k}")
    print(f"Silhouette candidate K: {silhouette_k}")
    print(f"Selected K for final K-Means: {best_k}")

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(ks, inertias, marker="o", color="tab:blue", label="Inertia")
    ax1.set_xlabel("K")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_title("Elbow Plot and Silhouette Scores")

    ax2 = ax1.twinx()
    ax2.plot(ks, silhouettes, marker="s", color="tab:orange", label="Silhouette")
    ax2.set_ylabel("Silhouette Score", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax1.axvline(best_k, color="tab:green", linestyle="--", linewidth=1.5)
    fig.tight_layout()
    elbow_path = output_dir / "elbow_and_silhouette.png"
    plt.savefig(elbow_path, dpi=DPI)
    plt.close()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*divide by zero encountered in matmul.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*overflow encountered in matmul.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*invalid value encountered in matmul.*", category=RuntimeWarning)
        final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
        cluster_labels = final_model.fit_predict(X_scaled)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca_coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(
        {
            "PC1": pca_coords[:, 0],
            "PC2": pca_coords[:, 1],
            "cluster_label": cluster_labels,
        }
    )
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="cluster_label",
        palette="tab10",
        s=16,
        alpha=0.65,
        linewidth=0,
    )
    plt.title("PCA Scatter of Patient Clusters")
    plt.tight_layout()
    pca_path = output_dir / "cluster_pca_scatter.png"
    plt.savefig(pca_path, dpi=DPI)
    plt.close()

    cluster_df = features[["PatientDurableKey"]].copy()
    cluster_df["cluster_label"] = cluster_labels.astype(int)
    centers = (
        pd.concat([features, cluster_df["cluster_label"]], axis=1)
        .groupby("cluster_label", sort=True)[FEATURE_COLS]
        .mean()
        .reset_index()
    )
    print("Cluster centers (feature means):")
    print(centers.to_string(index=False))

    dtw_df = pd.DataFrame(columns=["PatientDurableKey", "cluster_label_dtw"])
    if len(features) < 5000:
        print("DTW clustering requested by rule, but optional package installation is not part of this run.")
    else:
        print(f"Skipping DTW clustering because patient count is {len(features):,} >= 5,000.")

    return cluster_df, centers, dtw_df


def save_cluster_outputs(
    cluster_df: pd.DataFrame,
    centers: pd.DataFrame,
    dtw_df: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    cluster_path = output_dir / "patient_clusters.csv"
    if not dtw_df.empty:
        cluster_out = cluster_df.merge(dtw_df, on="PatientDurableKey", how="left")
    else:
        cluster_out = cluster_df
    cluster_out.to_csv(cluster_path, index=False, encoding="utf-8-sig")

    center_path = output_dir / "cluster_centers.csv"
    centers.to_csv(center_path, index=False, encoding="utf-8-sig")
    return cluster_path, center_path


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    age = DATE_REFERENCE_YEAR - df["PatientBirthYearBin"]
    df = df.copy()
    df["AgeGroup"] = pd.cut(
        age,
        bins=[-np.inf, 44, 54, 64, 74, np.inf],
        labels=["<45", "45-54", "55-64", "65-74", "75+"],
    )
    return df


def plot_cluster_feature_boxplots(patient_level: pd.DataFrame, output_dir: Path) -> Path:
    melted = patient_level.melt(
        id_vars=["cluster_label"],
        value_vars=FEATURE_COLS,
        var_name="feature",
        value_name="value",
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, feature in zip(axes.flat, FEATURE_COLS):
        subset = melted[melted["feature"] == feature]
        sns.boxplot(data=subset, x="cluster_label", y="value", ax=ax, color="#6baed6", fliersize=1.5)
        ax.set_title(feature)
        ax.set_xlabel("Cluster")
    plt.suptitle("Behavior Feature Boxplots by Cluster", y=1.02)
    plt.tight_layout()
    out_path = output_dir / "cluster_feature_boxplots.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    return out_path


def plot_age_distribution(patient_level: pd.DataFrame, output_dir: Path) -> Path:
    age_dist = pd.crosstab(patient_level["cluster_label"], patient_level["AgeGroup"], normalize="index")
    age_dist = age_dist.fillna(0)
    ax = age_dist.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="viridis")
    ax.set_title("Age Distribution by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    plt.legend(title="Age Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path = output_dir / "cluster_age_distribution.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    return out_path


def plot_mychart_rate(patient_level: pd.DataFrame, output_dir: Path) -> Path:
    mychart_rate = (
        patient_level.assign(
            mychart_activated=patient_level["MyChartStatus"]
            .astype("string")
            .str.upper()
            .eq("ACTIVATED")
            .astype(float)
        )
        .groupby("cluster_label", sort=True)["mychart_activated"]
        .mean()
        .mul(100)
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=mychart_rate, x="cluster_label", y="mychart_activated", color="#74c476")
    plt.title("MyChart Activation Rate by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Percent Activated")
    plt.tight_layout()
    out_path = output_dir / "cluster_mychart_rate.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    return out_path


def plot_sdoh_radar(patient_level: pd.DataFrame, output_dir: Path) -> Path:
    radar = patient_level.groupby("cluster_label", sort=True)[SDOH_DOMAIN_COLS].mean()
    labels = [col.replace("SDOH_", "").replace("_", " ") for col in SDOH_DOMAIN_COLS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for cluster, row in radar.iterrows():
        values = row.to_numpy(dtype=float)
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, linewidth=2, label=f"Cluster {cluster}")
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(0.1, float(radar.max().max()) * 1.1))
    ax.set_title("SDOH Positive Rate Radar by Cluster")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    out_path = output_dir / "cluster_sdoh_radar.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    return out_path


def plot_smoking_distribution(patient_level: pd.DataFrame, output_dir: Path) -> Path:
    smoking = patient_level["SmokingStatus"].astype("string").fillna("Missing")
    smoking_dist = pd.crosstab(patient_level["cluster_label"], smoking, normalize="index")
    ax = smoking_dist.plot(kind="bar", stacked=True, figsize=(11, 6), colormap="tab20")
    ax.set_title("Smoking Status Distribution by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    plt.legend(title="Smoking Status", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path = output_dir / "cluster_smoking.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    return out_path


def plot_mortality(patient_level: pd.DataFrame, output_dir: Path) -> Path:
    mortality = (
        patient_level.assign(
            deceased=patient_level["VitalStatus"]
            .astype("string")
            .str.upper()
            .eq("DECEASED")
            .astype(float)
        )
        .groupby("cluster_label", sort=True)["deceased"]
        .mean()
        .mul(100)
        .reset_index()
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(data=mortality, x="cluster_label", y="deceased", color="#fb6a4a")
    plt.title("Mortality Rate by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Percent Deceased")
    plt.tight_layout()
    out_path = output_dir / "cluster_mortality.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    return out_path


def build_timeline_points(group: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    previous_date = None
    for row in group.itertuples(index=False):
        current_date = row.Date
        if pd.notna(current_date) and row.silence_before and previous_date is not None:
            midpoint = previous_date + (current_date - previous_date) / 2
            rows.append({"Date": midpoint, "event_code": 0})
        rows.append({"Date": current_date, "event_code": int(row.event_code)})
        previous_date = current_date
    return pd.DataFrame(rows)


def plot_example_timelines(
    event_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    merged = cluster_df.copy()
    rng = np.random.default_rng(RANDOM_STATE)
    cluster_ids = sorted(merged["cluster_label"].unique())
    selected: List[Tuple[int, int]] = []
    for cluster in cluster_ids:
        patient_ids = merged.loc[merged["cluster_label"] == cluster, "PatientDurableKey"].to_numpy()
        n_pick = min(3, len(patient_ids))
        if n_pick == 0:
            continue
        picks = rng.choice(patient_ids, size=n_pick, replace=False)
        for patient_id in picks:
            selected.append((cluster, int(patient_id)))

    ncols = 3
    nrows = max(1, math.ceil(len(selected) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
    palette = {0: "#969696", 1: "#3182bd", 2: "#31a354", 3: "#de2d26"}

    for ax in axes.flat:
        ax.axis("off")

    for ax, (cluster, patient_id) in zip(axes.flat, selected):
        ax.axis("on")
        patient_events = event_df.loc[event_df["PatientDurableKey"] == patient_id, ["Date", "event_code", "silence_before"]]
        timeline = build_timeline_points(patient_events)
        ax.plot(timeline["Date"], timeline["event_code"], color="#636363", linewidth=1, alpha=0.7)
        for code, sub in timeline.groupby("event_code"):
            ax.scatter(sub["Date"], sub["event_code"], s=24, color=palette[int(code)], label=f"Code {int(code)}")
        ax.set_title(f"Cluster {cluster} | Patient {patient_id}")
        ax.set_ylim(-0.2, 3.2)
        ax.set_yticks([0, 1, 2, 3])
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=45)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color, label=f"Code {code}")
        for code, color in palette.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4)
    fig.suptitle("Example Behavior Timelines by Cluster", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = output_dir / "cluster_example_timelines.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    return out_path


def build_cluster_summary(patient_level: pd.DataFrame) -> pd.DataFrame:
    enriched = patient_level.copy()
    enriched["mychart_activated"] = enriched["MyChartStatus"].astype("string").str.upper().eq("ACTIVATED")
    enriched["deceased"] = enriched["VitalStatus"].astype("string").str.upper().eq("DECEASED")

    summary = (
        enriched.groupby("cluster_label", sort=True)
        .agg(
            patient_count=("PatientDurableKey", "size"),
            mean_total_encounters=("total_encounters", "mean"),
            mean_ed_hospital_ratio=("ed_hospital_ratio", "mean"),
            mean_max_gap_days=("max_gap_days", "mean"),
            mean_silence_to_ed_prob=("silence_to_ed_prob", "mean"),
            pct_mychart_activated=("mychart_activated", lambda s: float(np.mean(s) * 100)),
            pct_deceased=("deceased", lambda s: float(np.mean(s) * 100)),
            pct_any_sdoh_risk=("SDOH_Any_Risk", lambda s: float(np.mean(s) * 100)),
            mean_sdoh_domain_count=("SDOH_Domain_Count", "mean"),
        )
        .reset_index()
    )

    age_counts = pd.crosstab(enriched["cluster_label"], enriched["AgeGroup"])
    top_age_group = age_counts.idxmax(axis=1).rename("top_age_group").reset_index()
    summary = summary.merge(top_age_group, on="cluster_label", how="left")
    ordered_cols = [
        "cluster_label",
        "patient_count",
        "mean_total_encounters",
        "mean_ed_hospital_ratio",
        "mean_max_gap_days",
        "mean_silence_to_ed_prob",
        "top_age_group",
        "pct_mychart_activated",
        "pct_deceased",
        "pct_any_sdoh_risk",
        "mean_sdoh_domain_count",
    ]
    return summary[ordered_cols]


def save_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    out_path = output_dir / "cluster_summary.csv"
    summary.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def copy_script(output_dir: Path) -> None:
    target = output_dir / "analysis_pipeline.py"
    if Path(__file__).resolve() != target.resolve():
        shutil.copy2(Path(__file__).resolve(), target)


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("Set2")

    df = read_master_table(input_csv)
    print_basic_diagnostics(df)
    df = deduplicate_and_prepare(df)

    sequences, features, profiles, _ = build_patient_level_outputs(df)
    save_sequence_outputs(sequences, output_dir)
    save_feature_outputs(features, output_dir)
    plot_feature_correlation(features, output_dir)

    cluster_df, centers, dtw_df = run_clustering(features, output_dir, args.max_silhouette_sample)
    save_cluster_outputs(cluster_df, centers, dtw_df, output_dir)

    stage("Stage 4: Cluster Profiling")
    patient_level = features.merge(cluster_df, on="PatientDurableKey", how="left")
    patient_level = patient_level.merge(profiles, on="PatientDurableKey", how="left")
    patient_level = add_age_group(patient_level)

    plot_cluster_feature_boxplots(patient_level, output_dir)
    plot_age_distribution(patient_level, output_dir)
    plot_mychart_rate(patient_level, output_dir)
    plot_sdoh_radar(patient_level, output_dir)
    plot_smoking_distribution(patient_level, output_dir)
    plot_mortality(patient_level, output_dir)
    plot_example_timelines(df[["PatientDurableKey", "Date", "event_code", "silence_before"]], cluster_df, output_dir)

    summary = build_cluster_summary(patient_level)
    save_summary(summary, output_dir)
    print("Cluster summary:")
    print(summary.to_string(index=False))

    copy_script(output_dir)
    print(f"\nAll outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
