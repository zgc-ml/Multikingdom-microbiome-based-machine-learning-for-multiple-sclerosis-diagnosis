# -*- coding: utf-8 -*-
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Paths
    base_dir: str = r"~/ml/RRMS"
    output_parent_dir: str = r"~/ml/RRMS/results"

    # Train files
    train_metadata_filename: str = "metadata.csv"

    # External validation directory (contains its own metadata.csv + residual tables)
    external_dir: str = r"~/ml/RRMS/external"
    external_metadata_filename: str = "metadata.csv"

    # Residual file map (used for both train and external)
    residual_files: Dict[str, str] = None

    # Modeling switches
    use_smote: bool = True
    smote_ratio: float = 1.0

    # "All" model inclusion toggles
    all_include_archaea: bool = True
    all_include_fungi: bool = True
    all_include_virus: bool = True

    # Random seeds: randomly draw 10 unique integers
    n_random_seeds: int = 10
    seed_low_inclusive: int = 0
    seed_high_inclusive: int = 10_000

    # Train/test split
    test_size: float = 0.30

    # CV / grid search
    n_splits_cv: int = 5

    # Permutation importance
    permutation_top_n: int = 30
    permutation_repeats: int = 20

    # Bootstrap
    auc_ci_boot: int = 1000
    roc_ci_boot: int = 800

    # Plot export
    png_dpi: int = 600

    # RF parameter grid
    rf_param_grid: Dict = None


def make_default_config() -> Config:
    residual_files = {
        "archaea": "archaea_residual.csv",
        "bacteria": "bacteria_residual.csv",
        "fungi": "fungi_residual.csv",
        "virus": "virus_residual.csv",
        "ko": "ko_residual.csv",
        "path": "path_residual.csv",
    }

    rf_param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "class_weight": [None, "balanced"],
    }

    return Config(
        residual_files=residual_files,
        rf_param_grid=rf_param_grid,
    )


# =============================================================================
# Utilities
# =============================================================================

def make_dir(*paths: str) -> str:
    path = os.path.join(*paths)
    os.makedirs(path, exist_ok=True)
    return path


def set_publication_plot_style() -> None:
    """Clean, publication-oriented plotting style (vector-friendly)."""
    sns.set_style("white")
    sns.set_context(
        "paper",
        rc={
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "font.size": 12,
        },
    )
    plt.rcParams.update({
        "font.family": "Arial",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })


def calculate_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    random_state: int = 0
) -> Tuple[float, float, float]:
    """Bootstrap AUC and 95% CI."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    auc_val = roc_auc_score(y_true, y_prob)
    data = np.column_stack([y_true, y_prob])

    rng = np.random.default_rng(random_state)
    auc_boots: List[float] = []

    for _ in range(n_boot):
        idx = rng.choice(len(data), size=len(data), replace=True)
        y_b = data[idx, 0].astype(int)
        p_b = data[idx, 1].astype(float)
        if np.unique(y_b).size < 2:
            continue
        auc_boots.append(roc_auc_score(y_b, p_b))

    if len(auc_boots) == 0:
        return auc_val, auc_val, auc_val

    ci_low, ci_up = np.percentile(auc_boots, [2.5, 97.5])
    return auc_val, float(ci_low), float(ci_up)


def bootstrap_roc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 300,
    random_state: int = 0,
    fpr_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap ROC band (TPR 2.5% and 97.5% at fixed FPR grid)."""
    rng = np.random.RandomState(random_state)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    n_samples = len(y_true)
    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 400)

    tprs: List[np.ndarray] = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n_samples, n_samples)
        if np.unique(y_true[idx]).size < 2:
            continue
        fpr_i, tpr_i, _ = roc_curve(y_true[idx], y_prob[idx])
        tpr_interp = np.interp(fpr_grid, fpr_i, tpr_i)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    if len(tprs) == 0:
        tprs = [np.linspace(0, 1, len(fpr_grid))]

    tprs = np.asarray(tprs)
    tprs_lower = np.percentile(tprs, 2.5, axis=0)
    tprs_upper = np.percentile(tprs, 97.5, axis=0)
    return fpr_grid, tprs_lower, tprs_upper


def evaluate_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    random_state: int = 0,
    n_boot_auc: int = 1000,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    auc_v, auc_l, auc_u = calculate_auc_ci(
        y_true, y_prob, n_boot=n_boot_auc, random_state=random_state
    )
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "AUC": float(auc_v),
        "AUC_95CI_lower": float(auc_l),
        "AUC_95CI_upper": float(auc_u),
        "Accuracy": float(acc),
        "Precision": float(pre),
        "Recall": float(rec),
        "F1": float(f1),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }


def align_external_features_to_training(
    X_ext: pd.DataFrame,
    training_feature_names: List[str],
    fill_value: float = 0.0,
    verbose_prefix: str = ""
) -> pd.DataFrame:
    """
    Align external features to training features:
      - add missing columns (filled with fill_value)
      - drop extra columns
      - enforce exact column order
    """
    training_feature_names = list(training_feature_names)
    ext_cols = set(X_ext.columns)
    train_cols = set(training_feature_names)

    missing = [c for c in training_feature_names if c not in ext_cols]
    extra = [c for c in X_ext.columns if c not in train_cols]

    if len(missing) > 0 and verbose_prefix:
        print(f"{verbose_prefix} Missing {len(missing)} training features in external set; filling with {fill_value}.")
    for c in missing:
        X_ext[c] = fill_value

    if len(extra) > 0 and verbose_prefix:
        print(f"{verbose_prefix} Found {len(extra)} extra features in external set; dropping them.")
        X_ext = X_ext.drop(columns=extra)

    return X_ext[training_feature_names].copy()


def get_all_include_types(cfg: Config) -> Set[str]:
    include = {"bacteria", "ko", "path"}
    if cfg.all_include_archaea:
        include.add("archaea")
    if cfg.all_include_fungi:
        include.add("fungi")
    if cfg.all_include_virus:
        include.add("virus")
    return include


def build_all_features(
    df_type_dict: Dict[str, pd.DataFrame],
    include_types: Optional[Set[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Concatenate multi-table features and prefix columns with "type|feature".
    Input matrices must be (rows=samples, cols=features).
    """
    if include_types is None:
        include_types = set(df_type_dict.keys())

    blocks: List[pd.DataFrame] = []
    for t_name in sorted(include_types):
        if t_name not in df_type_dict:
            continue
        df_tmp = df_type_dict[t_name].copy()
        df_tmp.columns = [f"{t_name}|{c}" for c in df_tmp.columns.astype(str)]
        blocks.append(df_tmp)

    if len(blocks) == 0:
        return None
    return pd.concat(blocks, axis=1)


def plot_roc_curves_with_ci(
    roc_info: Dict[str, Tuple[np.ndarray, np.ndarray, float, float, float]],
    title: str,
    out_png: str,
    out_pdf: Optional[str] = None,
    random_state: int = 0,
    n_bootstrap: int = 800,
    dpi: int = 600,
    keep_order: bool = True,
) -> None:
    """
    roc_info: {name: (y_true, y_prob, auc, ci_l, ci_u)}
    """
    set_publication_plot_style()
    plt.figure(figsize=(7.2, 6.6))

    fpr_grid = np.linspace(0, 1, 400)
    color_cycle = sns.color_palette("tab10", n_colors=max(8, len(roc_info)))

    items = list(roc_info.items())
    if not keep_order:
        items = sorted(items, key=lambda kv: kv[0])

    for i, (name, (yt, yp, auc_v, ci_l, ci_u)) in enumerate(items):
        color = color_cycle[i % len(color_cycle)]
        fpr, tpr, _ = roc_curve(yt, yp)
        _, tprs_lower, tprs_upper = bootstrap_roc_ci(
            yt, yp, n_bootstrap=n_bootstrap, random_state=random_state, fpr_grid=fpr_grid
        )
        plt.fill_between(fpr_grid, tprs_lower, tprs_upper, color=color, alpha=0.12, linewidth=0)
        plt.plot(
            fpr, tpr, color=color,
            label=f"{name}  AUC={auc_v:.3f} ({ci_l:.3f}–{ci_u:.3f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="0.35", linewidth=1.8)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if title:
        plt.title(title, pad=10)

    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    if out_pdf is not None:
        plt.savefig(out_pdf)
    plt.close()


# =============================================================================
# Data loading
# =============================================================================

def load_metadata(meta_path: str) -> pd.DataFrame:
    meta = pd.read_csv(meta_path)
    required = {"SampleID", "disease", "sex"}
    if not required.issubset(set(meta.columns)):
        raise ValueError(f"metadata must contain columns {required}, got: {meta.columns.tolist()}")

    meta = meta[["SampleID", "disease", "sex"]].copy()
    meta["SampleID"] = meta["SampleID"].astype(str)

    label_map = {"Control": 0, "MS": 1}
    meta["label"] = meta["disease"].map(label_map)
    if meta["label"].isna().any():
        bad = meta.loc[meta["label"].isna(), "disease"].unique().tolist()
        raise ValueError(f"metadata contains unknown disease labels (expected Control/MS): {bad}")

    meta = meta.set_index("SampleID")
    return meta


def load_residual_tables(
    base_dir: str,
    residual_files: Dict[str, str]
) -> Dict[str, pd.DataFrame]:
    """
    Load residual CSVs: rows=features, cols=samples. Transpose to rows=samples, cols=features.
    """
    out: Dict[str, pd.DataFrame] = {}
    for t_name, fname in residual_files.items():
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] Missing residual file for {t_name}: {fpath} (skipped)")
            continue
        df_res = pd.read_csv(fpath, index_col=0).T
        df_res.index = df_res.index.astype(str)
        df_res.columns = df_res.columns.astype(str)
        out[t_name] = df_res
    return out


def intersect_samples(
    metadata: pd.DataFrame,
    features_dict: Dict[str, pd.DataFrame]
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    common = set(metadata.index)
    for _, df in features_dict.items():
        common = common.intersection(df.index)

    if len(common) == 0:
        raise ValueError("No overlapping SampleID between metadata and residual tables.")

    common = sorted(common)
    metadata2 = metadata.loc[common].copy()
    features2 = {k: v.loc[common].copy() for k, v in features_dict.items()}
    return metadata2, features2


# =============================================================================
# Modeling
# =============================================================================

def fit_rf_with_gridsearch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: Config,
    random_state: int
) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=cfg.n_splits_cv, shuffle=True, random_state=random_state)
    rf_base = RandomForestClassifier(
        random_state=random_state,
        oob_score=False,
        n_jobs=-1,
    )
    grid = GridSearchCV(
        estimator=rf_base,
        param_grid=cfg.rf_param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )
    grid.fit(X_train, y_train)
    return grid


def train_evaluate_for_sex(
    sex_label: str,
    features_dict: Dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    output_root: str,
    random_state: int,
    cfg: Config,
    metadata_ext: Optional[pd.DataFrame] = None,
    features_ext_dict: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    print("\n" + "#" * 86)
    print(f"Modeling: sex={sex_label} | seed={random_state}")
    print("#" * 86)

    sex_mask = (metadata["sex"] == sex_label)
    if int(sex_mask.sum()) < 10:
        print(f"[INFO] Too few samples for sex={sex_label} (<10). Skipped.")
        return

    meta_sex = metadata.loc[sex_mask].copy()
    y = meta_sex["label"].to_numpy(dtype=int)
    sample_ids = meta_sex.index.tolist()

    # Build per-type feature matrices for this sex
    X_type_dict: Dict[str, pd.DataFrame] = {}
    for t_name, df in features_dict.items():
        X_sub = df.loc[sample_ids].copy()
        if X_sub.shape[1] == 0:
            continue
        X_type_dict[t_name] = X_sub

    if len(X_type_dict) == 0:
        print(f"[INFO] No usable feature tables for sex={sex_label}. Skipped.")
        return

    include_types = get_all_include_types(cfg)
    X_all = build_all_features(X_type_dict, include_types=include_types)
    if X_all is None or X_all.shape[1] == 0:
        raise ValueError(f"No features for All model under include_types={sorted(include_types)}")

    print(f"[INFO] All include_types = {sorted(include_types)}")
    print(f"[INFO] All feature count = {X_all.shape[1]}")

    # Unified train/test split for all models
    X_train_dummy, X_test_dummy, y_train, y_test = train_test_split(
        X_all, y, test_size=cfg.test_size, random_state=random_state, stratify=y
    )
    train_index = X_train_dummy.index
    test_index = X_test_dummy.index

    # Output directories
    sex_dir = make_dir(output_root, f"sex_{sex_label}")
    metrics_dir = make_dir(sex_dir, "metrics")
    fig_dir = make_dir(sex_dir, "figures")
    imp_dir = make_dir(sex_dir, "permutation_importance")
    ext_dir = make_dir(sex_dir, "external_validation")
    ext_metrics_dir = make_dir(ext_dir, "metrics")
    ext_fig_dir = make_dir(ext_dir, "figures")

    results_rows: List[List] = []
    roc_info_internal: Dict[str, Tuple[np.ndarray, np.ndarray, float, float, float]] = {}

    # -------------------------------------------------------------------------
    # Single-table models
    # -------------------------------------------------------------------------
    for t_name in sorted(X_type_dict.keys()):
        print(f"\n[{sex_label}] Single-table model: {t_name}")

        X_t = X_type_dict[t_name]
        X_train_t = X_t.loc[train_index].to_numpy(dtype=float)
        X_test_t = X_t.loc[test_index].to_numpy(dtype=float)

        scaler = StandardScaler()
        X_train_t_scaled = scaler.fit_transform(X_train_t)
        X_test_t_scaled = scaler.transform(X_test_t)

        y_train_series = pd.Series(y_train, index=train_index)

        if cfg.use_smote:
            sm = SMOTE(sampling_strategy=cfg.smote_ratio, random_state=random_state)
            X_res, y_res = sm.fit_resample(X_train_t_scaled, y_train_series)
            print(f"[INFO] SMOTE class counts: before={y_train_series.value_counts().to_dict()} after={pd.Series(y_res).value_counts().to_dict()}")
        else:
            X_res, y_res = X_train_t_scaled, y_train_series.to_numpy(dtype=int)

        grid = fit_rf_with_gridsearch(X_res, y_res, cfg, random_state)
        best_rf = grid.best_estimator_
        print(f"[INFO] Best params: {grid.best_params_}")
        print(f"[INFO] Best CV AUC: {grid.best_score_:.4f}")

        best_rf.fit(X_res, y_res)
        y_prob_test = best_rf.predict_proba(X_test_t_scaled)[:, 1]

        m = evaluate_binary_metrics(
            y_test, y_prob_test,
            threshold=0.5,
            random_state=random_state,
            n_boot_auc=cfg.auc_ci_boot,
        )

        results_rows.append([
            sex_label, t_name, "SingleTable",
            m["AUC"], m["AUC_95CI_lower"], m["AUC_95CI_upper"],
            m["Accuracy"], m["Precision"], m["Recall"], m["F1"],
            m["TN"], m["FP"], m["FN"], m["TP"]
        ])
        roc_info_internal[t_name] = (y_test, y_prob_test, m["AUC"], m["AUC_95CI_lower"], m["AUC_95CI_upper"])

    # -------------------------------------------------------------------------
    # All concatenated model
    # -------------------------------------------------------------------------
    print(f"\n[{sex_label}] All-features model")

    X_train_all = X_all.loc[train_index].to_numpy(dtype=float)
    X_test_all = X_all.loc[test_index].to_numpy(dtype=float)

    scaler_all = StandardScaler()
    X_train_all_scaled = scaler_all.fit_transform(X_train_all)
    X_test_all_scaled = scaler_all.transform(X_test_all)

    y_train_series = pd.Series(y_train, index=train_index)
    if cfg.use_smote:
        sm = SMOTE(sampling_strategy=cfg.smote_ratio, random_state=random_state)
        X_res_all, y_res_all = sm.fit_resample(X_train_all_scaled, y_train_series)
        print(f"[INFO] SMOTE class counts: before={y_train_series.value_counts().to_dict()} after={pd.Series(y_res_all).value_counts().to_dict()}")
    else:
        X_res_all, y_res_all = X_train_all_scaled, y_train_series.to_numpy(dtype=int)

    grid_all = fit_rf_with_gridsearch(X_res_all, y_res_all, cfg, random_state)
    best_rf_all = grid_all.best_estimator_
    print(f"[INFO] Best params: {grid_all.best_params_}")
    print(f"[INFO] Best CV AUC: {grid_all.best_score_:.4f}")

    best_rf_all.fit(X_res_all, y_res_all)
    y_prob_all_test = best_rf_all.predict_proba(X_test_all_scaled)[:, 1]

    m_all = evaluate_binary_metrics(
        y_test, y_prob_all_test,
        threshold=0.5,
        random_state=random_state,
        n_boot_auc=cfg.auc_ci_boot,
    )
    results_rows.append([
        sex_label, "All", "AllTables",
        m_all["AUC"], m_all["AUC_95CI_lower"], m_all["AUC_95CI_upper"],
        m_all["Accuracy"], m_all["Precision"], m_all["Recall"], m_all["F1"],
        m_all["TN"], m_all["FP"], m_all["FN"], m_all["TP"]
    ])
    roc_info_internal["All"] = (y_test, y_prob_all_test, m_all["AUC"], m_all["AUC_95CI_lower"], m_all["AUC_95CI_upper"])

    # -------------------------------------------------------------------------
    # Permutation importance on internal test set; Top-N refit
    # -------------------------------------------------------------------------
    print(f"\n[{sex_label}] Permutation importance (Top {cfg.permutation_top_n})")

    perm = permutation_importance(
        estimator=best_rf_all,
        X=X_test_all_scaled,
        y=y_test,
        n_repeats=cfg.permutation_repeats,
        random_state=random_state,
        scoring="roc_auc",
    )
    feature_names_all = X_all.columns.to_list()
    if len(feature_names_all) != len(perm.importances_mean):
        raise RuntimeError("Permutation importance length mismatch with feature names.")

    imp_df = pd.DataFrame({
        "feature": feature_names_all,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)

    top_n = int(min(cfg.permutation_top_n, imp_df.shape[0]))
    imp_top_df = imp_df.head(top_n).copy()
    top_features = imp_top_df["feature"].tolist()

    imp_top_path = os.path.join(imp_dir, f"permutation_top{top_n}_sex_{sex_label}.csv")
    imp_top_df.to_csv(imp_top_path, index=False)

    print(f"[INFO] Saved Top-{top_n} permutation table: {imp_top_path}")

    print(f"\n[{sex_label}] Refit using Top-{top_n} features")

    X_all_top = X_all[top_features]
    X_train_top = X_all_top.loc[train_index].to_numpy(dtype=float)
    X_test_top = X_all_top.loc[test_index].to_numpy(dtype=float)

    scaler_top = StandardScaler()
    X_train_top_scaled = scaler_top.fit_transform(X_train_top)
    X_test_top_scaled = scaler_top.transform(X_test_top)

    y_train_series = pd.Series(y_train, index=train_index)
    if cfg.use_smote:
        sm = SMOTE(sampling_strategy=cfg.smote_ratio, random_state=random_state)
        X_res_top, y_res_top = sm.fit_resample(X_train_top_scaled, y_train_series)
        print(f"[INFO] SMOTE class counts: before={y_train_series.value_counts().to_dict()} after={pd.Series(y_res_top).value_counts().to_dict()}")
    else:
        X_res_top, y_res_top = X_train_top_scaled, y_train_series.to_numpy(dtype=int)

    grid_top = fit_rf_with_gridsearch(X_res_top, y_res_top, cfg, random_state)
    best_rf_top = grid_top.best_estimator_
    print(f"[INFO] Best params: {grid_top.best_params_}")
    print(f"[INFO] Best CV AUC: {grid_top.best_score_:.4f}")

    best_rf_top.fit(X_res_top, y_res_top)
    y_prob_top_test = best_rf_top.predict_proba(X_test_top_scaled)[:, 1]

    m_top = evaluate_binary_metrics(
        y_test, y_prob_top_test,
        threshold=0.5,
        random_state=random_state,
        n_boot_auc=cfg.auc_ci_boot,
    )
    results_rows.append([
        sex_label, f"Top{top_n}", "AllTables_PermutationTopN",
        m_top["AUC"], m_top["AUC_95CI_lower"], m_top["AUC_95CI_upper"],
        m_top["Accuracy"], m_top["Precision"], m_top["Recall"], m_top["F1"],
        m_top["TN"], m_top["FP"], m_top["FN"], m_top["TP"]
    ])
    roc_info_internal[f"Top{top_n}"] = (y_test, y_prob_top_test, m_top["AUC"], m_top["AUC_95CI_lower"], m_top["AUC_95CI_upper"])

    # -------------------------------------------------------------------------
    # Save internal metrics and ROC
    # -------------------------------------------------------------------------
    cols = [
        "sex", "data_type", "model_type",
        "AUC", "AUC_95CI_lower", "AUC_95CI_upper",
        "Accuracy", "Precision", "Recall", "F1",
        "TN", "FP", "FN", "TP",
    ]
    df_metrics = pd.DataFrame(results_rows, columns=cols)
    metrics_path = os.path.join(metrics_dir, f"metrics_internal_sex_{sex_label}.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"[INFO] Saved internal metrics: {metrics_path}")

    roc_png = os.path.join(fig_dir, f"roc_internal_sex_{sex_label}.png")
    roc_pdf = os.path.join(fig_dir, f"roc_internal_sex_{sex_label}.pdf")
    plot_roc_curves_with_ci(
        roc_info=roc_info_internal,
        title="",
        out_png=roc_png,
        out_pdf=roc_pdf,
        random_state=random_state,
        n_bootstrap=cfg.roc_ci_boot,
        dpi=cfg.png_dpi,
        keep_order=True,
    )
    print(f"[INFO] Saved internal ROC: {roc_png} and {roc_pdf}")

    # -------------------------------------------------------------------------
    # External validation (All and Top-N only)
    # -------------------------------------------------------------------------
    if metadata_ext is None or features_ext_dict is None:
        print(f"[INFO] External validation not provided. Skipped for sex={sex_label}.")
        return

    ext_mask = (metadata_ext["sex"] == sex_label)
    if int(ext_mask.sum()) < 5:
        print(f"[INFO] Too few external samples for sex={sex_label} (<5). Skipped external validation.")
        return

    meta_ext_sex = metadata_ext.loc[ext_mask].copy()
    ext_ids = meta_ext_sex.index.tolist()
    y_ext = meta_ext_sex["label"].to_numpy(dtype=int)

    X_ext_type: Dict[str, pd.DataFrame] = {}
    for t_name, df in features_ext_dict.items():
        if not set(ext_ids).issubset(set(df.index)):
            # The external set should already be intersected, but keep robust behavior.
            df = df.loc[df.index.intersection(ext_ids)].copy()
        X_sub = df.loc[ext_ids].copy()
        if X_sub.shape[1] == 0:
            continue
        X_ext_type[t_name] = X_sub

    X_ext_all = build_all_features(X_ext_type, include_types=include_types)
    if X_ext_all is None or X_ext_all.shape[1] == 0:
        print(f"[INFO] No usable external features under include_types={sorted(include_types)}. Skipped.")
        return

    # External - All
    X_ext_all_aligned = align_external_features_to_training(
        X_ext_all,
        training_feature_names=X_all.columns.tolist(),
        fill_value=0.0,
        verbose_prefix=f"[External-{sex_label}-All]"
    )
    X_ext_all_scaled = scaler_all.transform(X_ext_all_aligned.to_numpy(dtype=float))
    y_prob_ext_all = best_rf_all.predict_proba(X_ext_all_scaled)[:, 1]
    m_ext_all = evaluate_binary_metrics(
        y_ext, y_prob_ext_all,
        threshold=0.5,
        random_state=random_state,
        n_boot_auc=cfg.auc_ci_boot,
    )

    # External - TopN
    X_ext_top_aligned = align_external_features_to_training(
        X_ext_all,
        training_feature_names=top_features,
        fill_value=0.0,
        verbose_prefix=f"[External-{sex_label}-Top{top_n}]"
    )
    X_ext_top_scaled = scaler_top.transform(X_ext_top_aligned.to_numpy(dtype=float))
    y_prob_ext_top = best_rf_top.predict_proba(X_ext_top_scaled)[:, 1]
    m_ext_top = evaluate_binary_metrics(
        y_ext, y_prob_ext_top,
        threshold=0.5,
        random_state=random_state,
        n_boot_auc=cfg.auc_ci_boot,
    )

    ext_rows = [
        [
            sex_label, "All", "ExternalValidation",
            m_ext_all["AUC"], m_ext_all["AUC_95CI_lower"], m_ext_all["AUC_95CI_upper"],
            m_ext_all["Accuracy"], m_ext_all["Precision"], m_ext_all["Recall"], m_ext_all["F1"],
            m_ext_all["TN"], m_ext_all["FP"], m_ext_all["FN"], m_ext_all["TP"],
        ],
        [
            sex_label, f"Top{top_n}", "ExternalValidation",
            m_ext_top["AUC"], m_ext_top["AUC_95CI_lower"], m_ext_top["AUC_95CI_upper"],
            m_ext_top["Accuracy"], m_ext_top["Precision"], m_ext_top["Recall"], m_ext_top["F1"],
            m_ext_top["TN"], m_ext_top["FP"], m_ext_top["FN"], m_ext_top["TP"],
        ],
    ]
    df_ext = pd.DataFrame(ext_rows, columns=cols)
    ext_metrics_path = os.path.join(ext_metrics_dir, f"metrics_external_sex_{sex_label}.csv")
    df_ext.to_csv(ext_metrics_path, index=False)
    print(f"[INFO] Saved external metrics: {ext_metrics_path}")

    # External ROC: separate figures for All and TopN
    roc_ext_all_png = os.path.join(ext_fig_dir, f"roc_external_sex_{sex_label}_All.png")
    roc_ext_all_pdf = os.path.join(ext_fig_dir, f"roc_external_sex_{sex_label}_All.pdf")
    plot_roc_curves_with_ci(
        roc_info={"All": (y_ext, y_prob_ext_all, m_ext_all["AUC"], m_ext_all["AUC_95CI_lower"], m_ext_all["AUC_95CI_upper"])},
        title="",
        out_png=roc_ext_all_png,
        out_pdf=roc_ext_all_pdf,
        random_state=random_state,
        n_bootstrap=cfg.roc_ci_boot,
        dpi=cfg.png_dpi,
        keep_order=True,
    )
    print(f"[INFO] Saved external ROC (All): {roc_ext_all_png} and {roc_ext_all_pdf}")

    roc_ext_top_png = os.path.join(ext_fig_dir, f"roc_external_sex_{sex_label}_Top{top_n}.png")
    roc_ext_top_pdf = os.path.join(ext_fig_dir, f"roc_external_sex_{sex_label}_Top{top_n}.pdf")
    plot_roc_curves_with_ci(
        roc_info={f"Top{top_n}": (y_ext, y_prob_ext_top, m_ext_top["AUC"], m_ext_top["AUC_95CI_lower"], m_ext_top["AUC_95CI_upper"])},
        title="",
        out_png=roc_ext_top_png,
        out_pdf=roc_ext_top_pdf,
        random_state=random_state,
        n_bootstrap=cfg.roc_ci_boot,
        dpi=cfg.png_dpi,
        keep_order=True,
    )
    print(f"[INFO] Saved external ROC (Top{top_n}): {roc_ext_top_png} and {roc_ext_top_pdf}")


# =============================================================================
# Main
# =============================================================================

def draw_random_seeds(
    n: int,
    low_inclusive: int,
    high_inclusive: int,
    master_seed: Optional[int] = None
) -> List[int]:
    """
    Draw n unique random integers in [low_inclusive, high_inclusive].
    If master_seed is provided, the draw is reproducible.
    """
    if high_inclusive < low_inclusive:
        raise ValueError("high_inclusive must be >= low_inclusive")

    rng = np.random.default_rng(master_seed)
    population_size = high_inclusive - low_inclusive + 1
    if n > population_size:
        raise ValueError(f"Cannot draw {n} unique seeds from range size {population_size}")

    seeds = rng.choice(
        np.arange(low_inclusive, high_inclusive + 1, dtype=int),
        size=n,
        replace=False,
    )
    return [int(s) for s in seeds]


if __name__ == "__main__":
    cfg = make_default_config()
    set_publication_plot_style()

    # Output parent directory
    output_parent = make_dir(cfg.output_parent_dir)

    # -------------------------
    # Load training data
    # -------------------------
    train_meta_path = os.path.join(cfg.base_dir, cfg.train_metadata_filename)
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Training metadata not found: {train_meta_path}")

    metadata_train = load_metadata(train_meta_path)
    features_train = load_residual_tables(cfg.base_dir, cfg.residual_files)
    metadata_train, features_train = intersect_samples(metadata_train, features_train)

    print(f"[INFO] Training samples (after intersection): {metadata_train.shape[0]}")
    print(f"[INFO] Feature tables loaded: {sorted(features_train.keys())}")

    # -------------------------
    # Load external validation data (once)
    # -------------------------
    if not os.path.exists(cfg.external_dir):
        raise FileNotFoundError(f"External validation directory not found: {cfg.external_dir}")

    ext_meta_path = os.path.join(cfg.external_dir, cfg.external_metadata_filename)
    if not os.path.exists(ext_meta_path):
        raise FileNotFoundError(f"External metadata not found: {ext_meta_path}")

    metadata_ext = load_metadata(ext_meta_path)
    features_ext = load_residual_tables(cfg.external_dir, cfg.residual_files)
    metadata_ext, features_ext = intersect_samples(metadata_ext, features_ext)

    print(f"[INFO] External samples (after intersection): {metadata_ext.shape[0]}")
    print(f"[INFO] External feature tables loaded: {sorted(features_ext.keys())}")

    # -------------------------
    # Randomly draw 10 seeds
    # -------------------------
    # master_seed can be set for reproducible seed selection in manuscripts.
    # If you want non-reproducible selection each run, set master_seed=None.
    master_seed_for_seed_draw = 20260309
    seeds = draw_random_seeds(
        n=cfg.n_random_seeds,
        low_inclusive=cfg.seed_low_inclusive,
        high_inclusive=cfg.seed_high_inclusive,
        master_seed=master_seed_for_seed_draw,
    )
    print(f"[INFO] Randomly selected seeds (n={len(seeds)}): {seeds}")

    # -------------------------
    # Run modeling across seeds and sex strata
    # -------------------------
    for seed in seeds:
        output_root = make_dir(output_parent, f"model_seed_{seed}")
        print("\n" + "=" * 94)
        print(f"Running seed={seed}")
        print(f"Output root: {output_root}")
        print("=" * 94)

        for sex in ["M", "F"]:
            train_evaluate_for_sex(
                sex_label=sex,
                features_dict=features_train,
                metadata=metadata_train,
                output_root=output_root,
                random_state=seed,
                cfg=cfg,
                metadata_ext=metadata_ext,
                features_ext_dict=features_ext,
            )

    print("\nDone.")
    print(f"All outputs saved under: {output_parent}")