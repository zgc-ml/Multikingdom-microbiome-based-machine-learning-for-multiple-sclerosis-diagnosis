# -*- coding: utf-8 -*-
# Author: ZHU Gaochen
# Description:
#   Grid over:
#     - Feature-weight schemes (FILTERED_HIGH, NORMAL_HIGH, PV_LOW, LOWER_GROUP_WEIGHT)
#     - Top-N feature counts (10, 20, 30, 40, 50)
#     - P-value keep ratios for each data group (several presets)
#   For each (seed, weight-scheme, topN, pval-preset), run the full pipeline, evaluate,
#   export per-run TopN feature lists, and accumulate TopN feature frequency across seeds.

import os
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.combine import SMOTETomek
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ------------------ Reproducibility seeds ------------------
SEED_LIST = [0, 37, 42, 43, 12345]

# ------------------ Global aggregation (across seeds) for TopN features ------------------
aggregate_top_shap = {}  # feature -> {'count': int, 'seeds': set()}
aggregate_top_perm = {}  # feature -> {'count': int, 'seeds': set()}

# ------------------ Search grids ------------------
# 1) Feature-weight schemes (tuples: FILTERED_HIGH, NORMAL_HIGH, PV_LOW, LOWER_GROUP_WEIGHT)
WEIGHT_SCHEMES = [
    (2.0, 2.0, 1e-3, 0.0),
    (3.0, 2.0, 1e-3, 0.0),
    (2.0, 1.5, 5e-3, 0.0),
    (4.0, 2.0, 1e-4, 0.0),
    (2.0, 2.0, 1e-3, 0.2),
]

# 2) Top-N candidates
TOPN_LIST = [10, 20, 30, 40, 50]

# 3) P-value keep ratio presets (dict per preset)
PVAL_KEEP_PRESETS = [
    {
        "archaea": 0.04,
        "bacteria": 0.12,
        "fungi": 0.05,
        "virus": 0.05,
        "ko": 0.01,
        "path": 0.05,
    },
    {
        "archaea": 0.06,
        "bacteria": 0.15,
        "fungi": 0.07,
        "virus": 0.07,
        "ko": 0.02,
        "path": 0.08,
    },
    {
        "archaea": 0.02,
        "bacteria": 0.08,
        "fungi": 0.03,
        "virus": 0.03,
        "ko": 0.005,
        "path": 0.03,
    },
    {
        "archaea": 0.10,
        "bacteria": 0.20,
        "fungi": 0.10,
        "virus": 0.10,
        "ko": 0.05,
        "path": 0.10,
    },
]

# ------------------ Paths ------------------
base_rrms = "RRMS"
filtered_rrms = "filtered_RRMS"
metadata_path = "RRMS_metadata.csv"

# ------------------ Utility: safe directory ------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ------------------ Main loops ------------------
for RANDOM_STATE in SEED_LIST:
    print("\n" + "=" * 100)
    print(f"Running full pipeline with RANDOM_STATE = {RANDOM_STATE}")
    print("=" * 100)

    # Load core data (static across grid within one seed)
    features_df_b = pd.read_csv(f"{base_rrms}/bacteria_species.csv", index_col=0)
    features_df_k = pd.read_csv(f"{base_rrms}/ko.csv", index_col=0)
    features_df_p = pd.read_csv(f"{base_rrms}/path.csv", index_col=0)
    features_df_a = pd.read_csv(f"{base_rrms}/archaea_species", index_col=0)
    features_df_f = pd.read_csv(f"{base_rrms}/fungi_species", index_col=0)
    features_df_v = pd.read_csv(f"{base_rrms}/virus_species", index_col=0)
    metadata_df = pd.read_csv(metadata_path, index_col=0)

    # Helper to load volcano and filtered-list with prefix
    def load_feature_list_with_pval(path, prefix, filtered_path=None):
        df = pd.read_csv(path)
        if "feature" not in df.columns:
            raise ValueError(f"{path} must contain column 'feature'")
        if "pval" not in df.columns:
            df["pval"] = 0.0
        df["pref_feature"] = [f"{prefix}_{f}" for f in df["feature"].astype(str)]
        df = df[["pref_feature", "pval"]].rename(columns={"pref_feature": "feature"})

        filtered_set = set()
        if filtered_path:
            filtered_df = pd.read_csv(filtered_path)
            if "feature" in filtered_df.columns:
                filtered_set = set([f"{prefix}_{f}" for f in filtered_df["feature"].astype(str)])

        return df, filtered_set

    features_b_info, filtered_b = load_feature_list_with_pval(
        f"{filtered_rrms}/bacteria_volcano.csv",
        "b",
        f"{filtered_rrms}/filtered_bacteria.csv",
    )
    features_k_info, filtered_k = load_feature_list_with_pval(
        f"{filtered_rrms}/ko_volcano.csv", "k", f"{filtered_rrms}/filtered_ko.csv"
    )
    features_p_info, filtered_p = load_feature_list_with_pval(
        f"{filtered_rrms}/path_volcano.csv", "p", f"{filtered_rrms}/filtered_path.csv"
    )
    features_a_info, filtered_a = load_feature_list_with_pval(
        f"{filtered_rrms}/archaea_volcano.csv",
        "a",
        f"{filtered_rrms}/filtered_archaea.csv",
    )
    features_f_info, filtered_f = load_feature_list_with_pval(
        f"{filtered_rrms}/fungi_volcano.csv",
        "f",
        f"{filtered_rrms}/filtered_fungi.csv",
    )
    features_v_info, filtered_v = load_feature_list_with_pval(
        f"{filtered_rrms}/virus_volcano.csv",
        "v",
        f"{filtered_rrms}/filtered_virus.csv",
    )

    filtered_features_master = (
        filtered_b.union(filtered_k)
        .union(filtered_p)
        .union(filtered_a)
        .union(filtered_f)
        .union(filtered_v)
    )

    # Transpose to samples-as-rows
    features_df_b = features_df_b.T
    features_df_k = features_df_k.T
    features_df_p = features_df_p.T
    features_df_a = features_df_a.T
    features_df_f = features_df_f.T
    features_df_v = features_df_v.T

    # Prefix columns
    features_df_b.columns = ["b_" + str(col) for col in features_df_b.columns]
    features_df_k.columns = ["k_" + str(col) for col in features_df_k.columns]
    features_df_p.columns = ["p_" + str(col) for col in features_df_p.columns]
    features_df_a.columns = ["a_" + str(col) for col in features_df_a.columns]
    features_df_f.columns = ["f_" + str(col) for col in features_df_f.columns]
    features_df_v.columns = ["v_" + str(col) for col in features_df_v.columns]

    # Align common sample index
    dfs = [features_df_b, features_df_k, features_df_p, features_df_a, features_df_f, features_df_v]
    common_index = metadata_df.index
    for df_tmp in dfs:
        common_index = common_index.intersection(df_tmp.index)
    common_index = sorted(common_index)

    features_df_b = features_df_b.loc[common_index]
    features_df_k = features_df_k.loc[common_index]
    features_df_p = features_df_p.loc[common_index]
    features_df_a = features_df_a.loc[common_index]
    features_df_f = features_df_f.loc[common_index]
    features_df_v = features_df_v.loc[common_index]
    metadata_df = metadata_df.loc[common_index]

    # Loop over p-value presets, weight schemes, and TopN values
    for pval_preset_idx, PVAL_KEEP_RATIOS in enumerate(PVAL_KEEP_PRESETS, start=1):
        print("\n" + "-" * 80)
        print(f"Using PVAL preset #{pval_preset_idx}: {PVAL_KEEP_RATIOS}")
        print("-" * 80)

        # Helper: select by p-value proportion
        def select_by_pval_percentage(info_df, keep_ratio):
            if keep_ratio >= 1.0:
                return info_df["feature"].tolist()
            if keep_ratio <= 0.0:
                return []
            df_sorted = info_df.sort_values("pval", ascending=True)
            k = max(1, int(np.ceil(len(df_sorted) * keep_ratio)))
            return df_sorted["feature"].head(k).tolist()

        # Select columns by current preset
        selected_a_cols = select_by_pval_percentage(features_a_info, PVAL_KEEP_RATIOS.get("archaea", 1.0))
        selected_b_cols = select_by_pval_percentage(features_b_info, PVAL_KEEP_RATIOS.get("bacteria", 1.0))
        selected_f_cols = select_by_pval_percentage(features_f_info, PVAL_KEEP_RATIOS.get("fungi", 1.0))
        selected_v_cols = select_by_pval_percentage(features_v_info, PVAL_KEEP_RATIOS.get("virus", 1.0))
        selected_k_cols = select_by_pval_percentage(features_k_info, PVAL_KEEP_RATIOS.get("ko", 1.0))
        selected_p_cols = select_by_pval_percentage(features_p_info, PVAL_KEEP_RATIOS.get("path", 1.0))

        # Slice dataframes
        fdf_a = features_df_a.loc[:, [c for c in selected_a_cols if c in features_df_a.columns]]
        fdf_b = features_df_b.loc[:, [c for c in selected_b_cols if c in features_df_b.columns]]
        fdf_f = features_df_f.loc[:, [c for c in selected_f_cols if c in features_df_f.columns]]
        fdf_v = features_df_v.loc[:, [c for c in selected_v_cols if c in features_df_v.columns]]
        fdf_k = features_df_k.loc[:, [c for c in selected_k_cols if c in features_df_k.columns]]
        fdf_p = features_df_p.loc[:, [c for c in selected_p_cols if c in features_df_p.columns]]

        print(
            "Feature counts after p-value selection: "
            f"archaea={fdf_a.shape[1]}, bacteria={fdf_b.shape[1]}, fungi={fdf_f.shape[1]}, "
            f"virus={fdf_v.shape[1]}, ko={fdf_k.shape[1]}, path={fdf_p.shape[1]}"
        )

        # Merge features
        features_df = pd.concat([fdf_b, fdf_k, fdf_p, fdf_a, fdf_f, fdf_v], axis=1)

        # Label
        data = features_df.join(metadata_df[["Disease"]], how="inner")
        X_full_all = data.drop("Disease", axis=1)
        y_full_all = data["Disease"].map({"Control": 0, "RRMS": 1})
        target_names = ["Control", "RRMS"]

        valid_idx = y_full_all.dropna().index
        X_full_all = X_full_all.loc[valid_idx]
        y_full_all = y_full_all.loc[valid_idx].astype(int)

        print(f"Total samples: {len(X_full_all)}, total features: {X_full_all.shape[1]}")

        # Build p-value map for weighting
        pval_df = pd.concat(
            [
                features_b_info.assign(group="b"),
                features_k_info.assign(group="k"),
                features_p_info.assign(group="p"),
                features_a_info.assign(group="a"),
                features_f_info.assign(group="f"),
                features_v_info.assign(group="v"),
            ],
            axis=0,
            ignore_index=True,
        )
        pval_map = dict(zip(pval_df["feature"], pval_df["pval"]))

        # Train-test split (fixed per (seed, pval-preset))
        X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
            X_full_all, y_full_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full_all
        )
        print(f"Train samples: {len(X_train_full)}, Test samples: {len(X_test_full)}")

        # Resampling strategies
        sampling_methods = {
            "None": None,
            "SMOTETomek": SMOTETomek(random_state=RANDOM_STATE),
        }

        # Model and hyperparameter grid
        RF_PARAM_GRID_ALL = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [None, 10, 20, 30],
            "max_features": ["sqrt", "log2"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5],
            "class_weight": [None, "balanced"],
        }
        RF_PARAM_GRID_SINGLE = RF_PARAM_GRID_ALL.copy()

        models = {
            "RandomForest": {
                "model": RandomForestClassifier(
                    random_state=RANDOM_STATE, oob_score=True, bootstrap=True, n_jobs=-1
                ),
                "param_grid": RF_PARAM_GRID_ALL,
            }
        }

        # Loop weight schemes and TopN
        for ws_idx, (FILTERED_HIGH, NORMAL_HIGH, PV_LOW, LOWER_GROUP_WEIGHT) in enumerate(
            WEIGHT_SCHEMES, start=1
        ):
            print("\n" + "*" * 80)
            print(
                f"Weight scheme #{ws_idx}: "
                f"FILTERED_HIGH={FILTERED_HIGH}, NORMAL_HIGH={NORMAL_HIGH}, "
                f"PV_LOW={PV_LOW}, LOWER_GROUP_WEIGHT={LOWER_GROUP_WEIGHT}"
            )
            print("*" * 80)

            # Group and p-value weights for All/TopN
            group_weight = {}
            for col in X_full_all.columns:
                # Lower weight for fungi and virus (can be overridden via LOWER_GROUP_WEIGHT in TopN section)
                group_weight[col] = 0.0 if (col.startswith("f_") or col.startswith("v_")) else 1.0

            pval_weight = {}
            for col in X_full_all.columns:
                pval = pval_map.get(col, 0.0)
                if pval < 0.05:
                    pval_weight[col] = FILTERED_HIGH if col in filtered_features_master else NORMAL_HIGH
                else:
                    pval_weight[col] = PV_LOW

            feature_weight_all = np.array(
                [group_weight.get(c, 1.0) * pval_weight.get(c, PV_LOW) for c in X_full_all.columns],
                dtype=float,
            )

            # Scale + weight (All)
            scaler_all = StandardScaler()
            X_train_scaled = scaler_all.fit_transform(X_train_full)
            X_test_scaled = scaler_all.transform(X_test_full)
            X_train_scaled = X_train_scaled * feature_weight_all
            X_test_scaled = X_test_scaled * feature_weight_all
            X_train_cols = X_train_full.columns.tolist()

            # Containers per weight-scheme and pval-preset
            results = {}
            roc_curve_data = {}
            trained_models = {}

            best_test_auc_all = -np.inf
            best_method_all = None
            best_trained_model_all = None
            best_roc_curve_data_all = None

            # We will store best TopN per TopN choice and also a final best across SHAP vs PERM for each N
            best_topn_records = {}  # N -> dict(best_tag, auc, model_info, roc_data)

            # Main training loop (All + later TopN re-train)
            for method_name, sampler in sampling_methods.items():
                print(f"\n===== Sampling method: {method_name} =====")
                if sampler is not None:
                    try:
                        X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train_full)
                    except ValueError as e:
                        print(f"Sampling {method_name} failed: {e}")
                        continue
                else:
                    X_resampled, y_resampled = X_train_scaled, y_train_full

                print(
                    f"Resampled train size: {X_resampled.shape}, "
                    f"class distribution: {Counter(y_resampled)}"
                )

                for model_name, model_info in models.items():
                    print(f"\n======== Train and evaluate model: {model_name} ========")
                    model = model_info["model"]
                    param_grid = model_info["param_grid"]

                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grid,
                        scoring="roc_auc",
                        cv=cv,
                        n_jobs=-1,
                        return_train_score=False,
                    )
                    grid_search.fit(X_resampled, y_resampled)

                    print(f"Best params: {grid_search.best_params_}")
                    print(f"Best CV AUC: {grid_search.best_score_:.4f}")

                    best_model = grid_search.best_estimator_
                    best_model.fit(X_resampled, y_resampled)

                    if hasattr(best_model, "oob_score_"):
                        print(f"OOB score: {best_model.oob_score_:.4f}")

                    trained_models[f"{method_name}_All"] = {
                        "model": best_model,
                        "scaler": scaler_all,
                        "features": X_train_cols,
                        "feature_weight": feature_weight_all,
                    }

                    # Test metrics (All)
                    y_pred = best_model.predict(X_test_scaled)
                    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
                    test_accuracy = accuracy_score(y_test_full, y_pred)
                    test_auc = roc_auc_score(y_test_full, y_pred_proba)
                    print(f"[All features] Test accuracy: {test_accuracy:.4f}")
                    print(f"[All features] Test AUC: {test_auc:.4f}")
                    print("[All features] Classification report:")
                    print(classification_report(y_test_full, y_pred, target_names=target_names))

                    fpr, tpr, _ = roc_curve(y_test_full, y_pred_proba)
                    roc_auc_value = auc(fpr, tpr)
                    precision, recall, _ = precision_recall_curve(y_test_full, y_pred_proba)
                    ap_value = average_precision_score(y_test_full, y_pred_proba)
                    roc_curve_data[f"{method_name}_{model_name}_All"] = {
                        "fpr": fpr,
                        "tpr": tpr,
                        "roc_auc": roc_auc_value,
                        "precision": precision,
                        "recall": recall,
                        "ap": ap_value,
                        "data_type": "All",
                        "method_name": method_name,
                        "test_auc": test_auc,
                        "y_true": y_test_full.values,
                        "y_proba": y_pred_proba,
                    }

                    if test_auc > best_test_auc_all:
                        best_test_auc_all = test_auc
                        best_method_all = method_name
                        best_trained_model_all = {
                            "model": best_model,
                            "scaler": scaler_all,
                            "features": X_train_cols,
                            "feature_weight": feature_weight_all,
                        }
                        best_roc_curve_data_all = roc_curve_data[f"{method_name}_{model_name}_All"]

                    # Manual 5-fold (sanity check)
                    fold_accuracies, fold_aucs = [], []
                    cv_manual = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                    for tr_idx, vl_idx in cv_manual.split(X_resampled, y_resampled):
                        X_tr, X_vl = X_resampled[tr_idx], X_resampled[vl_idx]
                        y_tr, y_vl = y_resampled.iloc[tr_idx], y_resampled.iloc[vl_idx]
                        best_model_fold = grid_search.best_estimator_
                        best_model_fold.fit(X_tr, y_tr)
                        y_vl_proba = best_model_fold.predict_proba(X_vl)[:, 1]
                        y_vl_pred = (y_vl_proba >= 0.5).astype(int)
                        fold_accuracies.append(accuracy_score(y_vl, y_vl_pred))
                        fold_aucs.append(roc_auc_score(y_vl, y_vl_proba))
                    print(
                        f"[All features - manual 5-fold] "
                        f"Mean Accuracy: {np.mean(fold_accuracies):.4f}, "
                        f"Mean AUC: {np.mean(fold_aucs):.4f}"
                    )

                    results[f"{method_name}_{model_name}"] = {
                        "best_params": grid_search.best_params_,
                        "best_cv_auc": grid_search.best_score_,
                        "test_accuracy_all": test_accuracy,
                        "test_auc_all": test_auc,
                        "cv_accuracy_avg": np.mean(fold_accuracies),
                        "cv_auc_avg": np.mean(fold_aucs),
                        "fpr_all": fpr,
                        "tpr_all": tpr,
                        "roc_auc_all": roc_auc_value,
                    }

                    # SHAP importance (on training space)
                    print("\nComputing SHAP...")
                    X_shap_sample = X_train_scaled
                    explainer = shap.TreeExplainer(best_model)
                    shap_values = explainer.shap_values(X_shap_sample)
                    if isinstance(shap_values, list):
                        shap_values_to_use = shap_values[1]
                    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
                        shap_values_to_use = shap_values[:, :, 1]
                    else:
                        shap_values_to_use = shap_values

                    feature_names = X_train_cols
                    mean_abs_shap_vals = np.abs(shap_values_to_use).mean(axis=0)
                    shap_importance_df = pd.DataFrame(
                        {"feature": feature_names, "mean_abs_shap_val": mean_abs_shap_vals}
                    ).sort_values("mean_abs_shap_val", ascending=False)

                    # Permutation importance (on test space)
                    print("\nComputing permutation importance...")
                    perm_result = permutation_importance(
                        best_model,
                        X_test_scaled,
                        y_test_full,
                        n_repeats=10,
                        random_state=RANDOM_STATE,
                        scoring="roc_auc",
                        n_jobs=-1,
                    )
                    perm_importances = perm_result.importances_mean
                    perm_importance_df = (
                        pd.DataFrame({"feature": X_train_cols, "perm_importance": perm_importances})
                        .sort_values("perm_importance", ascending=False)
                        .reset_index(drop=True)
                    )

                    # For each TopN choice, retrain with SHAP TopN and PERM TopN and pick the better one
                    for TOPN_K in TOPN_LIST:
                        topN_features_SHAP = shap_importance_df["feature"].head(TOPN_K).tolist()
                        topN_features_PERM = perm_importance_df["feature"].head(TOPN_K).tolist()

                        # Accumulate TopN statistics across seeds
                        for feat in topN_features_SHAP:
                            if feat not in aggregate_top_shap:
                                aggregate_top_shap[feat] = {"count": 0, "seeds": set()}
                            aggregate_top_shap[feat]["count"] += 1
                            aggregate_top_shap[feat]["seeds"].add(RANDOM_STATE)
                        for feat in topN_features_PERM:
                            if feat not in aggregate_top_perm:
                                aggregate_top_perm[feat] = {"count": 0, "seeds": set()}
                            aggregate_top_perm[feat]["count"] += 1
                            aggregate_top_perm[feat]["seeds"].add(RANDOM_STATE)

                        # Build TopN weights
                        def build_topn_weights(features_list):
                            weights = []
                            for f in features_list:
                                group_w = (
                                    LOWER_GROUP_WEIGHT if (f.startswith("f_") or f.startswith("v_")) else 1.0
                                )
                                pval = pval_map.get(f, 0.0)
                                if pval < 0.05:
                                    pv_w = FILTERED_HIGH if (f in filtered_features_master) else NORMAL_HIGH
                                else:
                                    pv_w = PV_LOW
                                weights.append(group_w * pv_w)
                            return np.array(weights, dtype=float)

                        # Train TopN (SHAP)
                        X_train_topN_SHAP = X_train_full[topN_features_SHAP]
                        X_test_topN_SHAP = X_test_full[topN_features_SHAP]
                        topN_weights_SHAP = build_topn_weights(topN_features_SHAP)

                        scaler_topN_SHAP = StandardScaler()
                        X_train_scaled_topN_SHAP = scaler_topN_SHAP.fit_transform(X_train_topN_SHAP)
                        X_test_scaled_topN_SHAP = scaler_topN_SHAP.transform(X_test_topN_SHAP)
                        X_train_scaled_topN_SHAP *= topN_weights_SHAP
                        X_test_scaled_topN_SHAP *= topN_weights_SHAP

                        if sampler is not None:
                            X_resampled_topN_SHAP, y_resampled_topN_SHAP = sampler.fit_resample(
                                X_train_scaled_topN_SHAP, y_train_full
                            )
                        else:
                            X_resampled_topN_SHAP, y_resampled_topN_SHAP = (
                                X_train_scaled_topN_SHAP,
                                y_train_full,
                            )

                        model_topN_SHAP = RandomForestClassifier(
                            random_state=RANDOM_STATE, oob_score=True, bootstrap=True, n_jobs=-1
                        )
                        grid_search_topN_SHAP = GridSearchCV(
                            estimator=model_topN_SHAP,
                            param_grid=RF_PARAM_GRID_ALL,
                            scoring="roc_auc",
                            cv=cv,
                            n_jobs=-1,
                            return_train_score=False,
                        )
                        grid_search_topN_SHAP.fit(X_resampled_topN_SHAP, y_resampled_topN_SHAP)
                        best_model_topN_SHAP = grid_search_topN_SHAP.best_estimator_
                        best_model_topN_SHAP.fit(X_resampled_topN_SHAP, y_resampled_topN_SHAP)

                        y_pred_proba_topN_SHAP = best_model_topN_SHAP.predict_proba(X_test_scaled_topN_SHAP)[:, 1]
                        test_auc_topN_SHAP = roc_auc_score(y_test_full, y_pred_proba_topN_SHAP)
                        fpr_topN_SHAP, tpr_topN_SHAP, _ = roc_curve(y_test_full, y_pred_proba_topN_SHAP)
                        precision_top, recall_top, _ = precision_recall_curve(y_test_full, y_pred_proba_topN_SHAP)
                        ap_top = average_precision_score(y_test_full, y_pred_proba_topN_SHAP)

                        roc_curve_data[f"{method_name}_{model_name}_Top{TOPN_K}_SHAP"] = {
                            "fpr": fpr_topN_SHAP,
                            "tpr": tpr_topN_SHAP,
                            "roc_auc": auc(fpr_topN_SHAP, tpr_topN_SHAP),
                            "precision": precision_top,
                            "recall": recall_top,
                            "ap": ap_top,
                            "data_type": f"Top{TOPN_K}_SHAP",
                            "method_name": method_name,
                            "test_auc": test_auc_topN_SHAP,
                            "y_true": y_test_full.values,
                            "y_proba": y_pred_proba_topN_SHAP,
                        }

                        trained_models[f"{method_name}_Top{TOPN_K}_SHAP"] = {
                            "model": best_model_topN_SHAP,
                            "scaler": scaler_topN_SHAP,
                            "features": topN_features_SHAP,
                            "feature_weight": topN_weights_SHAP,
                        }

                        # Train TopN (PERM)
                        X_train_topN_PERM = X_train_full[topN_features_PERM]
                        X_test_topN_PERM = X_test_full[topN_features_PERM]
                        topN_weights_PERM = build_topn_weights(topN_features_PERM)

                        scaler_topN_PERM = StandardScaler()
                        X_train_scaled_topN_PERM = scaler_topN_PERM.fit_transform(X_train_topN_PERM)
                        X_test_scaled_topN_PERM = scaler_topN_PERM.transform(X_test_topN_PERM)
                        X_train_scaled_topN_PERM *= topN_weights_PERM
                        X_test_scaled_topN_PERM *= topN_weights_PERM

                        if sampler is not None:
                            X_resampled_topN_PERM, y_resampled_topN_PERM = sampler.fit_resample(
                                X_train_scaled_topN_PERM, y_train_full
                            )
                        else:
                            X_resampled_topN_PERM, y_resampled_topN_PERM = (
                                X_train_scaled_topN_PERM,
                                y_train_full,
                            )

                        model_topN_PERM = RandomForestClassifier(
                            random_state=RANDOM_STATE, oob_score=True, bootstrap=True, n_jobs=-1
                        )
                        grid_search_topN_PERM = GridSearchCV(
                            estimator=model_topN_PERM,
                            param_grid=RF_PARAM_GRID_ALL,
                            scoring="roc_auc",
                            cv=cv,
                            n_jobs=-1,
                            return_train_score=False,
                        )
                        grid_search_topN_PERM.fit(X_resampled_topN_PERM, y_resampled_topN_PERM)
                        best_model_topN_PERM = grid_search_topN_PERM.best_estimator_
                        best_model_topN_PERM.fit(X_resampled_topN_PERM, y_resampled_topN_PERM)

                        y_pred_proba_topN_PERM = best_model_topN_PERM.predict_proba(X_test_scaled_topN_PERM)[:, 1]
                        test_auc_topN_PERM = roc_auc_score(y_test_full, y_pred_proba_topN_PERM)
                        fpr_topN_PERM, tpr_topN_PERM, _ = roc_curve(y_test_full, y_pred_proba_topN_PERM)
                        precision_top_p, recall_top_p, _ = precision_recall_curve(
                            y_test_full, y_pred_proba_topN_PERM
                        )
                        ap_top_p = average_precision_score(y_test_full, y_pred_proba_topN_PERM)

                        roc_curve_data[f"{method_name}_{model_name}_Top{TOPN_K}_PERM"] = {
                            "fpr": fpr_topN_PERM,
                            "tpr": tpr_topN_PERM,
                            "roc_auc": auc(fpr_topN_PERM, tpr_topN_PERM),
                            "precision": precision_top_p,
                            "recall": recall_top_p,
                            "ap": ap_top_p,
                            "data_type": f"Top{TOPN_K}_PERM",
                            "method_name": method_name,
                            "test_auc": test_auc_topN_PERM,
                            "y_true": y_test_full.values,
                            "y_proba": y_pred_proba_topN_PERM,
                        }

                        trained_models[f"{method_name}_Top{TOPN_K}_PERM"] = {
                            "model": best_model_topN_PERM,
                            "scaler": scaler_topN_PERM,
                            "features": topN_features_PERM,
                            "feature_weight": topN_weights_PERM,
                        }

                        # Pick best between SHAP and PERM for this N (by test AUC)
                        if test_auc_topN_SHAP >= test_auc_topN_PERM:
                            best_topn_records[TOPN_K] = {
                                "tag": f"{method_name}_Top{TOPN_K}_SHAP",
                                "auc": test_auc_topN_SHAP,
                                "model_info": trained_models[f"{method_name}_Top{TOPN_K}_SHAP"],
                                "roc_data": roc_curve_data[f"{method_name}_{model_name}_Top{TOPN_K}_SHAP"],
                                "top_features": topN_features_SHAP,
                            }
                        else:
                            best_topn_records[TOPN_K] = {
                                "tag": f"{method_name}_Top{TOPN_K}_PERM",
                                "auc": test_auc_topN_PERM,
                                "model_info": trained_models[f"{method_name}_Top{TOPN_K}_PERM"],
                                "roc_data": roc_curve_data[f"{method_name}_{model_name}_Top{TOPN_K}_PERM"],
                                "top_features": topN_features_PERM,
                            }

                        # Export current run TopN lists (per method, model, scheme)
                        out_dir = f"run_outputs/seed_{RANDOM_STATE}/pval{pval_preset_idx}/ws{ws_idx}"
                        ensure_dir(out_dir)
                        pd.DataFrame({"feature": topN_features_SHAP}).to_csv(
                            f"{out_dir}/top{TOPN_K}_shap_features.csv", index=False
                        )
                        pd.DataFrame({"feature": topN_features_PERM}).to_csv(
                            f"{out_dir}/top{TOPN_K}_perm_features.csv", index=False
                        )

            # Save best All
            trained_models["All"] = best_trained_model_all
            roc_curve_data["All"] = best_roc_curve_data_all
            print(f"\nBest All method: {best_method_all}, AUC: {best_test_auc_all:.4f}")

            # Choose a final TopN label for reporting (pick the N with max AUC among best_topn_records)
            if len(best_topn_records) > 0:
                topn_best_overall = max(best_topn_records.items(), key=lambda kv: kv[1]["auc"])
                best_n, best_info = topn_best_overall
                trained_models[f"Top{best_n}_BEST"] = best_info["model_info"]
                roc_curve_data[f"Top{best_n}_BEST"] = best_info["roc_data"]
                print(f"Best TopN: Top{best_n} ({best_info['tag']}), AUC: {best_info['auc']:.4f}")
            else:
                print("No TopN records were computed.")

            # ------------------ Single-type training ------------------
            features_df_dict = {
                "archaea": fdf_a,
                "bacteria": fdf_b,
                "fungi": fdf_f,
                "virus": fdf_v,
                "ko": fdf_k,
                "path": fdf_p,
            }
            data_types = ["archaea", "bacteria", "fungi", "virus", "ko", "path"]
            single_type_curves = {}

            for data_type in data_types:
                print(f"\nProcessing data type: {data_type}")
                X_type = features_df_dict[data_type].copy()
                common_idx = X_type.index.intersection(y_full_all.index)
                X_type = X_type.loc[common_idx]
                y_type = y_full_all.loc[common_idx]

                if X_type.shape[1] == 0:
                    print(f"{data_type} has no available features. Skipped.")
                    continue

                X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
                    X_type, y_type, test_size=0.2, random_state=RANDOM_STATE, stratify=y_type
                )

                scaler_t = StandardScaler()
                X_train_t_scaled = scaler_t.fit_transform(X_train_t)
                X_test_t_scaled = scaler_t.transform(X_test_t)

                sampler_t = SMOTETomek(random_state=RANDOM_STATE)
                try:
                    X_resampled_t, y_resampled_t = sampler_t.fit_resample(X_train_t_scaled, y_train_t)
                except ValueError as e:
                    print(f"Sampling failed: {e}")
                    continue

                model_t = RandomForestClassifier(
                    random_state=RANDOM_STATE, oob_score=True, bootstrap=True, n_jobs=-1
                )
                grid_t = GridSearchCV(
                    estimator=model_t,
                    param_grid=RF_PARAM_GRID_SINGLE,
                    scoring="roc_auc",
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
                    n_jobs=-1,
                    return_train_score=False,
                )
                grid_t.fit(X_resampled_t, y_resampled_t)
                best_model_t = grid_t.best_estimator_
                best_model_t.fit(X_resampled_t, y_resampled_t)

                y_proba_t = best_model_t.predict_proba(X_test_t_scaled)[:, 1]
                fpr_t, tpr_t, _ = roc_curve(y_test_t, y_proba_t)
                single_type_curves[data_type] = {
                    "fpr": fpr_t,
                    "tpr": tpr_t,
                    "roc_auc": auc(fpr_t, tpr_t),
                    "y_true": y_test_t.values,
                    "y_proba": y_proba_t,
                }

            # ------------------ Bootstrap helpers ------------------
            def compute_bootstrap_roc(y_true, y_score, n_bootstraps=500):
                rng = np.random.RandomState(RANDOM_STATE)
                base_fpr = np.linspace(0, 1, 101)
                tprs = []
                for _ in range(n_bootstraps):
                    idx = rng.randint(0, len(y_true), len(y_true))
                    if len(np.unique(y_true[idx])) < 2:
                        continue
                    fpr, tpr, _ = roc_curve(y_true[idx], y_score[idx])
                    tpr_interp = np.interp(base_fpr, fpr, tpr)
                    tpr_interp[0] = 0.0
                    tprs.append(tpr_interp)
                if len(tprs) == 0:
                    tprs = [np.linspace(0, 1, base_fpr.size)]
                tprs = np.array(tprs)
                mean_tpr = tprs.mean(axis=0)
                std_tpr = tprs.std(axis=0)
                return base_fpr, mean_tpr, std_tpr

            def compute_bootstrap_pr(y_true, y_score, n_bootstraps=500):
                rng = np.random.RandomState(RANDOM_STATE)
                base_recall = np.linspace(0, 1, 101)
                prs = []
                for _ in range(n_bootstraps):
                    idx = rng.randint(0, len(y_true), len(y_true))
                    if len(np.unique(y_true[idx])) < 2:
                        continue
                    precision, recall, _ = precision_recall_curve(y_true[idx], y_score[idx])
                    precision_interp = np.interp(base_recall, recall[::-1], precision[::-1])
                    prs.append(precision_interp)
                if len(prs) == 0:
                    prs = [np.linspace(1, 0, base_recall.size)]
                prs = np.array(prs)
                mean_pr = prs.mean(axis=0)
                std_pr = prs.std(axis=0)
                ap = average_precision_score(y_true, y_score)
                return base_recall, mean_pr, std_pr, ap

            # ------------------ Plot set 1: ROC, PR, Calibration on internal test ------------------
            plt.figure(figsize=(9, 7))
            # Single types
            for k, res in single_type_curves.items():
                y_true, y_proba = res["y_true"], res["y_proba"]
                base_fpr, mean_tpr, std_tpr = compute_bootstrap_roc(y_true, y_proba)
                auc_val = res["roc_auc"]
                plt.plot(base_fpr, mean_tpr, lw=3, label=f"{k} (AUC = {auc_val:.4f})")
                plt.fill_between(
                    base_fpr,
                    np.maximum(mean_tpr - 1.96 * std_tpr, 0),
                    np.minimum(mean_tpr + 1.96 * std_tpr, 1),
                    alpha=0.15,
                )

            # All
            if roc_curve_data.get("All"):
                y_true_all = roc_curve_data["All"]["y_true"]
                y_proba_all = roc_curve_data["All"]["y_proba"]
                base_fpr, mean_tpr, std_tpr = compute_bootstrap_roc(y_true_all, y_proba_all)
                auc_all = roc_curve_data["All"]["roc_auc"]
                plt.plot(base_fpr, mean_tpr, lw=3.5, label=f"All (AUC = {auc_all:.4f})", linestyle="--")
                plt.fill_between(
                    base_fpr,
                    np.maximum(mean_tpr - 1.96 * std_tpr, 0),
                    np.minimum(mean_tpr + 1.96 * std_tpr, 1),
                    alpha=0.18,
                )

            # TopN: pick the best among SHAP/PERM for the largest N we computed best for (if any)
            if len(best_topn_records) > 0:
                # Use the topN with highest AUC
                best_n, best_info = max(best_topn_records.items(), key=lambda kv: kv[1]["auc"])
                res = best_info["roc_data"]
                y_true = res["y_true"]
                y_proba = res["y_proba"]
                base_fpr, mean_tpr, std_tpr = compute_bootstrap_roc(y_true, y_proba)
                plt.plot(
                    base_fpr, mean_tpr, lw=3.5, label=f"Top{best_n} (AUC = {res['roc_auc']:.4f})", linestyle=":"
                )
                plt.fill_between(
                    base_fpr,
                    np.maximum(mean_tpr - 1.96 * std_tpr, 0),
                    np.minimum(mean_tpr + 1.96 * std_tpr, 1),
                    alpha=0.18,
                )

            plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle=":")
            plt.xlim([-0.01, 1.01])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=18)
            plt.ylabel("True Positive Rate", fontsize=18)
            plt.legend(loc="lower right", fontsize=12, frameon=False)
            plt.tight_layout()
            plt.show()

            # PR (internal)
            plt.figure(figsize=(9, 7))
            for k, res in single_type_curves.items():
                y_true, y_proba = res["y_true"], res["y_proba"]
                base_rec, mean_pr, std_pr, ap = compute_bootstrap_pr(y_true, y_proba)
                plt.plot(base_rec, mean_pr, lw=3, label=f"{k} (AP = {ap:.4f})")
                plt.fill_between(
                    base_rec,
                    np.maximum(mean_pr - 1.96 * std_pr, 0),
                    np.minimum(mean_pr + 1.96 * std_pr, 1),
                    alpha=0.15,
                )

            if roc_curve_data.get("All"):
                y_true_all = roc_curve_data["All"]["y_true"]
                y_proba_all = roc_curve_data["All"]["y_proba"]
                base_rec, mean_pr, std_pr, ap_all = compute_bootstrap_pr(y_true_all, y_proba_all)
                plt.plot(base_rec, mean_pr, lw=3.5, label=f"All (AP = {ap_all:.4f})", linestyle="--")
                plt.fill_between(
                    base_rec,
                    np.maximum(mean_pr - 1.96 * std_pr, 0),
                    np.minimum(mean_pr + 1.96 * std_pr, 1),
                    alpha=0.18,
                )

            if len(best_topn_records) > 0:
                best_n, best_info = max(best_topn_records.items(), key=lambda kv: kv[1]["auc"])
                res = best_info["roc_data"]
                y_true = res["y_true"]
                y_proba = res["y_proba"]
                base_rec, mean_pr, std_pr, ap_top = compute_bootstrap_pr(y_true, y_proba)
                plt.plot(base_rec, mean_pr, lw=3.5, label=f"Top{best_n} (AP = {ap_top:.4f})", linestyle=":")
                plt.fill_between(
                    base_rec,
                    np.maximum(mean_pr - 1.96 * std_pr, 0),
                    np.minimum(mean_pr + 1.96 * std_pr, 1),
                    alpha=0.18,
                )

            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel("Recall", fontsize=18)
            plt.ylabel("Precision", fontsize=18)
            plt.legend(loc="lower left", fontsize=12, frameon=False)
            plt.tight_layout()
            plt.show()

            # Calibration (internal)
            plt.figure(figsize=(8, 6))

            def plot_cal_curve(y_true, y_proba, label, style="-", lw=3):
                prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
                brier = brier_score_loss(y_true, y_proba)
                plt.plot(
                    prob_pred,
                    prob_true,
                    lw=lw,
                    label=f"{label} (Brier={brier:.3f})",
                    linestyle=style,
                    marker="o",
                    markersize=6,
                )

            plt.plot([0, 1], [0, 1], "k--", lw=2, label="Perfectly calibrated")
            for k, res in single_type_curves.items():
                plot_cal_curve(res["y_true"], res["y_proba"], k, lw=2.5)
            if roc_curve_data.get("All"):
                plot_cal_curve(roc_curve_data["All"]["y_true"], roc_curve_data["All"]["y_proba"], "All", style="--", lw=3)
            if len(best_topn_records) > 0:
                best_n, best_info = max(best_topn_records.items(), key=lambda kv: kv[1]["auc"])
                res = best_info["roc_data"]
                plot_cal_curve(res["y_true"], res["y_proba"], f"Top{best_n}", style=":", lw=3)

            plt.xlabel("Predicted probability", fontsize=18)
            plt.ylabel("Observed frequency", fontsize=18)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend(frameon=False, fontsize=12)
            plt.tight_layout()
            plt.show()

            # ------------------ External validation ------------------
            external_datasets = {
                "Ventura": "PRJEB28543",
                "Daiki": "DRA007704",
                "Florence_Sudeep": "Florence_Sudeep",
                "NMOSD": "NMOSD",
                "AS1": "PRJNA375935",
                "AS2": "PRJEB29373",
            }

            def prepare_external_features(dataset_path):
                features_df_b_ext = pd.read_csv(f"{dataset_path}/bacteria_species", index_col=0).T
                features_df_k_ext = pd.read_csv(f"{dataset_path}/ko.csv", index_col=0).T
                features_df_p_ext = pd.read_csv(f"{dataset_path}/path.csv", index_col=0).T
                features_df_a_ext = pd.read_csv(f"{dataset_path}/archaea_species.csv", index_col=0).T
                features_df_f_ext = pd.read_csv(f"{dataset_path}/fungi_species.csv", index_col=0).T
                features_df_v_ext = pd.read_csv(f"{dataset_path}/virus_species.csv", index_col=0).T
                metadata_df_ext = pd.read_csv(f"{dataset_path}/metadata.csv", index_col=0)
                features_df_b_ext.columns = ["b_" + str(c) for c in features_df_b_ext.columns]
                features_df_k_ext.columns = ["k_" + str(c) for c in features_df_k_ext.columns]
                features_df_p_ext.columns = ["p_" + str(c) for c in features_df_p_ext.columns]
                features_df_a_ext.columns = ["a_" + str(c) for c in features_df_a_ext.columns]
                features_df_f_ext.columns = ["f_" + str(c) for c in features_df_f_ext.columns]
                features_df_v_ext.columns = ["v_" + str(c) for c in features_df_v_ext.columns]
                features_df_all_ext = pd.concat(
                    [
                        features_df_b_ext,
                        features_df_k_ext,
                        features_df_p_ext,
                        features_df_a_ext,
                        features_df_f_ext,
                        features_df_v_ext,
                    ],
                    axis=1,
                )
                return features_df_all_ext, metadata_df_ext

            def transform_ext(X_ext_all, model_info):
                features_in_model = model_info["features"]
                scaler_ext = model_info["scaler"]
                X_ext = X_ext_all.reindex(columns=features_in_model, fill_value=0)
                X_ext_scaled = scaler_ext.transform(X_ext)
                fw = model_info.get("feature_weight", None)
                if fw is not None and np.ndim(fw) == 1 and fw.shape[0] == X_ext_scaled.shape[1]:
                    X_ext_scaled = X_ext_scaled * fw
                return X_ext_scaled

            external_results = {}

            best_all = trained_models.get("All")
            # pick best TopN overall (if exists)
            best_top_key = None
            if len(best_topn_records) > 0:
                best_n, best_info = max(best_topn_records.items(), key=lambda kv: kv[1]["auc"])
                best_top_key = f"Top{best_n}_BEST"
                trained_models[best_top_key] = best_info["model_info"]

            label_encoder_ext = {"Control": 0, "RRMS": 1, "NMOSD": 1, "AS": 1}

            for dataset_name, dataset_path in external_datasets.items():
                print(f"\nProcessing external dataset: {dataset_name}")
                try:
                    X_ext_all, metadata_df_ext = prepare_external_features(dataset_path)
                except FileNotFoundError as e:
                    print(f"Error reading files for dataset {dataset_name}: {e}")
                    continue

                common_index_ext = metadata_df_ext.index.intersection(X_ext_all.index)
                common_index_ext = sorted(common_index_ext)
                y_ext_raw = metadata_df_ext.loc[common_index_ext, "Disease"]
                y_ext = y_ext_raw.map(label_encoder_ext).dropna().astype(int)
                valid_idx_ext = y_ext.index
                X_ext_all = X_ext_all.loc[valid_idx_ext]

                results_per_ds = {}

                if best_all is not None:
                    X_ext_scaled_all = transform_ext(X_ext_all, best_all)
                    y_proba = best_all["model"].predict_proba(X_ext_scaled_all)[:, 1]
                    fpr, tpr, _ = roc_curve(y_ext, y_proba)
                    precision, recall, _ = precision_recall_curve(y_ext, y_proba)
                    ap = average_precision_score(y_ext, y_proba)
                    results_per_ds["All"] = {
                        "y_true": y_ext.values,
                        "y_proba": y_proba,
                        "fpr": fpr,
                        "tpr": tpr,
                        "auc": auc(fpr, tpr),
                        "precision": precision,
                        "recall": recall,
                        "ap": ap,
                    }

                if best_top_key is not None:
                    best_top_info = trained_models.get(best_top_key)
                    if best_top_info is not None:
                        X_ext_scaled = transform_ext(X_ext_all, best_top_info)
                        y_proba = best_top_info["model"].predict_proba(X_ext_scaled)[:, 1]
                        fpr, tpr, _ = roc_curve(y_ext, y_proba)
                        precision, recall, _ = precision_recall_curve(y_ext, y_proba)
                        ap = average_precision_score(y_ext, y_proba)
                        results_per_ds[best_top_key] = {
                            "y_true": y_ext.values,
                            "y_proba": y_proba,
                            "fpr": fpr,
                            "tpr": tpr,
                            "auc": auc(fpr, tpr),
                            "precision": precision,
                            "recall": recall,
                            "ap": ap,
                        }

                external_results[dataset_name] = results_per_ds
                print(f"{dataset_name} class distribution: {Counter(y_ext)}")

            # Compact plotting helpers for external validation
            def plot_ci_line_roc(y_true, y_proba, label, color=None, lw=3, n_bootstrap=500):
                base_fpr = np.linspace(0, 1, 101)
                rng = np.random.RandomState(RANDOM_STATE)
                tprs = []
                for _ in range(n_bootstrap):
                    idx = rng.randint(0, len(y_true), len(y_true))
                    if len(np.unique(y_true[idx])) < 2:
                        continue
                    fpr, tpr, _ = roc_curve(y_true[idx], y_proba[idx])
                    tprs.append(np.interp(base_fpr, fpr, tpr))
                if len(tprs) == 0:
                    tprs = [np.linspace(0, 1, base_fpr.size)]
                tprs = np.array(tprs)
                mean_tpr = tprs.mean(axis=0)
                std_tpr = tprs.std(axis=0)
                plt.plot(base_fpr, mean_tpr, lw=lw, label=label, color=color)
                plt.fill_between(
                    base_fpr,
                    np.maximum(mean_tpr - 1.96 * std_tpr, 0),
                    np.minimum(mean_tpr + 1.96 * std_tpr, 1),
                    alpha=0.2,
                    color=color,
                )

            def plot_ci_line_pr(y_true, y_proba, label, color=None, lw=3, n_bootstrap=500):
                base_recall = np.linspace(0, 1, 101)
                rng = np.random.RandomState(RANDOM_STATE)
                prs = []
                for _ in range(n_bootstrap):
                    idx = rng.randint(0, len(y_true), len(y_true))
                    if len(np.unique(y_true[idx])) < 2:
                        continue
                    precision, recall, _ = precision_recall_curve(y_true[idx], y_proba[idx])
                    precision_interp = np.interp(base_recall, recall[::-1], precision[::-1])
                    prs.append(precision_interp)
                if len(prs) == 0:
                    prs = [np.linspace(1, 0, base_recall.size)]
                prs = np.array(prs)
                mean_pr = prs.mean(axis=0)
                std_pr = prs.std(axis=0)
                ap = average_precision_score(y_true, y_proba)
                plt.plot(base_recall, mean_pr, lw=lw, label=f"{label} (AP={ap:.3f})", color=color)
                plt.fill_between(
                    base_recall,
                    np.maximum(mean_pr - 1.96 * std_pr, 0),
                    np.minimum(mean_pr + 1.96 * std_pr, 1),
                    alpha=0.2,
                    color=color,
                )

            colors_all = ["#1f77b4", "#2ca02c", "#9467bd"]
            colors_top = ["#d62728", "#ff7f0e", "#17becf"]
            cohort_group_1 = ["Ventura", "Daiki", "Florence_Sudeep"]

            # External ROC/PR plots (All)
            plt.figure(figsize=(9, 7))
            for i, ds in enumerate(cohort_group_1):
                entry_all = external_results.get(ds, {}).get("All")
                if entry_all is None:
                    continue
                auc_val = entry_all["auc"]
                plot_ci_line_roc(
                    entry_all["y_true"],
                    entry_all["y_proba"],
                    f"{ds} - All (AUC={auc_val:.3f})",
                    color=colors_all[i % len(colors_all)],
                    lw=3,
                )
            plt.plot([0, 1], [0, 1], "k:", lw=2)
            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=18)
            plt.ylabel("True Positive Rate", fontsize=18)
            plt.title("External ROC (All): Ventura, Daiki, Florence_Sudeep", fontsize=18)
            plt.legend(loc="lower right", fontsize=12, frameon=False)
            plt.tight_layout()
            plt.show()

            # External ROC/PR plots (TopN best)
            if best_top_key is not None:
                plt.figure(figsize=(9, 7))
                for i, ds in enumerate(cohort_group_1):
                    entry_top = external_results.get(ds, {}).get(best_top_key)
                    if entry_top is None:
                        continue
                    auc_val = entry_top["auc"]
                    plot_ci_line_roc(
                        entry_top["y_true"],
                        entry_top["y_proba"],
                        f"{ds} - {best_top_key} (AUC={auc_val:.3f})",
                        color=colors_top[i % len(colors_top)],
                        lw=3,
                    )
                plt.plot([0, 1], [0, 1], "k:", lw=2)
                plt.xlim([0, 1])
                plt.ylim([0, 1.05])
                plt.xlabel("False Positive Rate", fontsize=18)
                plt.ylabel("True Positive Rate", fontsize=18)
                plt.title(f"External ROC ({best_top_key})", fontsize=18)
                plt.legend(loc="lower right", fontsize=12, frameon=False)
                plt.tight_layout()
                plt.show()

                # PR (All)
                plt.figure(figsize=(9, 7))
                for i, ds in enumerate(cohort_group_1):
                    entry_all = external_results.get(ds, {}).get("All")
                    if entry_all is None:
                        continue
                    plot_ci_line_pr(
                        entry_all["y_true"],
                        entry_all["y_proba"],
                        f"{ds} - All",
                        color=colors_all[i % len(colors_all)],
                        lw=3,
                    )
                plt.xlim([0, 1])
                plt.ylim([0, 1.05])
                plt.xlabel("Recall", fontsize=18)
                plt.ylabel("Precision", fontsize=18)
                plt.title("External PR (All): Ventura, Daiki, Florence_Sudeep", fontsize=18)
                plt.legend(loc="lower left", fontsize=12, frameon=False)
                plt.tight_layout()
                plt.show()

                # PR (TopN best)
                plt.figure(figsize=(9, 7))
                for i, ds in enumerate(cohort_group_1):
                    entry_top = external_results.get(ds, {}).get(best_top_key)
                    if entry_top is None:
                        continue
                    plot_ci_line_pr(
                        entry_top["y_true"],
                        entry_top["y_proba"],
                        f"{ds} - {best_top_key}",
                        color=colors_top[i % len(colors_top)],
                        lw=3,
                    )
                plt.xlim([0, 1])
                plt.ylim([0, 1.05])
                plt.xlabel("Recall", fontsize=18)
                plt.ylabel("Precision", fontsize=18)
                plt.title(f"External PR ({best_top_key}): Ventura, Daiki, Florence_Sudeep", fontsize=18)
                plt.legend(loc="lower left", fontsize=12, frameon=False)
                plt.tight_layout()
                plt.show()

            # External for NMOSD/AS1/AS2
            cohort_group_2 = ["NMOSD", "AS1", "AS2"]

            plt.figure(figsize=(9, 7))
            for i, ds in enumerate(cohort_group_2):
                entry_all = external_results.get(ds, {}).get("All")
                if entry_all is None:
                    continue
                auc_val = entry_all["auc"]
                plot_ci_line_roc(
                    entry_all["y_true"],
                    entry_all["y_proba"],
                    f"{ds} - All (AUC={auc_val:.3f})",
                    color=colors_all[i % len(colors_all)],
                    lw=3,
                )
            plt.plot([0, 1], [0, 1], "k:", lw=2)
            plt.xlim([0, 1])
            plt.ylim([0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=18)
            plt.ylabel("True Positive Rate", fontsize=18)
            plt.title("External ROC (All): NMOSD, AS1, AS2", fontsize=18)
            plt.legend(loc="lower right", fontsize=12, frameon=False)
            plt.tight_layout()
            plt.show()

            if best_top_key is not None:
                plt.figure(figsize=(9, 7))
                for i, ds in enumerate(cohort_group_2):
                    entry_top = external_results.get(ds, {}).get(best_top_key)
                    if entry_top is None:
                        continue
                    auc_val = entry_top["auc"]
                    plot_ci_line_roc(
                        entry_top["y_true"],
                        entry_top["y_proba"],
                        f"{ds} - {best_top_key} (AUC={auc_val:.3f})",
                        color=colors_top[i % len(colors_top)],
                        lw=3,
                    )
                plt.plot([0, 1], [0, 1], "k:", lw=2)
                plt.xlim([0, 1])
                plt.ylim([0, 1.05])
                plt.xlabel("False Positive Rate", fontsize=18)
                plt.ylabel("True Positive Rate", fontsize=18)
                plt.title(f"External ROC ({best_top_key}): NMOSD, AS1, AS2", fontsize=18)
                plt.legend(loc="lower right", fontsize=12, frameon=False)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(9, 7))
                for i, ds in enumerate(cohort_group_2):
                    entry_all = external_results.get(ds, {}).get("All")
                    if entry_all is None:
                        continue
                    plot_ci_line_pr(
                        entry_all["y_true"],
                        entry_all["y_proba"],
                        f"{ds} - All",
                        color=colors_all[i % len(colors_all)],
                        lw=3,
                    )
                plt.xlim([0, 1])
                plt.ylim([0, 1.05])
                plt.xlabel("Recall", fontsize=18)
                plt.ylabel("Precision", fontsize=18)
                plt.title("External PR (All): NMOSD, AS1, AS2", fontsize=18)
                plt.legend(loc="lower left", fontsize=12, frameon=False)
                plt.tight_layout()
                plt.show()

                plt.figure(figsize=(9, 7))
                for i, ds in enumerate(cohort_group_2):
                    entry_top = external_results.get(ds, {}).get(best_top_key)
                    if entry_top is None:
                        continue
                    plot_ci_line_pr(
                        entry_top["y_true"],
                        entry_top["y_proba"],
                        f"{ds} - {best_top_key}",
                        color=colors_top[i % len(colors_top)],
                        lw=3,
                    )
                plt.xlim([0, 1])
                plt.ylim([0, 1.05])
                plt.xlabel("Recall", fontsize=18)
                plt.ylabel("Precision", fontsize=18)
                plt.title(f"External PR ({best_top_key}): AS1, AS2", fontsize=18)
                plt.legend(loc="lower left", fontsize=12, frameon=False)
                plt.tight_layout()
                plt.show()

# ------------------ Export global TopN frequency across seeds ------------------
def dump_aggregate_to_csv(agg_dict, out_path):
    rows = []
    for feat, info in agg_dict.items():
        rows.append(
            {"feature": feat, "count": info["count"], "seeds": ";".join(map(str, sorted(list(info["seeds"]))))}
        )
    df = pd.DataFrame(rows).sort_values(["count", "feature"], ascending=[False, True])
    df.to_csv(out_path, index=False)
    return df


ensure_dir("outputs")
df_shap_all = dump_aggregate_to_csv(aggregate_top_shap, "outputs/top_shap_aggregate.csv")
df_perm_all = dump_aggregate_to_csv(aggregate_top_perm, "outputs/top_perm_aggregate.csv")
print("Exported global aggregates: outputs/top_shap_aggregate.csv, outputs/top_perm_aggregate.csv")