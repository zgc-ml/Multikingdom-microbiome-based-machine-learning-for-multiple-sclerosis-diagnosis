import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc,
    brier_score_loss,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from imblearn.combine import SMOTETomek
from collections import Counter
import warnings
import os

warnings.filterwarnings("ignore")

# ------------------ 配置：特征权重与网格搜索 ------------------
FILTERED_HIGH = 4
NORMAL_HIGH = 2
PV_LOW = 0.0001
LOWER_GROUP_WEIGHT = 0.000

TOPN_K = 30

RF_PARAM_GRID_ALL = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'class_weight': [None, 'balanced']
}

RF_PARAM_GRID_SINGLE = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [None, 10, 20, 30],
    'max_features': ['sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'class_weight': [None, 'balanced']
}

# ------------------ p 值筛选比例 ------------------
PVAL_KEEP_RATIOS = {
    'archaea': 0.01,
    'bacteria': 0.15,
    'fungi': 0.05,
    'virus': 0.05,
    'ko': 0.015,
    'path': 0.01
}

# ------------------ 路径 ------------------
base_rrms = 'D:/PNAS/PMS'
base_diet_rrms = 'D:/PNAS/Diet/PMS'
metadata_path = 'E:/iMSMS/PMS/metadata.csv'

# 运行的随机种子集合
SEEDS = [19, 27, 33, 42, 777]

# 为导出的Top特征创建输出目录（可选）
export_dir = 'D:/Codes/PMS'
os.makedirs(export_dir, exist_ok=True)

for RANDOM_STATE in SEEDS:
    print("\n" + "="*80)
    print(f"Starting full pipeline with RANDOM_STATE = {RANDOM_STATE}")
    print("="*80)

    # ------------------ 读取数据 ------------------
    features_df_b = pd.read_csv(f'{base_rrms}/bacteria_prop_residual.csv', index_col=0)
    features_df_k = pd.read_csv(f'{base_rrms}/ko_prop_residual.csv', index_col=0)
    features_df_p = pd.read_csv(f'{base_rrms}/path_prop_residual.csv', index_col=0)
    features_df_a = pd.read_csv(f'{base_rrms}/archaea_prop_residual.csv', index_col=0)
    features_df_f = pd.read_csv(f'{base_rrms}/fungi_prop_residual.csv', index_col=0)
    features_df_v = pd.read_csv(f'{base_rrms}/virus_prop_residual.csv', index_col=0)
    metadata_df = pd.read_csv(metadata_path, index_col=0)

    def load_feature_list_with_pval(path, prefix, filtered_path=None):
        df = pd.read_csv(path)
        if 'feature' not in df.columns:
            raise ValueError(f"{path} 必须包含列 'feature'")
        if 'pval' not in df.columns:
            df['pval'] = 0.0
        df['pref_feature'] = [f"{prefix}_{f}" for f in df['feature'].astype(str)]
        df = df[['pref_feature', 'pval']].rename(columns={'pref_feature': 'feature'})

        filtered_set = set()
        if filtered_path:
            filtered_df = pd.read_csv(filtered_path)
            if 'feature' in filtered_df.columns:
                filtered_set = set([f"{prefix}_{f}" for f in filtered_df['feature'].astype(str)])

        return df, filtered_set

    features_b_info, filtered_b = load_feature_list_with_pval(f'{base_diet_rrms}/bacteria_volcano.csv', 'b',
                                                              f'{base_diet_rrms}/filtered_bacteria.csv')
    features_k_info, filtered_k = load_feature_list_with_pval(f'{base_diet_rrms}/ko_volcano.csv', 'k',
                                                              f'{base_diet_rrms}/filtered_ko.csv')
    features_p_info, filtered_p = load_feature_list_with_pval(f'{base_diet_rrms}/path_volcano.csv', 'p',
                                                              f'{base_diet_rrms}/filtered_path.csv')
    features_a_info, filtered_a = load_feature_list_with_pval(f'{base_diet_rrms}/archaea_volcano.csv', 'a',
                                                              f'{base_diet_rrms}/filtered_archaea.csv')
    features_f_info, filtered_f = load_feature_list_with_pval(f'{base_diet_rrms}/fungi_volcano.csv', 'f',
                                                              f'{base_diet_rrms}/filtered_fungi.csv')
    features_v_info, filtered_v = load_feature_list_with_pval(f'{base_diet_rrms}/virus_volcano.csv', 'v',
                                                              f'{base_diet_rrms}/filtered_virus.csv')

    filtered_features = filtered_b.union(filtered_k).union(filtered_p).union(filtered_a).union(filtered_f).union(filtered_v)

    # 转置：样本为行
    features_df_b = features_df_b.T
    features_df_k = features_df_k.T
    features_df_p = features_df_p.T
    features_df_a = features_df_a.T
    features_df_f = features_df_f.T
    features_df_v = features_df_v.T

    # 添加前缀
    features_df_b.columns = ['b_' + str(col) for col in features_df_b.columns]
    features_df_k.columns = ['k_' + str(col) for col in features_df_k.columns]
    features_df_p.columns = ['p_' + str(col) for col in features_df_p.columns]
    features_df_a.columns = ['a_' + str(col) for col in features_df_a.columns]
    features_df_f.columns = ['f_' + str(col) for col in features_df_f.columns]
    features_df_v.columns = ['v_' + str(col) for col in features_df_v.columns]

    # 对齐样本索引
    dfs = [features_df_b, features_df_k, features_df_p, features_df_a, features_df_f, features_df_v]
    common_index = metadata_df.index
    for df in dfs:
        common_index = common_index.intersection(df.index)
    common_index = sorted(common_index)

    features_df_b = features_df_b.loc[common_index]
    features_df_k = features_df_k.loc[common_index]
    features_df_p = features_df_p.loc[common_index]
    features_df_a = features_df_a.loc[common_index]
    features_df_f = features_df_f.loc[common_index]
    features_df_v = features_df_v.loc[common_index]
    metadata_df = metadata_df.loc[common_index]

    # ------------------ 按 p 值筛选 ------------------
    def select_by_pval_percentage(info_df, keep_ratio):
        if keep_ratio >= 1.0:
            return info_df['feature'].tolist()
        if keep_ratio <= 0.0:
            return []
        df_sorted = info_df.sort_values('pval', ascending=True)
        k = max(1, int(np.ceil(len(df_sorted) * keep_ratio)))
        return df_sorted['feature'].head(k).tolist()

    keep_archaea = PVAL_KEEP_RATIOS.get('archaea', 1.0)
    keep_bacteria = PVAL_KEEP_RATIOS.get('bacteria', 1.0)
    keep_fungi = PVAL_KEEP_RATIOS.get('fungi', 1.0)
    keep_virus = PVAL_KEEP_RATIOS.get('virus', 1.0)
    keep_ko = PVAL_KEEP_RATIOS.get('ko', 1.0)
    keep_path = PVAL_KEEP_RATIOS.get('path', 1.0)

    selected_a_cols = select_by_pval_percentage(features_a_info, keep_archaea)
    selected_b_cols = select_by_pval_percentage(features_b_info, keep_bacteria)
    selected_f_cols = select_by_pval_percentage(features_f_info, keep_fungi)
    selected_v_cols = select_by_pval_percentage(features_v_info, keep_virus)
    selected_k_cols = select_by_pval_percentage(features_k_info, keep_ko)
    selected_p_cols = select_by_pval_percentage(features_p_info, keep_path)

    features_df_a = features_df_a.loc[:, [c for c in selected_a_cols if c in features_df_a.columns]]
    features_df_b = features_df_b.loc[:, [c for c in selected_b_cols if c in features_df_b.columns]]
    features_df_f = features_df_f.loc[:, [c for c in selected_f_cols if c in features_df_f.columns]]
    features_df_v = features_df_v.loc[:, [c for c in selected_v_cols if c in features_df_v.columns]]
    features_df_k = features_df_k.loc[:, [c for c in selected_k_cols if c in features_df_k.columns]]
    features_df_p = features_df_p.loc[:, [c for c in selected_p_cols if c in features_df_p.columns]]

    print("各类筛选后特征数：",
          f"archaea={features_df_a.shape[1]}, bacteria={features_df_b.shape[1]}, fungi={features_df_f.shape[1]}, "
          f"virus={features_df_v.shape[1]}, ko={features_df_k.shape[1]}, path={features_df_p.shape[1]}")

    # 合并特征
    features_df = pd.concat(
        [features_df_b, features_df_k, features_df_p, features_df_a, features_df_f, features_df_v],
        axis=1
    )

    # 标签
    data = features_df.join(metadata_df[['Disease']], how='inner')
    X_full_all = data.drop('Disease', axis=1)
    y_full_all = data['Disease'].map({'Control': 0, 'PMS': 1})
    target_names = ['Control', 'PMS']

    valid_idx = y_full_all.dropna().index
    X_full_all = X_full_all.loc[valid_idx]
    y_full_all = y_full_all.loc[valid_idx].astype(int)

    print(f"总样本数: {len(X_full_all)}，总特征数: {X_full_all.shape[1]}")

    # ------------------ 权重（仅 All/TopN 使用） ------------------
    pval_df = pd.concat([
        features_b_info.assign(group='b'),
        features_k_info.assign(group='k'),
        features_p_info.assign(group='p'),
        features_a_info.assign(group='a'),
        features_f_info.assign(group='f'),
        features_v_info.assign(group='v'),
    ], axis=0, ignore_index=True)

    pval_map = dict(zip(pval_df['feature'], pval_df['pval']))

    group_weight = {}
    for col in X_full_all.columns:
        group_weight[col] = 0.0 if (col.startswith('f_') or col.startswith('v_')) else 1.0

    pval_weight = {}
    for col in X_full_all.columns:
        pval = pval_map.get(col, 0.0)
        if pval < 0.05:
            pval_weight[col] = FILTERED_HIGH if col in filtered_features else NORMAL_HIGH
        else:
            pval_weight[col] = PV_LOW

    feature_weight = np.array([group_weight.get(c, 1.0) * pval_weight.get(c, PV_LOW) for c in X_full_all.columns], dtype=float)

    # ------------------ 训练/测试划分 ------------------
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_full_all, y_full_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full_all
    )
    print(f"训练样本数: {len(X_train_full)}，测试样本数: {len(X_test_full)}")

    # ------------------ 标准化 + 权重（All/TopN） ------------------
    scaler_all = StandardScaler()
    X_train_scaled = scaler_all.fit_transform(X_train_full)
    X_test_scaled = scaler_all.transform(X_test_full)
    X_train_scaled = X_train_scaled * feature_weight
    X_test_scaled = X_test_scaled * feature_weight
    X_train_cols = X_train_full.columns.tolist()

    # ------------------ 数据增强 ------------------
    sampling_methods = {
        'None': None,
        'SMOTETomek': SMOTETomek(random_state=RANDOM_STATE),
    }

    # ------------------ 模型字典 ------------------
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE, oob_score=True, bootstrap=True, n_jobs=-1),
            'param_grid': RF_PARAM_GRID_ALL
        }
    }

    # ------------------ 结果容器 ------------------
    results = {}
    roc_curve_data = {}
    trained_models = {}

    best_test_auc_all = -np.inf
    best_method_all = None
    best_trained_model_all = None
    best_roc_curve_data_all = None

    best_test_auc_topN_SHAP = -np.inf
    best_method_topN_SHAP = None
    best_trained_model_topN_SHAP = None
    best_roc_curve_data_topN_SHAP = None

    best_test_auc_topN_PERM = -np.inf
    best_method_topN_PERM = None
    best_trained_model_topN_PERM = None
    best_roc_curve_data_topN_PERM = None

    # 为输出Top30特征做占位
    final_topN_features_SHAP = None
    final_topN_importance_SHAP = None
    final_topN_features_PERM = None
    final_topN_importance_PERM = None

    # ------------------ 主循环 ------------------
    for method_name, sampler in sampling_methods.items():
        print(f"\n===== 数据增强方法: {method_name} =====")
        if sampler is not None:
            try:
                X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train_full)
            except ValueError as e:
                print(f"采样方法 {method_name} 失败，错误信息：{e}")
                continue
        else:
            X_resampled, y_resampled = X_train_scaled, y_train_full

        print(f"采样后训练集大小: {X_resampled.shape}，类别分布: {Counter(y_resampled)}")

        for model_name, model_info in models.items():
            print(f"\n======== 训练和评估模型：{model_name} ========")
            model = model_info['model']
            param_grid = model_info['param_grid']

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1,
                return_train_score=False
            )
            grid_search.fit(X_resampled, y_resampled)

            print(f"最佳参数: {grid_search.best_params_}")
            print(f"最佳交叉验证 AUC: {grid_search.best_score_:.4f}")

            best_model = grid_search.best_estimator_
            best_model.fit(X_resampled, y_resampled)

            if hasattr(best_model, 'oob_score_'):
                print(f"OOB（袋外）分数: {best_model.oob_score_:.4f}")

            trained_models[f"{method_name}_All"] = {
                'model': best_model,
                'scaler': scaler_all,
                'features': X_train_cols,
                'feature_weight': feature_weight
            }

            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
            test_accuracy = accuracy_score(y_test_full, y_pred)
            test_auc = roc_auc_score(y_test_full, y_pred_proba)
            print(f"[所有特征] 测试集准确率: {test_accuracy:.4f}")
            print(f"[所有特征] 测试集 AUC: {test_auc:.4f}")
            print("[所有特征] 测试集分类报告:")
            print(classification_report(y_test_full, y_pred, target_names=target_names))

            fpr, tpr, _ = roc_curve(y_test_full, y_pred_proba)
            roc_auc_value = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_test_full, y_pred_proba)
            ap_value = average_precision_score(y_test_full, y_pred_proba)
            roc_curve_data[f"{method_name}_{model_name}_All"] = {
                'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc_value,
                'precision': precision, 'recall': recall, 'ap': ap_value,
                'data_type': 'All', 'method_name': method_name, 'test_auc': test_auc,
                'y_true': y_test_full.values, 'y_proba': y_pred_proba
            }

            if test_auc > best_test_auc_all:
                best_test_auc_all = test_auc
                best_method_all = method_name
                best_trained_model_all = {
                    'model': best_model,
                    'scaler': scaler_all,
                    'features': X_train_cols,
                    'feature_weight': feature_weight
                }
                best_roc_curve_data_all = roc_curve_data[f"{method_name}_{model_name}_All"]

            # 手动 5 折
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
            print(f"[所有特征-手动5折] 平均 Accuracy: {np.mean(fold_accuracies):.4f}, 平均 AUC: {np.mean(fold_aucs):.4f}")

            results[f"{method_name}_{model_name}"] = {
                'best_params': grid_search.best_params_,
                'best_cv_auc': grid_search.best_score_,
                'test_accuracy_all': test_accuracy,
                'test_auc_all': test_auc,
                'cv_accuracy_avg': np.mean(fold_accuracies),
                'cv_auc_avg': np.mean(fold_aucs),
                'fpr_all': fpr,
                'tpr_all': tpr,
                'roc_auc_all': roc_auc_value
            }

            # ------------------ SHAP TopN ------------------
            print("\n开始 SHAP 分析...")
            X_shap_sample = X_train_scaled
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_shap_sample)
            if isinstance(shap_values, list):
                shap_values_to_use = shap_values[1]
            elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
                shap_values_to_use = shap_values[:, :, 1]
            else:
                shap_values_to_use = shap_values

            feature_names = X_train_cols
            mean_abs_shap_vals = np.abs(shap_values_to_use).mean(axis=0)
            shap_importance_df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap_val': mean_abs_shap_vals})
            shap_importance_df = shap_importance_df.sort_values('mean_abs_shap_val', ascending=False)
            topN_features_SHAP = shap_importance_df['feature'].head(TOPN_K).tolist()
            topN_importance_SHAP = shap_importance_df['mean_abs_shap_val'].head(TOPN_K).tolist()
            final_topN_features_SHAP = topN_features_SHAP
            final_topN_importance_SHAP = topN_importance_SHAP

            # 置换重要性
            print("\n开始置换重要性分析 (Permutation Importance)...")
            perm_result = permutation_importance(
                best_model, X_test_scaled, y_test_full,
                n_repeats=10, random_state=RANDOM_STATE, scoring='roc_auc', n_jobs=-1
            )
            perm_importances = perm_result.importances_mean
            perm_importance_df = pd.DataFrame({
                'feature': X_train_cols,
                'perm_importance': perm_importances
            }).sort_values('perm_importance', ascending=False)
            topN_features_PERM = perm_importance_df['feature'].head(TOPN_K).tolist()
            topN_importance_PERM = perm_importance_df['perm_importance'].head(TOPN_K).tolist()
            final_topN_features_PERM = topN_features_PERM
            final_topN_importance_PERM = topN_importance_PERM

            # 训练 TopN（SHAP）
            print(f"\n使用前 {TOPN_K} 个特征（SHAP）重新训练模型...")
            X_train_topN_SHAP = X_train_full[topN_features_SHAP]
            X_test_topN_SHAP = X_test_full[topN_features_SHAP]
            topN_feature_weights_SHAP = np.array(
                [ (LOWER_GROUP_WEIGHT if (f.startswith('f_') or f.startswith('v_')) else 1.0) *
                  (FILTERED_HIGH if (pval_map.get(f, 0.0) < 0.05 and f in filtered_features)
                   else (NORMAL_HIGH if pval_map.get(f, 0.0) < 0.05 else PV_LOW))
                  for f in topN_features_SHAP], dtype=float)

            scaler_topN_SHAP = StandardScaler()
            X_train_scaled_topN_SHAP = scaler_topN_SHAP.fit_transform(X_train_topN_SHAP)
            X_test_scaled_topN_SHAP = scaler_topN_SHAP.transform(X_test_topN_SHAP)
            X_train_scaled_topN_SHAP *= topN_feature_weights_SHAP
            X_test_scaled_topN_SHAP *= topN_feature_weights_SHAP

            if sampler is not None:
                X_resampled_topN_SHAP, y_resampled_topN_SHAP = sampler.fit_resample(X_train_scaled_topN_SHAP, y_train_full)
            else:
                X_resampled_topN_SHAP, y_resampled_topN_SHAP = X_train_scaled_topN_SHAP, y_train_full

            model_topN_SHAP = RandomForestClassifier(random_state=RANDOM_STATE, oob_score=True, bootstrap=True, n_jobs=-1)
            grid_search_topN_SHAP = GridSearchCV(
                estimator=model_topN_SHAP,
                param_grid=RF_PARAM_GRID_ALL,
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1,
                return_train_score=False
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
                'fpr': fpr_topN_SHAP, 'tpr': tpr_topN_SHAP, 'roc_auc': auc(fpr_topN_SHAP, tpr_topN_SHAP),
                'precision': precision_top, 'recall': recall_top, 'ap': ap_top,
                'data_type': f'Top{TOPN_K}_SHAP', 'method_name': method_name, 'test_auc': test_auc_topN_SHAP,
                'y_true': y_test_full.values, 'y_proba': y_pred_proba_topN_SHAP
            }

            trained_models[f"{method_name}_Top{TOPN_K}_SHAP"] = {
                'model': best_model_topN_SHAP,
                'scaler': scaler_topN_SHAP,
                'features': topN_features_SHAP,
                'feature_weight': topN_feature_weights_SHAP
            }

            if test_auc_topN_SHAP > best_test_auc_topN_SHAP:
                best_test_auc_topN_SHAP = test_auc_topN_SHAP
                best_method_topN_SHAP = method_name
                best_trained_model_topN_SHAP = trained_models[f"{method_name}_Top{TOPN_K}_SHAP"]
                best_roc_curve_data_topN_SHAP = roc_curve_data[f"{method_name}_{model_name}_Top{TOPN_K}_SHAP"]

            # 训练 TopN（PERM）
            print(f"\n使用前 {TOPN_K} 个特征（Permutation）重新训练模型...")
            X_train_topN_PERM = X_train_full[topN_features_PERM]
            X_test_topN_PERM = X_test_full[topN_features_PERM]

            topN_feature_weights_PERM = np.array(
                [ (LOWER_GROUP_WEIGHT if (f.startswith('f_') or f.startswith('v_')) else 1.0) *
                  (FILTERED_HIGH if (pval_map.get(f, 0.0) < 0.05 and f in filtered_features)
                   else (NORMAL_HIGH if pval_map.get(f, 0.0) < 0.05 else PV_LOW))
                  for f in topN_features_PERM], dtype=float)

            scaler_topN_PERM = StandardScaler()
            X_train_scaled_topN_PERM = scaler_topN_PERM.fit_transform(X_train_topN_PERM)
            X_test_scaled_topN_PERM = scaler_topN_PERM.transform(X_test_topN_PERM)
            X_train_scaled_topN_PERM *= topN_feature_weights_PERM
            X_test_scaled_topN_PERM *= topN_feature_weights_PERM

            if sampler is not None:
                X_resampled_topN_PERM, y_resampled_topN_PERM = sampler.fit_resample(X_train_scaled_topN_PERM, y_train_full)
            else:
                X_resampled_topN_PERM, y_resampled_topN_PERM = X_train_scaled_topN_PERM, y_train_full

            model_topN_PERM = RandomForestClassifier(random_state=RANDOM_STATE, oob_score=True, bootstrap=True, n_jobs=-1)
            grid_search_topN_PERM = GridSearchCV(
                estimator=model_topN_PERM,
                param_grid=RF_PARAM_GRID_ALL,
                scoring='roc_auc',
                cv=cv,
                n_jobs=-1,
                return_train_score=False
            )
            grid_search_topN_PERM.fit(X_resampled_topN_PERM, y_resampled_topN_PERM)
            best_model_topN_PERM = grid_search_topN_PERM.best_estimator_
            best_model_topN_PERM.fit(X_resampled_topN_PERM, y_resampled_topN_PERM)

            y_pred_proba_topN_PERM = best_model_topN_PERM.predict_proba(X_test_scaled_topN_PERM)[:, 1]
            test_auc_topN_PERM = roc_auc_score(y_test_full, y_pred_proba_topN_PERM)
            fpr_topN_PERM, tpr_topN_PERM, _ = roc_curve(y_test_full, y_pred_proba_topN_PERM)
            precision_top_p, recall_top_p, _ = precision_recall_curve(y_test_full, y_pred_proba_topN_PERM)
            ap_top_p = average_precision_score(y_test_full, y_pred_proba_topN_PERM)

            roc_curve_data[f"{method_name}_{model_name}_Top{TOPN_K}_PERM"] = {
                'fpr': fpr_topN_PERM, 'tpr': tpr_topN_PERM, 'roc_auc': auc(fpr_topN_PERM, tpr_topN_PERM),
                'precision': precision_top_p, 'recall': recall_top_p, 'ap': ap_top_p,
                'data_type': f'Top{TOPN_K}_PERM', 'method_name': method_name, 'test_auc': test_auc_topN_PERM,
                'y_true': y_test_full.values, 'y_proba': y_pred_proba_topN_PERM
            }

            trained_models[f"{method_name}_Top{TOPN_K}_PERM"] = {
                'model': best_model_topN_PERM,
                'scaler': scaler_topN_PERM,
                'features': topN_features_PERM,
                'feature_weight': topN_feature_weights_PERM
            }

            if test_auc_topN_PERM > best_test_auc_topN_PERM:
                best_test_auc_topN_PERM = test_auc_topN_PERM
                best_method_topN_PERM = method_name
                best_trained_model_topN_PERM = trained_models[f"{method_name}_Top{TOPN_K}_PERM"]
                best_roc_curve_data_topN_PERM = roc_curve_data[f"{method_name}_{model_name}_Top{TOPN_K}_PERM"]

    # 保存最佳 All 与 TopN
    trained_models['All'] = best_trained_model_all
    roc_curve_data['All'] = best_roc_curve_data_all

    # 选择 TopN 全局最佳（用于标签与外部展示）
    if best_test_auc_topN_SHAP >= best_test_auc_topN_PERM:
        trained_models[f'Top{TOPN_K}_BEST'] = best_trained_model_topN_SHAP
        roc_curve_data[f'Top{TOPN_K}_BEST'] = best_roc_curve_data_topN_SHAP
        best_topn_label = f"Top{TOPN_K} (SHAP)"
    else:
        trained_models[f'Top{TOPN_K}_BEST'] = best_trained_model_topN_PERM
        roc_curve_data[f'Top{TOPN_K}_BEST'] = best_roc_curve_data_topN_PERM
        best_topn_label = f"Top{TOPN_K} (Permutation)"

    print(f"\n最佳 All 方法: {best_method_all}, AUC: {best_test_auc_all:.4f}")
    print(f"最佳 TopN: {best_topn_label}, AUC: {roc_curve_data[f'Top{TOPN_K}_BEST']['test_auc']:.4f}")

    # 输出Top30特征列表到文件（含 importance），附加随机种子后缀以避免覆盖
    if final_topN_features_SHAP is not None and len(final_topN_features_SHAP)>0:
        pd.DataFrame({'feature': final_topN_features_SHAP, 'importance': final_topN_importance_SHAP})\
          .to_csv(f'{export_dir}/top30_shap_features_seed{RANDOM_STATE}.csv', index=False)
    if final_topN_features_PERM is not None and len(final_topN_features_PERM)>0:
        pd.DataFrame({'feature': final_topN_features_PERM, 'importance': final_topN_importance_PERM})\
          .to_csv(f'{export_dir}/top30_perm_features_seed{RANDOM_STATE}.csv', index=False)
    print(f"已导出 Top30 特征列表（含分数）：top30_shap_features_seed{RANDOM_STATE}.csv, top30_perm_features_seed{RANDOM_STATE}.csv")

    # ------------------ 单一类型：不使用权重训练，并保留 y_true/y_proba 以计算CI ------------------
    features_df_dict = {
        'archaea': features_df_a,
        'bacteria': features_df_b,
        'fungi': features_df_f,
        'virus': features_df_v,
        'ko': features_df_k,
        'path': features_df_p
    }

    data_types = ['archaea', 'bacteria', 'fungi', 'virus', 'ko', 'path']

    single_type_curves = {}

    for data_type in data_types:
        print(f"\nProcessing data type: {data_type}")
        X_type = features_df_dict[data_type].copy()
        common_idx = X_type.index.intersection(y_full_all.index)
        X_type = X_type.loc[common_idx]
        y_type = y_full_all.loc[common_idx]

        if X_type.shape[1] == 0:
            print(f"{data_type} 无可用特征，跳过。")
            continue

        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
            X_type, y_type, test_size=0.2, random_state=RANDOM_STATE, stratify=y_type
        )

        scaler_t = StandardScaler()
        X_train_t_scaled = scaler_t.fit_transform(X_train_t)
        X_test_t_scaled = scaler_t.transform(X_test_t)

        sampler = SMOTETomek(random_state=RANDOM_STATE)
        try:
            X_resampled_t, y_resampled_t = sampler.fit_resample(X_train_t_scaled, y_train_t)
        except ValueError as e:
            print(f"采样失败：{e}")
            continue

        model_t = RandomForestClassifier(random_state=RANDOM_STATE, oob_score=True, bootstrap=True, n_jobs=-1)
        grid_t = GridSearchCV(
            estimator=model_t,
            param_grid=RF_PARAM_GRID_SINGLE,
            scoring='roc_auc',
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1,
            return_train_score=False
        )
        grid_t.fit(X_resampled_t, y_resampled_t)
        best_model_t = grid_t.best_estimator_
        best_model_t.fit(X_resampled_t, y_resampled_t)

        y_proba_t = best_model_t.predict_proba(X_test_t_scaled)[:, 1]
        fpr_t, tpr_t, _ = roc_curve(y_test_t, y_proba_t)
        single_type_curves[data_type] = {
            'fpr': fpr_t, 'tpr': tpr_t,
            'roc_auc': auc(fpr_t, tpr_t),
            'y_true': y_test_t.values,
            'y_proba': y_proba_t
        }

    # ------------------ 自助法 CI 函数 ------------------
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
            tprs = [np.linspace(0,1,base_fpr.size)]
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
            prs = [np.linspace(1,0,base_recall.size)]
        prs = np.array(prs)
        mean_pr = prs.mean(axis=0)
        std_pr = prs.std(axis=0)
        ap = average_precision_score(y_true, y_score)
        return base_recall, mean_pr, std_pr, ap

    # ------------------ DCA（决策曲线分析）函数（改良版） ------------------
    def decision_curve(y_true, y_score, thresholds=None):
        """
        计算单条模型在一系列阈值下的净获益 Net Benefit:
          NB = TP/N - FP/N * (pt/(1-pt))
        Treat-All: NB_all = prevalence - (1 - prevalence) * (pt/(1-pt))
        Treat-None: NB_none = 0
        """
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=float)
        N = len(y_true)
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.95, 95)  # 扩展阈值范围，避免0和1的奇异点
        thresholds = np.asarray(thresholds, dtype=float)

        # 避免极端概率导致的数值问题
        y_score = np.clip(y_score, 1e-12, 1 - 1e-12)

        nb_model = []
        for pt in thresholds:
            if pt <= 0 or pt >= 1:
                nb_model.append(np.nan)
                continue
            y_pred = (y_score >= pt).astype(int)
            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            w = pt / (1 - pt)
            nb = (TP / N) - (FP / N) * w
            nb_model.append(nb)
        nb_model = np.array(nb_model, dtype=float)

        prevalence = np.mean(y_true)
        valid = (thresholds > 0) & (thresholds < 1)
        w_all = np.zeros_like(thresholds, dtype=float)
        w_all[valid] = thresholds[valid] / (1 - thresholds[valid])
        nb_all = np.zeros_like(thresholds, dtype=float)
        nb_all[valid] = prevalence - (1 - prevalence) * w_all[valid]
        nb_none = np.zeros_like(thresholds, dtype=float)

        return thresholds, nb_model, nb_all, nb_none

    def bootstrap_decision_curve(y_true, y_score, thresholds=None, n_boot=500, random_state=RANDOM_STATE):
        """
        对 DCA 曲线进行自助法评估，返回：
          thresholds, mean_nb, std_nb, (nb_all, nb_none 基于整体 y_true 计算，不自助)
        """
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.95, 95)
        thresholds = np.asarray(thresholds, dtype=float)

        rng = np.random.RandomState(random_state)
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score, dtype=float)

        nb_mat = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(y_true), len(y_true))
            y_b = y_true[idx]
            s_b = y_score[idx]
            th, nb_b, _, _ = decision_curve(y_b, s_b, thresholds)
            nb_mat.append(nb_b)
        nb_mat = np.array(nb_mat)
        mean_nb = np.nanmean(nb_mat, axis=0)
        std_nb = np.nanstd(nb_mat, axis=0)

        # 基线始终基于完整 y_true（更稳定）
        _, _, nb_all, nb_none = decision_curve(y_true, y_score, thresholds)

        return thresholds, mean_nb, std_nb, nb_all, nb_none

    def _set_cns_style():
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 22,
            'axes.labelsize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16
        })

    def plot_dca_curves(curves, title, thresholds=None, n_boot=500):
        """
        curves: list of dicts {
            'label': str,
            'y_true': 1d array,
            'y_proba': 1d array,
            'color': optional
        }
        在同一人群（相同 y_true）下比较多模型时，Treat-All/None 参考线有严格意义；
        如人群不同，则以第一条曲线的人群为参考计算基线，并在标题中已明确展示数据集信息。
        """
        _set_cns_style()
        if thresholds is None:
            thresholds = np.linspace(0.01, 0.95, 95)

        plt.figure(figsize=(10, 8))
        all_nb_vals = []
        base_th = None
        base_nb_all = None
        base_nb_none = None

        # 判断是否同一人群（用于严格绘制基线）
        same_population = True
        if len(curves) >= 2:
            ref = np.asarray(curves[0]['y_true']).astype(int)
            for c in curves[1:]:
                if not np.array_equal(ref, np.asarray(c['y_true']).astype(int)):
                    same_population = False
                    break

        for c in curves:
            th, mean_nb, std_nb, nb_all, nb_none = bootstrap_decision_curve(
                c['y_true'], c['y_proba'], thresholds=thresholds, n_boot=n_boot, random_state=RANDOM_STATE
            )
            color = c.get('color', None)
            plt.plot(th, mean_nb, lw=3, label=c['label'], color=color)
            plt.fill_between(th, mean_nb - 1.96 * std_nb, mean_nb + 1.96 * std_nb,
                             alpha=0.18, color=color)
            all_nb_vals.append(mean_nb)

            # 保存基线（首个曲线的人群）
            if base_th is None:
                base_th = th
                base_nb_all = nb_all
                base_nb_none = nb_none

        # Treat-All / Treat-None
        if base_th is not None:
            plt.plot(base_th, base_nb_all, color='gray', lw=2.5, linestyle='-.', label='Treat-All')
            plt.plot(base_th, base_nb_none, color='black', lw=2.5, linestyle='-', label='Treat-None')

        # y 轴自适应与留白
        if all_nb_vals:
            all_vals = np.concatenate(all_nb_vals)
            finite_vals = all_vals[np.isfinite(all_vals)]
            if finite_vals.size > 0:
                y_min = float(np.min(finite_vals))
                y_max = float(np.max(finite_vals))
                margin = max(0.002, 0.25 * (y_max - y_min + 1e-6))
                plt.ylim(y_min - margin, y_max + margin)

        plt.xlabel('Threshold probability (pt)')
        plt.ylabel('Net benefit')
        plt.title(title)
        plt.legend(loc='lower right', frameon=False)
        plt.tight_layout()
        plt.show()

    # 全局绘图字号设置
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })

    # ------------------ 图一：六个单类 + All + Top30(仅AUC更高的一种)，全部带95%CI阴影 ------------------
    plt.figure(figsize=(10, 8))

    # 单类
    for k, res in single_type_curves.items():
        y_true, y_proba = res['y_true'], res['y_proba']
        base_fpr, mean_tpr, std_tpr = compute_bootstrap_roc(y_true, y_proba)
        auc_val = res['roc_auc']
        plt.plot(base_fpr, mean_tpr, lw=3, label=f'{k} (AUC = {auc_val:.4f})')
        plt.fill_between(base_fpr, np.maximum(mean_tpr-1.96*std_tpr,0), np.minimum(mean_tpr+1.96*std_tpr,1), alpha=0.15)

    # All
    if roc_curve_data.get('All'):
        y_true_all = roc_curve_data['All']['y_true']
        y_proba_all = roc_curve_data['All']['y_proba']
        base_fpr, mean_tpr, std_tpr = compute_bootstrap_roc(y_true_all, y_proba_all)
        auc_all = roc_curve_data['All']['roc_auc']
        plt.plot(base_fpr, mean_tpr, lw=3.5, label=f'All (AUC = {auc_all:.4f})', linestyle='--')
        plt.fill_between(base_fpr, np.maximum(mean_tpr-1.96*std_tpr,0), np.minimum(mean_tpr+1.96*std_tpr,1), alpha=0.18)

    # Top30：只选 AUC 更高的一种（SHAP 或 PERM），legend 统一为 Top30
    best_top_key = None
    best_top_auc = -np.inf
    cand_keys = [k for k in roc_curve_data.keys() if k.endswith(f'Top{TOPN_K}_SHAP')] + \
                [k for k in roc_curve_data.keys() if k.endswith(f'Top{TOPN_K}_PERM')]
    for k in cand_keys:
        if 'roc_auc' in roc_curve_data[k]:
            auc_val = roc_curve_data[k]['roc_auc']
            if auc_val > best_top_auc:
                best_top_auc = auc_val
                best_top_key = k

    if best_top_key is not None:
        res = roc_curve_data[best_top_key]
        y_true = res['y_true']; y_proba = res['y_proba']
        base_fpr, mean_tpr, std_tpr = compute_bootstrap_roc(y_true, y_proba)
        plt.plot(base_fpr, mean_tpr, lw=3.5, label=f'Top{TOPN_K} (AUC = {best_top_auc:.4f})', linestyle=':')
        plt.fill_between(base_fpr, np.maximum(mean_tpr-1.96*std_tpr,0), np.minimum(mean_tpr+1.96*std_tpr,1), alpha=0.18)

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':')
    plt.xlim([-0.01, 1.01]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.show()

    # 图一对应的 DCA（同一套模型，改良版）
    curves = []
    for k, res in single_type_curves.items():
        curves.append({'label': k, 'y_true': res['y_true'], 'y_proba': res['y_proba']})
    if roc_curve_data.get('All'):
        curves.append({'label': 'All', 'y_true': roc_curve_data['All']['y_true'], 'y_proba': roc_curve_data['All']['y_proba']})
    if best_top_key is not None:
        res = roc_curve_data[best_top_key]
        curves.append({'label': f'Top{TOPN_K}', 'y_true': res['y_true'], 'y_proba': res['y_proba']})
    plot_dca_curves(curves, 'Decision Curve (internal)', thresholds=np.linspace(0.01, 0.95, 95), n_boot=500)

    # 图一对应的 PR 曲线（内部测试集）
    plt.figure(figsize=(10,8))

    # 单类 PR
    for k, res in single_type_curves.items():
        y_true, y_proba = res['y_true'], res['y_proba']
        base_rec, mean_pr, std_pr, ap = compute_bootstrap_pr(y_true, y_proba)
        plt.plot(base_rec, mean_pr, lw=3, label=f'{k} (AP = {ap:.4f})')
        plt.fill_between(base_rec, np.maximum(mean_pr-1.96*std_pr,0), np.minimum(mean_pr+1.96*std_pr,1), alpha=0.15)

    # All PR
    if roc_curve_data.get('All'):
        y_true_all = roc_curve_data['All']['y_true']
        y_proba_all = roc_curve_data['All']['y_proba']
        base_rec, mean_pr, std_pr, ap_all = compute_bootstrap_pr(y_true_all, y_proba_all)
        plt.plot(base_rec, mean_pr, lw=3.5, label=f'All (AP = {ap_all:.4f})', linestyle='--')
        plt.fill_between(base_rec, np.maximum(mean_pr-1.96*std_pr,0), np.minimum(mean_pr+1.96*std_pr,1), alpha=0.18)

    # Top30 PR
    if best_top_key is not None:
        res = roc_curve_data[best_top_key]
        y_true = res['y_true']; y_proba = res['y_proba']
        base_rec, mean_pr, std_pr, ap_top = compute_bootstrap_pr(y_true, y_proba)
        plt.plot(base_rec, mean_pr, lw=3.5, label=f'Top{TOPN_K} (AP = {ap_top:.4f})', linestyle=':')
        plt.fill_between(base_rec, np.maximum(mean_pr-1.96*std_pr,0), np.minimum(mean_pr+1.96*std_pr,1), alpha=0.18)

    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    plt.show()

    # 图一对应的 校准曲线（内部测试集）+ Brier
    plt.figure(figsize=(9,7))
    def plot_cal_curve(y_true, y_proba, label, style='-', lw=3):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy='quantile')
        brier = brier_score_loss(y_true, y_proba)
        plt.plot(prob_pred, prob_true, lw=lw, label=f'{label} (Brier={brier:.3f})', linestyle=style, marker='o', markersize=8)

    plt.plot([0,1],[0,1],'k--', lw=2, label='Perfectly calibrated')
    # 单类
    for k, res in single_type_curves.items():
        plot_cal_curve(res['y_true'], res['y_proba'], k, lw=3)
    # All
    if roc_curve_data.get('All'):
        plot_cal_curve(roc_curve_data['All']['y_true'], roc_curve_data['All']['y_proba'], 'All', style='--', lw=3.5)
    # Top30
    if best_top_key is not None:
        res = roc_curve_data[best_top_key]
        plot_cal_curve(res['y_true'], res['y_proba'], f'Top{TOPN_K}', style=':', lw=3.5)

    plt.xlabel('Predicted probability')
    plt.ylabel('Observed frequency')
    plt.xlim([0,1]); plt.ylim([0,1])
    plt.legend(frameon=False, loc='lower right')
    plt.tight_layout()
    plt.show()

    # ------------------ 外部验证（含 AS1、AS2） ------------------
    external_datasets = {
        'Daiki': 'E:/DRA007704/SPMS',
        'AS1': 'D:/PNAS/PRJNA375935',
        'AS2': 'D:/PNAS/PRJEB29373',
        'NMOSD': 'E:/DRA007704/NMOSD'
    }

    def prepare_external_features(dataset_path):
        features_df_b_ext = pd.read_csv(f'{dataset_path}/bacteria_prop_residual.csv', index_col=0).T
        features_df_k_ext = pd.read_csv(f'{dataset_path}/ko_prop_residual.csv', index_col=0).T
        features_df_p_ext = pd.read_csv(f'{dataset_path}/path_prop_residual.csv', index_col=0).T
        features_df_a_ext = pd.read_csv(f'{dataset_path}/archaea_prop_residual.csv', index_col=0).T
        features_df_f_ext = pd.read_csv(f'{dataset_path}/fungi_prop_residual.csv', index_col=0).T
        features_df_v_ext = pd.read_csv(f'{dataset_path}/virus_prop_residual.csv', index_col=0).T
        metadata_df_ext = pd.read_csv(f'{dataset_path}/metadata.csv', index_col=0)
        features_df_b_ext.columns = ['b_' + str(c) for c in features_df_b_ext.columns]
        features_df_k_ext.columns = ['k_' + str(c) for c in features_df_k_ext.columns]
        features_df_p_ext.columns = ['p_' + str(c) for c in features_df_p_ext.columns]
        features_df_a_ext.columns = ['a_' + str(c) for c in features_df_a_ext.columns]
        features_df_f_ext.columns = ['f_' + str(c) for c in features_df_f_ext.columns]
        features_df_v_ext.columns = ['v_' + str(c) for c in features_df_v_ext.columns]
        features_df_all_ext = pd.concat(
            [features_df_b_ext, features_df_k_ext, features_df_p_ext, features_df_a_ext, features_df_f_ext, features_df_v_ext],
            axis=1
        )
        return features_df_all_ext, metadata_df_ext

    def transform_ext(X_ext_all, model_info):
        features_in_model = model_info['features']
        scaler_ext = model_info['scaler']
        X_ext = X_ext_all.reindex(columns=features_in_model, fill_value=0)
        X_ext_scaled = scaler_ext.transform(X_ext)
        fw = model_info.get('feature_weight', None)
        if fw is not None and np.ndim(fw) == 1 and fw.shape[0] == X_ext_scaled.shape[1]:
            X_ext_scaled = X_ext_scaled * fw
        return X_ext_scaled

    external_results = {}  # {dataset: {'All':..., 'TopN':...}}

    best_all = trained_models.get('All')
    best_top_best = trained_models.get(f'Top{TOPN_K}_BEST')

    label_encoder_ext = {'Control': 0, 'CTRL': 0, 'Healthy': 0, 'PMS': 1, 'MS': 1, 'AS': 1, 'NMOSD': 1}

    for dataset_name, dataset_path in external_datasets.items():
        print(f"\nProcessing external dataset: {dataset_name}")
        try:
            X_ext_all, metadata_df_ext = prepare_external_features(dataset_path)
        except FileNotFoundError as e:
            print(f"Error reading files for dataset {dataset_name}: {e}")
            continue

        common_index_ext = metadata_df_ext.index.intersection(X_ext_all.index)
        common_index_ext = sorted(common_index_ext)
        y_ext_raw = metadata_df_ext.loc[common_index_ext, 'Disease']
        y_ext = y_ext_raw.map(label_encoder_ext).dropna().astype(int)
        valid_idx = y_ext.index
        X_ext_all = X_ext_all.loc[valid_idx]

        results_per_ds = {}

        # All
        if best_all is not None:
            X_ext_scaled_all = transform_ext(X_ext_all, best_all)
            y_proba = best_all['model'].predict_proba(X_ext_scaled_all)[:, 1]
            fpr, tpr, _ = roc_curve(y_ext, y_proba)
            precision, recall, _ = precision_recall_curve(y_ext, y_proba)
            ap = average_precision_score(y_ext, y_proba)
            results_per_ds['All'] = {
                'y_true': y_ext.values, 'y_proba': y_proba,
                'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr),
                'precision': precision, 'recall': recall, 'ap': ap
            }

        # TopN（统一命名为 Top30）
        if best_top_best is not None:
            X_ext_scaled = transform_ext(X_ext_all, best_top_best)
            y_proba = best_top_best['model'].predict_proba(X_ext_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_ext, y_proba)
            precision, recall, _ = precision_recall_curve(y_ext, y_proba)
            ap = average_precision_score(y_ext, y_proba)
            results_per_ds[f'Top{TOPN_K}'] = {
                'y_true': y_ext.values, 'y_proba': y_proba,
                'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr),
                'precision': precision, 'recall': recall, 'ap': ap
            }

        external_results[dataset_name] = results_per_ds
        print(f"{dataset_name} class distribution: {Counter(y_ext)}")

    # ------------------ 外部验证：前三个数据集分别在同一图中展示（分开 All 与 Top30），并各自画 PR + DCA ------------------
    def plot_ci_line_roc(y_true, y_proba, label, color=None, lw=3, n_bootstrap=500):
        base_fpr = np.linspace(0,1,101)
        rng = np.random.RandomState(RANDOM_STATE)
        tprs = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true[idx], y_proba[idx])
            tprs.append(np.interp(base_fpr, fpr, tpr))
        if len(tprs)==0:
            tprs=[np.linspace(0,1,base_fpr.size)]
        tprs = np.array(tprs)
        mean_tpr = tprs.mean(axis=0)
        std_tpr = tprs.std(axis=0)
        plt.plot(base_fpr, mean_tpr, lw=lw, label=label, color=color)
        plt.fill_between(base_fpr, np.maximum(mean_tpr-1.96*std_tpr,0), np.minimum(mean_tpr+1.96*std_tpr,1),
                         alpha=0.2, color=color)

    def plot_ci_line_pr(y_true, y_proba, label, color=None, lw=3, n_bootstrap=500):
        base_recall = np.linspace(0,1,101)
        rng = np.random.RandomState(RANDOM_STATE)
        prs = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(y_true), len(y_true))
            if len(np.unique(y_true[idx])) < 2:
                continue
            precision, recall, _ = precision_recall_curve(y_true[idx], y_proba[idx])
            precision_interp = np.interp(base_recall, recall[::-1], precision[::-1])
            prs.append(precision_interp)
        if len(prs)==0:
            prs=[np.linspace(1,0,base_recall.size)]
        prs = np.array(prs)
        mean_pr = prs.mean(axis=0)
        std_pr = prs.std(axis=0)
        ap = average_precision_score(y_true, y_proba)
        plt.plot(base_recall, mean_pr, lw=lw, label=f'{label} (AP={ap:.3f})', color=color)
        plt.fill_between(base_recall, np.maximum(mean_pr-1.96*std_pr,0), np.minimum(mean_pr+1.96*std_pr,1),
                         alpha=0.2, color=color)

    colors_all = ['#1f77b4', '#2ca02c', '#9467bd']
    colors_top = ['#d62728', '#ff7f0e', '#17becf']
    cohort_group_1 = ['Daiki']

    # ROC: All
    plt.figure(figsize=(10,8))
    for i, ds in enumerate(cohort_group_1):
        entry_all = external_results.get(ds, {}).get('All')
        if entry_all is None:
            continue
        auc_val = entry_all['auc']
        plot_ci_line_roc(entry_all['y_true'], entry_all['y_proba'], f'{ds} - All (AUC={auc_val:.3f})', color=colors_all[i%len(colors_all)], lw=3)
    plt.plot([0,1],[0,1],'k:', lw=2)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('External ROC (All): Ventura, Daiki, Florence_Sudeep')
    plt.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.show()

    # DCA: All（改良版）
    curves = []
    for i, ds in enumerate(cohort_group_1):
        entry_all = external_results.get(ds, {}).get('All')
        if entry_all is None:
            continue
        curves.append({'label': f'{ds} - All', 'y_true': entry_all['y_true'], 'y_proba': entry_all['y_proba'], 'color': colors_all[i%len(colors_all)]})
    plot_dca_curves(curves, 'External DCA (All): Ventura, Daiki, Florence_Sudeep', thresholds=np.linspace(0.01,0.95,95), n_boot=500)

    # ROC: Top30
    plt.figure(figsize=(10,8))
    for i, ds in enumerate(cohort_group_1):
        entry_top = external_results.get(ds, {}).get(f'Top{TOPN_K}')
        if entry_top is None:
            continue
        auc_val = entry_top['auc']
        plot_ci_line_roc(entry_top['y_true'], entry_top['y_proba'], f'{ds} - Top{TOPN_K} (AUC={auc_val:.3f})', color=colors_top[i%len(colors_top)], lw=3)
    plt.plot([0,1],[0,1],'k:', lw=2)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('External ROC (Top30): Ventura, Daiki, Florence_Sudeep')
    plt.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.show()

    # DCA: Top30（改良版）
    curves = []
    for i, ds in enumerate(cohort_group_1):
        entry_top = external_results.get(ds, {}).get(f'Top{TOPN_K}')
        if entry_top is None:
            continue
        curves.append({'label': f'{ds} - Top{TOPN_K}', 'y_true': entry_top['y_true'], 'y_proba': entry_top['y_proba'], 'color': colors_top[i%len(colors_top)]})
    plot_dca_curves(curves, 'External DCA (Top30): Ventura, Daiki, Florence_Sudeep', thresholds=np.linspace(0.01,0.95,95), n_boot=500)

    # PR: All
    plt.figure(figsize=(10,8))
    for i, ds in enumerate(cohort_group_1):
        entry_all = external_results.get(ds, {}).get('All')
        if entry_all is None:
            continue
        plot_ci_line_pr(entry_all['y_true'], entry_all['y_proba'], f'{ds} - All', color=colors_all[i%len(colors_all)], lw=3)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('External PR (All): Ventura, Daiki, Florence_Sudeep')
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    plt.show()

    # PR: Top30
    plt.figure(figsize=(10,8))
    for i, ds in enumerate(cohort_group_1):
        entry_top = external_results.get(ds, {}).get(f'Top{TOPN_K}')
        if entry_top is None:
            continue
        plot_ci_line_pr(entry_top['y_true'], entry_top['y_proba'], f'{ds} - Top{TOPN_K}', color=colors_top[i%len(colors_top)], lw=3)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('External PR (Top30): Ventura, Daiki, Florence_Sudeep')
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    plt.show()

    # ------------------ 外部验证：AS1、AS2 分开 All 与 Top30 两套 ROC + PR + DCA ------------------
    cohort_group_2 = ['AS1', 'AS2', 'NMOSD']

    # ROC: All
    plt.figure(figsize=(10,8))
    for i, ds in enumerate(cohort_group_2):
        entry_all = external_results.get(ds, {}).get('All')
        if entry_all is None:
            continue
        auc_val = entry_all['auc']
        plot_ci_line_roc(entry_all['y_true'], entry_all['y_proba'], f'{ds} - All (AUC={auc_val:.3f})', color=colors_all[i%len(colors_all)], lw=3)
    plt.plot([0,1],[0,1],'k:', lw=2)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('External ROC (All): AS1, AS2, NMOSD')
    plt.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.show()

    # DCA: All（改良版）
    curves = []
    for i, ds in enumerate(cohort_group_2):
        entry_all = external_results.get(ds, {}).get('All')
        if entry_all is None:
            continue
        curves.append({'label': f'{ds} - All', 'y_true': entry_all['y_true'], 'y_proba': entry_all['y_proba'], 'color': colors_all[i%len(colors_all)]})
    plot_dca_curves(curves, 'External DCA (All): AS1, AS2, NMOSD', thresholds=np.linspace(0.01,0.95,95), n_boot=500)

    # ROC: Top30
    plt.figure(figsize=(10,8))
    for i, ds in enumerate(cohort_group_2):
        entry_top = external_results.get(ds, {}).get(f'Top{TOPN_K}')
        if entry_top is None:
            continue
        auc_val = entry_top['auc']
        plot_ci_line_roc(entry_top['y_true'], entry_top['y_proba'], f'{ds} - Top{TOPN_K} (AUC={auc_val:.3f})', color=colors_top[i%len(colors_top)], lw=3)
    plt.plot([0,1],[0,1],'k:', lw=2)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('External ROC (Top30): AS1, AS2, NMOSD')
    plt.legend(loc='lower right', frameon=False)
    plt.tight_layout()
    plt.show()

    # DCA: Top30（改良版）
    curves = []
    for i, ds in enumerate(cohort_group_2):
        entry_top = external_results.get(ds, {}).get(f'Top{TOPN_K}')
        if entry_top is None:
            continue
        curves.append({'label': f'{ds} - Top{TOPN_K}', 'y_true': entry_top['y_true'], 'y_proba': entry_top['y_proba'], 'color': colors_top[i%len(colors_top)]})
    plot_dca_curves(curves, 'External DCA (Top30): AS1, AS2, NMOSD', thresholds=np.linspace(0.01,0.95,95), n_boot=500)

    # PR: All
    plt.figure(figsize=(10,8))
    for i, ds in enumerate(cohort_group_2):
        entry_all = external_results.get(ds, {}).get('All')
        if entry_all is None:
            continue
        plot_ci_line_pr(entry_all['y_true'], entry_all['y_proba'], f'{ds} - All', color=colors_all[i%len(colors_all)], lw=3)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('External PR (All): AS1, AS2, NMOSD')
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    plt.show()

    # PR: Top30
    plt.figure(figsize=(10,8))
    for i, ds in enumerate(cohort_group_2):
        entry_top = external_results.get(ds, {}).get(f'Top{TOPN_K}')
        if entry_top is None:
            continue
        plot_ci_line_pr(entry_top['y_true'], entry_top['y_proba'], f'{ds} - Top{TOPN_K}', color=colors_top[i%len(colors_top)], lw=3)
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('External PR (Top30): AS1, AS2, NMOSD')
    plt.legend(loc='lower left', frameon=False)
    plt.tight_layout()
    plt.show()

    # ------------------ AS 专属：两张ROC图（All、Top30），并计算Control样本准确率（保持不变） + 各自DCA（改良版） ------------------
    best_all = trained_models.get('All')

    for as_name in ['AS1','AS2', 'NMOSD']:
        if as_name not in external_results:
            print(f"{as_name} not found in external results.")
            continue
        res = external_results[as_name]

        plt.figure(figsize=(9,7))
        for tag, pretty in [('All','All'), (f'Top{TOPN_K}', f'Top{TOPN_K}')]:
            if tag in res:
                entry = res[tag]
                fpr, tpr = entry['fpr'], entry['tpr']
                auc_val = entry['auc']
                plt.plot(fpr, tpr, lw=3, label=f'{pretty} (AUC={auc_val:.3f})')
        plt.plot([0,1],[0,1],'k--')
        plt.xlim([0,1]); plt.ylim([0,1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'{as_name} ROC')
        plt.legend(loc='lower right', frameon=False)
        plt.tight_layout()
        plt.show()

        # DCA（改良版）
        curves = []
        for tag, pretty in [('All','All'), (f'Top{TOPN_K}', f'Top{TOPN_K}')]:
            if tag in res:
                entry = res[tag]
                curves.append({'label': pretty, 'y_true': entry['y_true'], 'y_proba': entry['y_proba']})
        plot_dca_curves(curves, f'{as_name} DCA', thresholds=np.linspace(0.01,0.95,95), n_boot=500)

        # Control 样本准确率
        try:
            X_ext_all, metadata_df_ext = prepare_external_features({'AS1':'D:/PNAS/PRJNA375935','AS2':'D:/PNAS/PRJEB29373', 'NMOSD':'E:/DRA007704/NMOSD'}[as_name])
        except Exception as e:
            print(f"Reload {as_name} failed: {e}")
            continue
        common_index_ext = metadata_df_ext.index.intersection(X_ext_all.index)
        common_index_ext = sorted(common_index_ext)
        y_ext_raw = metadata_df_ext.loc[common_index_ext, 'Disease']
        y_map = y_ext_raw.map({'Control': 0, 'CTRL': 0, 'Healthy': 0, 'RRMS': 1, 'MS': 1, 'AS': 1, 'NMOSD':1}).dropna().astype(int)
        valid_idx = y_map.index
        X_ext_all = X_ext_all.loc[valid_idx]
        control_idx = y_map[y_map==0].index
        if len(control_idx)==0:
            print(f"{as_name} 无 Control 样本")
            continue
        X_control = X_ext_all.loc[control_idx]

        if best_all is None:
            print("无 All 最佳模型，无法做 Control 准确率")
            continue
        X_control_scaled = transform_ext(X_control, best_all)
        y_pred_control = (best_all['model'].predict_proba(X_control_scaled)[:,1] >= 0.5).astype(int)
        control_accuracy = (y_pred_control==0).mean()
        print(f"{as_name} Control 样本数={len(control_idx)}，预测为阴性的准确率={control_accuracy:.4f}")

    print(f"Finished pipeline for RANDOM_STATE = {RANDOM_STATE}")