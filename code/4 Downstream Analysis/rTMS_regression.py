import random
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    path_excel = '../b_data/total_table.xlsx'
    with pd.ExcelFile(path_excel) as xls:
        rtms = pd.read_excel(xls, sheet_name='table_3_rTMS')
        KG = pd.read_excel(xls, sheet_name='XJ_1_KG_73')
        KG_B = KG[KG['Classification'] == 'B']
        QG = pd.read_excel(xls, sheet_name='XJ_2_QG_44')
        QG_B = QG[QG['Classification'] == 'B']

    index_rtms = rtms['HAMD-30d']
    index_kg = KG_B['HAMD-17-pre'] - KG_B['HAMD-17-post']
    index_qg = QG_B['HAMD-17-pre'] - QG_B['HAMD-17-post']

    X = np.load('../e_prognosis/rTMS_data_512.npz')['X']
    y = pd.concat([index_rtms, index_kg, index_qg], ignore_index=True).values.astype(np.float32)
    return X, y


def calculate_metrics(y_true, y_pred):
    p_r, _ = pearsonr(y_true, y_pred)
    s_r, _ = spearmanr(y_true, y_pred)
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pearson_r': p_r,
        'spearman_r': s_r
    }


def get_regressor(model_name, seed):
    models = {
        'LightGBM': lgb.LGBMRegressor(n_estimators=1000, max_depth=3, learning_rate=0.01, random_state=seed,
                                      verbosity=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=1000, max_depth=3, learning_rate=0.05, random_state=seed, verbosity=0),
        'RandomForest': RandomForestRegressor(n_estimators=500, max_depth=10, random_state=seed, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.05,
                                                      random_state=seed),
        'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
        'Ridge': Ridge(alpha=1.0, random_state=seed),
        'Lasso': Lasso(alpha=1.0, random_state=seed, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=seed, max_iter=10000)
    }
    return models[model_name]


def nested_cv_with_model(X, y, model_name, n_components, seed):
    set_seed(seed)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_metrics = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        pca = PCA(n_components=n_components, random_state=seed)
        X_train_pca = pca.fit_transform(X_train_s)
        X_test_pca = pca.transform(X_test_s)

        model = get_regressor(model_name, seed)
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)

        fold_metrics.append(calculate_metrics(y_test, y_pred))

    return {
        metric: {
            'mean': np.mean([m[metric] for m in fold_metrics]),
            'std': np.std([m[metric] for m in fold_metrics])
        } for metric in fold_metrics[0].keys()
    }


def main():
    X, y = load_data()
    models = ['LightGBM', 'XGBoost', 'RandomForest', 'GradientBoosting', 'SVR', 'Ridge', 'Lasso', 'ElasticNet']
    dimensions = [10, 20, 30, 40, 50, 60]
    results = {}

    for name in models:
        results[name] = {}
        for dim in dimensions:
            res = nested_cv_with_model(X, y, name, dim, 98)
            results[name][dim] = {
                'r_mean': res['pearson_r']['mean'],
                'r_std': res['pearson_r']['std']
            }

    m_names = list(results.keys())
    matrix_mean = np.array([[results[m][d]['r_mean'] for d in dimensions] for m in m_names])
    matrix_std = np.array([[results[m][d]['r_std'] for d in dimensions] for m in m_names])

    fig, ax = plt.subplots(figsize=(14, 8))
    annot = np.array([[f"{matrix_mean[i, j]:.3f}\n±{matrix_std[i, j]:.3f}" for j in range(len(dimensions))] for i in
                      range(len(m_names))])

    sns.heatmap(matrix_mean, annot=annot, fmt='', cmap='RdYlGn', xticklabels=dimensions, yticklabels=m_names, ax=ax)
    plt.title('rTMS Treatment Outcome Prediction Performance')
    plt.savefig('comparison_heatmap.png', dpi=300)
    plt.show()

    res_df = []
    for m in results:
        for d in results[m]:
            res_df.append({'Model': m, 'PCA': d, 'R_Mean': results[m][d]['r_mean'], 'R_Std': results[m][d]['r_std']})
    pd.DataFrame(res_df).to_csv('regression_results.csv', index=False)


if __name__ == "__main__":
    main()