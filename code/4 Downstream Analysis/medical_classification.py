"""TreatmentResponsive classification - Simplified"""
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def train_mlp(X_train, Y_train, hidden_dim, lr):
    set_seed()
    model = MLP(X_train.shape[1], hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X_t, Y_t = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
    for _ in range(50):
        optimizer.zero_grad()
        nn.BCELoss()(model(X_t), Y_t).backward()
        optimizer.step()
    return model


models_config = {
    'KNN': {'class': KNeighborsClassifier, 'params': {'n_neighbors': [3,5,7,9,11,15,21], 'weights': ['uniform','distance']}},
    'XGBoost': {'class': xgb.XGBClassifier, 'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 9],'learning_rate': [0.01, 0.05, 0.1, 0.3]}},
    'LogisticRegression': {'class': LogisticRegression,'params': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'],'solver': ['liblinear']}},
    'MLP': {'class': None,'params': {'hidden_dim': [16, 32, 64, 128, 256, 512, 1024], 'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}},
    'LightGBM': {'class': lgb.LGBMClassifier, 'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, -1], 'learning_rate': [0.01, 0.05, 0.1, 0.3],'verbosity': [-1]}},
    'GradientBoosting': {'class': GradientBoostingClassifier, 'params': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1, 0.3]}},
}


def nested_cv(X, y, name, config, n_outer=5, n_inner=3, max_comb=50):
    set_seed()
    outer_cv = KFold(n_outer, shuffle=True, random_state=42)
    inner_cv = KFold(n_inner, shuffle=True, random_state=42)
    results = []

    from itertools import product
    params_list = list(config['params'].keys())
    if name == 'MLP':
        combinations = [dict(zip(params_list, c)) for c in product(*config['params'].values())]
    else:
        combinations = [{p: random.choice(config['params'][p]) for p in params_list} for _ in range(max_comb)]

    supports_random_state = name not in ['KNN']

    for train_idx, test_idx in outer_cv.split(X):
        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        best_params, best_score = None, -np.inf

        for params in combinations:
            scores = []
            for tr_idx, val_idx in inner_cv.split(X_train):
                try:
                    if name == 'MLP':
                        model = train_mlp(X_train[tr_idx], y_train[tr_idx], **params)
                        with torch.no_grad():
                            pred = (model(torch.tensor(X_train[val_idx], dtype=torch.float32)).numpy().flatten() > 0.5).astype(int)
                    else:
                        if supports_random_state:
                            model = config['class'](random_state=42, **params)
                        else:
                            model = config['class'](**params)
                        model.fit(X_train[tr_idx], y_train[tr_idx])
                        pred = model.predict(X_train[val_idx])
                    scores.append(accuracy_score(y_train[val_idx], pred))
                except:
                    continue
            if scores and np.mean(scores) > best_score:
                best_score, best_params = np.mean(scores), params

        if best_params is None:
            best_params = combinations[0]

        if name == 'MLP':
            model = train_mlp(X_train, y_train, **best_params)
            with torch.no_grad():
                pred = (model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten() > 0.5).astype(int)
        else:
            if supports_random_state:
                model = config['class'](random_state=42, **best_params)
            else:
                model = config['class'](**best_params)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        results.append(accuracy_score(y_test, pred))

    return np.mean(results), np.std(results)

def set_seed(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    data_loaded = np.load('../e_prognosis/drug_data_512.npz')
    csv_medical = pd.read_csv('../e_prognosis/table_7_medication.csv', header=0, encoding='gbk')
    group = csv_medical['MedicationGroup'].index
    X = data_loaded['X'][group, :]
    Y = data_loaded['Y'][group]

    print("\n" + "="*60)
    print("ACCURACY RESULTS")
    print("="*60)

    for name, config in models_config.items():
        acc_mean, acc_std = nested_cv(X, Y, name, config)
        print(f"{name:25s}: {acc_mean:.4f} ± {acc_std:.4f}")


if __name__ == "__main__":
    main()
