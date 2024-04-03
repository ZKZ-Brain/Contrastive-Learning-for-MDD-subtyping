"""Treatment effect regression of rTMS dataset"""
import numpy as np
import statistics
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.feature_selection import SelectKBest, f_regression

"Load rTMS table"
path1='../b_data/3_rTMS_table.csv'
csv = pd.read_csv(path1,header=0,encoding='gbk')
Index=csv['HAMD-30d']       # dependent/regression variable

"Select model and feature selection method"
path2='../d_results_of_z/latent_z_rTMS512.npz'
latent_z=np.load(path2)
encoder='Z_cvae_sencoder'
data=latent_z[encoder][0,:]
model = SVR(kernel='rbf', C=10)
selector = SelectKBest(score_func=f_regression, k=175)
y=Index.to_numpy()
X = selector.fit_transform(data,y)

# "Grid search"
# param_grid = {'C': np.linspace(0.1,15,100)}
# cv_strategy = RepeatedKFold(n_splits=5, n_repeats=3)  # cross-validation settings
# grid_search = GridSearchCV(model, param_grid, cv=cv_strategy, scoring='neg_mean_absolute_error')
# grid_search.fit(X, y)
# print("best parameters is:", grid_search.best_params_)
# print("best score is:", grid_search.best_score_)

"Regression of Index"
kf = RepeatedKFold(n_splits=5, n_repeats=100)
test_y_list,test_y_list_pred = np.empty((0)),np.empty((0))

for train_index, test_index in kf.split(X):
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]

    model.fit(train_X, train_y)

    train_y_pred = model.predict(train_X)
    tr_rmse = np.sqrt(np.mean((train_y - train_y_pred) ** 2))
    tr_mae = np.mean(np.abs(train_y - train_y_pred))
    tr_pcc=np.corrcoef(train_y,train_y_pred)[0,1]

    test_y_pred = model.predict(test_X)

    test_y_list = np.append(test_y_list, test_y, axis=0)                # test_y_list: save true labels
    test_y_list_pred=np.append(test_y_list_pred, test_y_pred, axis=0)   # test_y_list_pred: save predicted labels

te_mae = np.mean(np.abs(test_y_list  - test_y_list_pred))
te_rmse = np.sqrt(np.mean((test_y_list  - test_y_list_pred) ** 2))
te_pcc = np.corrcoef(test_y_list ,test_y_list_pred)[0,1]
print("MAE: {:.2f}, RMSE: {:.2f}, PCC: {:.2f}".format(te_mae, te_rmse, te_pcc))

"Calculate mean and var every 26 labels"
span=26
test_mae,test_rmse,test_pcc = [],[],[]
for i in range(0, len(test_y_list), span):
    temp1=test_y_list[i:i+span].astype(float)
    temp2=test_y_list_pred[i:i+span].astype(float)
    te_mae = np.mean(np.abs(temp1 - temp2))
    te_rmse = np.sqrt(np.mean((temp1 - temp2) ** 2))
    te_pcc = np.corrcoef(temp1, temp2)[0, 1]

    test_mae.append(te_mae)
    test_rmse.append(te_rmse)
    test_pcc.append(te_pcc)

mean_mae = statistics.mean(test_mae)
mean_rmse = statistics.mean(test_rmse)
mean_pcc = statistics.mean(test_pcc)

variance_mae = statistics.variance(test_mae)
variance_rmse = statistics.variance(test_rmse)
variance_pcc = statistics.variance(test_pcc)
print("mean_MAE: {:.4f}, mean_RMSE: {:.4f}, mean_PCC: {:.4f}".format(mean_mae, mean_rmse, mean_pcc))
print("var_MAE: {:.4f}, var_RMSE: {:.4f}, var_PCC: {:.4f}".format(variance_mae, variance_rmse, variance_pcc))

