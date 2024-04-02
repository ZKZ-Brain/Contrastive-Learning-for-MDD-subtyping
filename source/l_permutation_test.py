"""Permutation test-validated for LR classification"""
import numpy as np
import pandas as pd
import statistics
from sklearn.svm import SVR
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, f_regression

"Load whole data of train set"
result_path= '../d_results_of_z/latent_z_train512.npz'
encoder='Z_cvae_sencoder'
latent_z=np.load(result_path)
data=latent_z[encoder][1,:]  # latent Z, size: 3000*512

path1='../b_data/1_trainset_table.csv'
csv = pd.read_csv(path1,header=0,encoding='gbk')
ID_total=csv['ID']  # whole ID, size: 3000*1

"Load medical table"
path2='../b_data/SVM_398.csv'
csv_medical = pd.read_csv(path2,header=0, encoding='gbk')
ID_medical=csv_medical['ID']

# location: save the position of the medical ID in the original ID.
locations=[]
for item in ID_medical:
    location=np.where(ID_total == item)[0]
    locations.extend(location)

# save latent z corrosponding to medical ID
data_meidcal=np.empty((len(locations),512))
for row,location in enumerate(locations):
    data_meidcal[row,:]=data[location,:]

y_total=csv_medical['TreatmentResponsive'].replace(-1, 0).values   #read treatment label, replace -1 with 0

"Drug grouping"
MedicationName=csv_medical['MedicationName']
group1_location,group2_location=[],[]
group2_name=['Sertraline','Fluoxetine','Paroxetine','Escitalopram','Fluvoxamine','Citalopram']  #SSRI
for idx,name in enumerate(MedicationName):
    if name == 'Medication-free':
        group1_location.append(idx)
    elif name in group2_name:
        group2_location.append(idx)

#select input group
group=group1_location
X=data_meidcal[group,:]
y=y_total[group]

# X=data_meidcal  #select all
# y=y_total

"Select model and feature selection method"
N=175
model = LogisticRegression(C=1, solver='liblinear')
selector = RFE(model, n_features_to_select=N)
X = selector.fit_transform(X,y)

"Permutation test"
kf = RepeatedKFold(n_splits=10, n_repeats=100)
permu_num=0
base_acc=[0.89,0.98,0.95]  #reported acc
for i in range(1000):
    np.random.shuffle(y)  # shuffle y
    permu_X,permu_y=X,y
    num,acc_accum = 0,0
    for train_index, test_index in kf.split(permu_X):
        num = num + 1
        train_X, train_y = permu_X[train_index], permu_y[train_index]
        test_X, test_y = permu_X[test_index], permu_y[test_index]

        model.fit(train_X, train_y)
        test_pred = model.predict(test_X)
        test_accuracy = accuracy_score(test_y, test_pred)

        acc_accum = acc_accum + test_accuracy

    if acc_accum/num > base_acc[0]:
        permu_num+=1
    print(i,permu_num)
print("permutation test result is",permu_num/1000)

#############################################
"""Permutation test-validated for rTMS regression"""

"Load rTMS table"
path1='../b_data/3_rTMS_table.csv'
path2= '../d_results_of_z/latent_z_rTMS512.npz'

csv = pd.read_csv(path1,header=0,encoding='gbk')
Index=csv['BDI-30d']       # dependent/regression variable

encoder='Z_cvae_sencoder'
latent_z=np.load(path2)
data=latent_z[encoder][1,:]

"Select model and feature selection method"
model = SVR(kernel='rbf', C=10)
selector = SelectKBest(score_func=f_regression, k=175)
y=Index.to_numpy()
X = selector.fit_transform(data,y)

"Permutation test-validated "
kf = RepeatedKFold(n_splits=5, n_repeats=100)
permu_num=0
base_pcc=[0.80,0.82,0.85,0.76,0.65,0.71,0.83,0.73,0.83] # reported pcc
for j in range(1000):
    np.random.shuffle(y)  # shuffle y
    permu_X,permu_y=X,y

    test_y_list, test_y_list_pred = np.empty((0)), np.empty((0))
    for train_index, test_index in kf.split(permu_X):
        train_X, train_y = permu_X[train_index],permu_y[train_index]
        test_X, test_y = permu_X[test_index],permu_y[test_index]

        model.fit(train_X, train_y)
        test_y_pred = model.predict(test_X)

        test_y_list = np.append(test_y_list, test_y, axis=0)
        test_y_list_pred = np.append(test_y_list_pred, test_y_pred, axis=0)

    span,test_pcc=26,[]
    for i in range(0, len(test_y_list), span):
        temp1 = test_y_list[i:i + span].astype(float)
        temp2 = test_y_list_pred[i:i + span].astype(float)
        te_pcc = np.corrcoef(temp1, temp2)[0, 1]
        test_pcc.append(te_pcc)
    permu_pcc = statistics.mean(test_pcc)
    if permu_pcc > base_pcc[0]:
        permu_num+=1
    print(j,permu_num)
print("permutation test result is",permu_num/1000)