"""TreatmentResponsive classification"""
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from h_auxilary_functions import calc_performance_statistics

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

# select input group
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

# "Grid search"
# param_grid = {'C': np.linspace(0.1,10,100)}
# cv_strategy = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)  # cross-validation settings
# grid_search = GridSearchCV(model, param_grid, cv=cv_strategy, scoring='accuracy')
# grid_search.fit(X, y)
# print("best parameters is:", grid_search.best_params_)
# print("best score is:", grid_search.best_score_)

"Calculate mean and var of acc,auc.."
i,num=0,0
acc_accum, auc_accum, sen_accum, spc_accum, prc_accum, f1s_accum, mcc_accum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
acc_var,auc_var,sen_var,spc_var,prc_var,f1s_var,mcc_var=[],[],[],[],[],[],[]

kf = RepeatedKFold(n_splits=10, n_repeats=100)

for train_index, test_index in kf.split(X):
    num=num+1
    np.random.shuffle(train_index)

    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]

    model.fit(train_X,train_y)
    train_pred = model.predict(train_X)
    train_accuracy = accuracy_score(train_y,train_pred)

    test_pred = model.predict(test_X)
    test_accuracy = accuracy_score(test_y,test_pred)

    acc, auc, sen, spc, prc, f1s, mcc = calc_performance_statistics(test_y,test_pred)

    acc_var.append(acc) #calculate var
    auc_var.append(auc)
    sen_var.append(sen)
    spc_var.append(spc)
    prc_var.append(prc)
    f1s_var.append(f1s)
    mcc_var.append(mcc)

    acc_accum = acc_accum+acc #calculate mean
    auc_accum = auc_accum+auc
    sen_accum = sen_accum+sen
    spc_accum = spc_accum+spc
    prc_accum = prc_accum+prc
    f1s_accum = f1s_accum+f1s
    mcc_accum = mcc_accum+mcc

non_zero_auc=[x for x in auc_var if x !=0]
mean_auc=sum(non_zero_auc) / (len(non_zero_auc)+0.00001)

non_zero_mcc=[x for x in mcc_var if x !=0]
mean_mcc=sum(non_zero_mcc) / (len(non_zero_mcc)+0.00001)

print("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(np.var(acc_var), np.var(non_zero_auc),np.var(sen_var),np.var(spc_var),np.var(f1s_var),np.var(non_zero_mcc)))
print("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(acc_accum/num,mean_auc,sen_accum/num,spc_accum/num,f1s_accum/num,mean_mcc))

