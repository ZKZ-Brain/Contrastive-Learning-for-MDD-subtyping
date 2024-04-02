"""Load and select dataset"""
import numpy as np
import pandas as pd
from h_auxilary_functions import dataset_choose

choose=dataset_choose(dataset="testset") #dataset: "trainset":REST-II, "testset":REST-I or "rTMSset":SAINT

if choose=="trainset":
    print("Using train dataset (3000)")

    path='../b_data/S3000.npz'
    raw_data= np.load(path,allow_pickle=True)
    Data = raw_data['data']  # Data of nii, size:3000*64*64*64
    ID_subs = raw_data['subs']  # Data of BIDS_ID, size:3000
    nsubs = Data.shape[0]    # Number of subjects

    # load table data
    path_csv='../b_data/1_trainset_table.csv'
    csv = pd.read_csv(path_csv,header=0,encoding='gbk')
    csv.reset_index(inplace=True)

    assert len(csv) == len(ID_subs), 'lenght mismatch between nii and table'
    assert all([csv['BIDS_ID'][s] == ID_subs[s] for s in range(len(csv))]), 'order mismatch between nii and table'

    # seperate subgroup by DxGroup
    patients = csv['DxGroup'].values==1   #size:1660
    controls = csv['DxGroup'].values==2   #size:1340

    HC_subs = Data[controls,:,:,:] # Data of Typically Developing participants, size=1340
    MDD_subs = Data[patients,:,:,:] # Data of MDD participants, size=1660

elif choose=="testset":
    print("Using test dataset (2380)")

    path='../b_data/S2380_test.npz'
    raw_data= np.load(path,allow_pickle=True)
    Data = raw_data['data']  # Data of all people,size:2380*64*64*64
    ID_subs = raw_data['subs']  # Data of BIDS_ID,size:2380
    nsubs = Data.shape[0]

    # load table data
    path_csv='../b_data/2_testset_table.csv'
    csv = pd.read_csv(path_csv,header=0,encoding='gbk')
    csv.reset_index(inplace=True)

    assert len(csv) == len(ID_subs), 'lenght mismatch between nii and table'
    assert all([csv['BIDS_ID'][s] == ID_subs[s] for s in range(len(csv))]), 'order mismatch between nii and table'

    patients = csv['DxGroup'].values==1   #size:1276
    controls = csv['DxGroup'].values==2   #size:1104

    HC_subs = Data[controls,:,:,:] # Data of Typically Developing participants, size=1104
    MDD_subs = Data[patients,:,:,:] # Data of MDD participants, size=1276

elif choose=="rTMSset":
    print("Using rTMS dataset")

    path='../b_data/S3000.npz'
    raw_data= np.load(path,allow_pickle=True)
    Data_total = raw_data['data']  #size:3000*64*64*64
    ID_subs = raw_data['subs']
    nsubs = Data_total.shape[0]

    # load table data
    path_csv='../b_data/1_trainset_table.csv'
    csv = pd.read_csv(path_csv,header=0,encoding='gbk')
    csv.reset_index(inplace=True)

    assert len(csv) == len(ID_subs), 'lenght mismatch between nii and table'
    assert all([csv['BIDS_ID'][s] == ID_subs[s] for s in range(len(csv))]), 'order mismatch between nii and table'

    # combine subgroup
    controls = csv['DxGroup'].values==2   #size:1340
    HC_subs = Data_total[controls,:,:,:]

    rTMS_data= np.load('../b_data/S26_rTMS.npz',allow_pickle=True)
    MDD_subs = rTMS_data['data']      # Data of patients,size:26*64*64*64
    Data=rTMS_data['data']

else:
    raise ValueError("Invalid dataset")