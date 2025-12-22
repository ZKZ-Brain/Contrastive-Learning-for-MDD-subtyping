"""Load and select dataset"""
import numpy as np
import pandas as pd
from auxiliary_functions import dataset_choose

choose=dataset_choose(dataset="trainset") #dataset: "trainset":REST-II, "testset":REST-I or "rTMSset":SAINT

if choose=="trainset":
    print("Using train dataset (2898)")

    path='../b_data/S2898.npz'
    raw_data= np.load(path,allow_pickle=True)
    Data = raw_data['data']  # Data of nii, size:2898*64*64*64
    ID_subs = raw_data['subs']  # Data of BIDS_ID, size:2898
    nsubs = Data.shape[0]    # Number of subjects

    # load table data
    path_csv = "../b_data/total_table.xlsx" 
    csv = pd.read_excel(path_csv, sheet_name="table_1_2898", header=0)
    csv.reset_index(inplace=True)

    assert len(csv) == len(ID_subs), 'lenght mismatch between nii and table'
    assert all([csv['BIDS_ID'][s] == ID_subs[s] for s in range(len(csv))]), 'order mismatch between nii and table'

    # seperate subgroup by DxGroup
    patients = csv['DxGroup'].values==1   #size:1590
    controls = csv['DxGroup'].values==2   #size:1308

    HC_subs = Data[controls,:,:,:] # Data of Typically Developing participants, size=1308
    MDD_subs = Data[patients,:,:,:] # Data of MDD participants, size=1590

elif choose=="testset":
    print("Using test dataset (2336)")

    path='../b_data/S2336_test.npz'
    raw_data= np.load(path,allow_pickle=True)
    Data = raw_data['data']  # Data of all people,size:2336*64*64*64
    ID_subs = raw_data['subs']  # Data of BIDS_ID,size:2336
    nsubs = Data.shape[0]

    # load table data
    path_csv = "../b_data/total_table.xlsx"
    csv = pd.read_excel(path_csv, sheet_name="table_2_2336", header=0)
    csv.reset_index(inplace=True)

    assert len(csv) == len(ID_subs), 'lenght mismatch between nii and table'
    assert all([csv['BIDS_ID'][s] == ID_subs[s] for s in range(len(csv))]), 'order mismatch between nii and table'

    patients = csv['DxGroup'].values==1   #size:1248
    controls = csv['DxGroup'].values==2   #size:1088

    HC_subs = Data[controls,:,:,:] # Data of Typically Developing participants, size=1088
    MDD_subs = Data[patients,:,:,:] # Data of MDD participants, size=1248

elif choose == "rTMSset":
    print("Using rTMS dataset")
    raw_data = np.load('../b_data/S26_rTMS.npz', allow_pickle=True)
    Data_total = raw_data['data']  # size: 26*64*64*64
    ID_subs = raw_data['subs']
    csv = pd.read_excel("../b_data/total_table.xlsx", sheet_name="table_3_rTMS")

    assert len(csv) == len(ID_subs), 'Length mismatch'
    assert (csv['BIDS_ID'].values == ID_subs).all(), 'Order mismatch'

    HC_subs = Data_total[csv['DxGroup'] == 2]

    Data = Data_total
    patients = np.ones(len(Data), dtype=bool)