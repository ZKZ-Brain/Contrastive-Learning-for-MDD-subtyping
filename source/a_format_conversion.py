"""Format conversion of three datasets (nii to npz)"""
import numpy as np
import pandas as pd
import os
from h_auxilary_functions import load_nifti_files


"train data conversion(3000)"
raw_data_path="../a_raw_data/fALFF_FunVoloWC_1"
file_names = os.listdir(raw_data_path)

for _ in range(2):  #remove two suffix
    file_names = [os.path.splitext(file_name)[0] for file_name in file_names]
split_names = [name.split('_')[1:] for name in file_names] #remove preffix
split_names = list(map(lambda x: x[0], split_names))       #remove bracket

ID_file=[str(item) for item in split_names]                #list to str

#read csv
path1="../a_raw_data/table_1_3000.csv"
csv_ours=pd.read_csv(path1, header=0,encoding='gbk')

csv_ours = csv_ours.dropna(axis=0, how='all')  #remove null string
ID_csv=csv_ours["ID"].tolist()                 #seies to list
BIDS_ID =np.squeeze(np.array([csv_ours['BIDS_ID']]))  #read BIDS_ID

#Compare the differences between the table and the fMRI file(IS005-2-0057~0059)
difference = list(set(ID_file) - set(ID_csv))
print(difference)

data_array = load_nifti_files(raw_data_path)

#save as compressed files (.npz)
save_path="../a_raw_data"
np.savez_compressed(save_path+"/S3000.npz", data=data_array, subs=BIDS_ID)


"test data conversion(2380)"
raw_data_path="../a_raw_data/fALFF_FunImgARCW_2"
file_names = os.listdir(raw_data_path)

for _ in range(2):      #remove two suffix
    file_names = [os.path.splitext(file_name)[0] for file_name in file_names]
split_names = [name.split('_')[1:] for name in file_names]
split_names = list(map(lambda x: x[0], split_names))

ID_file=[str(item) for item in split_names]

#read csv
path2="../a_raw_data/table_2_2380.csv"
csv_ours=pd.read_csv(path2, header=0,encoding='gbk')

csv_ours = csv_ours.dropna(axis=0, how='all')
ID_csv=csv_ours["ID_for_preprocessing"].tolist()
BIDS_ID =np.squeeze(np.array([csv_ours['BIDS_ID']]))
difference = list(set(ID_file) - set(ID_csv))
print(difference)

data_array = load_nifti_files(raw_data_path)

# save as compressed files (.npz)
save_path="../a_raw_data"
np.savez_compressed(save_path+"/S2380_test.npz", data=data_array, subs=BIDS_ID)


"rTMS data conversion"
raw_data_path="../a_raw_data/rTMS"
file_names = os.listdir(raw_data_path)

data_array = load_nifti_files(raw_data_path)

# save as compressed files (.npz)
save_path="../a_raw_data"
np.savez_compressed(save_path+"/S26_rTMS.npz", data=data_array)
