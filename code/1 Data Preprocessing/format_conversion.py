"""Format Conversion Module"""
import os
import numpy as np
import pandas as pd
from auxiliary_functions import load_nifti_files


def main():
    # Define paths
    raw_data_path = "../a_raw_data/fALFF_FunVoloWC_1"
    metadata_path = "../b_data/total_table.xlsx"
    save_path = "../a_raw_data"

    # Extract subject IDs from filenames
    file_names = os.listdir(raw_data_path)
    for _ in range(2):  # Remove two suffixes (.nii.gz -> .nii -> filename)
        file_names = [os.path.splitext(f)[0] for f in file_names]
    
    # Extract subject ID component from filenames
    subject_ids = [name.split('_')[1:] for name in file_names]
    subject_ids = [item[0] for item in subject_ids]
    id_from_files = [str(item) for item in subject_ids]

    # Load and validate metadata
    csv_data = pd.read_excel(metadata_path, sheet_name="table_1_2898", header=0)
    csv_data = csv_data.dropna(axis=0, how='all')
    
    id_from_csv = csv_data["ID"].tolist()
    bids_ids = np.squeeze(np.array([csv_data['BIDS_ID']]))

    # Check for discrepancies between file IDs and metadata IDs
    difference = list(set(id_from_files) - set(id_from_csv))
    if difference:
        print(f"IDs in files but not in metadata: {sorted(difference)}")

    # Load NIfTI data and save as compressed format
    data_array = load_nifti_files(raw_data_path)
    np.savez_compressed(f"{save_path}/S2898.npz", data=data_array, subs=bids_ids)
    print(f"Data saved to {save_path}/S2898.npz")


if __name__ == "__main__":
    main()
