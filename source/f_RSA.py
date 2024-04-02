"""Representational Similarity Analysis (RSA) analysis"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from c_load_data import csv,patients,choose
from h_auxilary_functions import *

"Calculate RDM matrix of Z"
# load latent Z
result_path= '../d_results_of_z/latent_z.npz'
n_samples=10   # sampling numbers
latent_z=np.load(result_path)

keys = ['Z_cvae_encoder', 'Z_cvae_sencoder']  # key values in save_z function, marking the chosen encoder.
Z_cvae_encoder = latent_z[keys[0]]
Z_cvae_sencoder = latent_z[keys[1]]

cvae_matrix = data2cmat(Z_cvae_encoder)
scvae_matrix = data2cmat(Z_cvae_sencoder)


"Calculate RDM matrix of Keys"
# model_rdms: save all keys' RDM matrixes listed in key_in_csv.
# model_idxs: boolean list, idx: True & NaN: False
model_rdms = dict()
model_idxs = dict()

if choose=="trainset":
    key_in_csv = ['Sex','Age','Education','FirstEpisode','OnMedication','TreatmentResponsive',
                  'FamilyHistory','HAMD-17','HAMA']
    type_of_key = ['categorical','ratio','ratio','categorical','categorical','categorical',
                   'categorical','ratio','ratio']
    # types of keys. categorical: keys with bool value; ratio: keys with int value
    csv['Education'] = pd.to_numeric(csv['Education'], errors='coerce')  # obj to float64

elif choose=="testset":
    key_in_csv = ['Sex','Age','Education','FirstEpisode','OnMedication','CurrentEpisodeDuration',
                  'HAMD-17','HAMA']
    type_of_key = ['categorical','ratio','ratio','categorical','categorical','ratio',
                   'ratio','ratio']
    csv['Education'] = pd.to_numeric(csv['Education'], errors='coerce')
else:
    raise ValueError("Invalid selection")

for i in range(len(key_in_csv)):
    inVec = csv[key_in_csv[i]].values[patients]
    idx = ~np.isnan(inVec) # boolean index
    inVec = inVec[idx]
    this_rdm = calculate_RDM(inVec, data_scale=type_of_key[i])

    model_rdms.update({key_in_csv[i]: this_rdm})
    model_idxs.update({key_in_csv[i]: idx})

"Calculate RDM matrix similarity between Z and Keys"
#rsa_results: save similarity results, size: Keys*n_samples*2
rsa_results = dict()
Z = [cvae_matrix,scvae_matrix]
for key in key_in_csv:
    res = np.array([fit_rsa(z_type,key,model_idxs,model_rdms,patients,n_sample=n_samples) for z_type in Z]).transpose()
    rsa_results.update({key : res})

"Visualization"
plt.figure(figsize=(15,15))
ncols = 3
nrows = int(np.ceil(len(key_in_csv)/3))
plt.figure(figsize=np.array((ncols,nrows))*4)
for i, key in enumerate(key_in_csv):
    ax = plt.subplot(nrows, ncols, i + 1)
    plot_nice_bar(key, rsa_results,
                  ax=ax, figsize=None,
                  dpi=300, fontsize=12,
                  fontsize_star=12,
                  fontweight='bold',
                  line_width=2.5,
                  marker_size=12, title=key_in_csv[i])

plt.subplots_adjust(
    left=None,
    bottom=None,
    right=None,
    top=None,
    wspace=.5,
    hspace=.5)

plt.suptitle('RESULTS', fontsize=20, y=.95)
plt.show()