"""Auxiliary functions"""
import os
import numpy as np
import torch
import scipy
import pandas as pd
import seaborn as sns
import nibabel as nib
from scipy.ndimage import zoom
from scipy import stats
from scipy.spatial.distance import squareform,pdist
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind as ttest
from sklearn.metrics import confusion_matrix, roc_auc_score

#read nii files
def load_nifti_files(path):
    nifti_files = [file for file in os.listdir(path) if file.endswith('.nii') or file.endswith('.nii.gz')]

    data = []   #save all nii file
    for file in nifti_files:
        file_path = os.path.join(path, file)
        nifti_data = nib.load(file_path).get_fdata()
        target_shape = (64, 64, 64)  #change resolution
        resized_data = zoom(nifti_data, target_shape / np.array(nifti_data.shape), order=0)
        retype_data=resized_data.astype(np.float32)
        data.append(retype_data)
    return np.array(data)

def dataset_choose(dataset):
    valid_strings = ['trainset','testset','rTMSset']
    if dataset in valid_strings:
        return dataset
    else:
        raise ValueError("Invalid selection")

def save_weight(path,model_state,curr_loss,loss_list):
    if not os.path.exists(path):
        os.makedirs(path)

    if curr_loss <= min(loss_list):
        best_file=os.path.join(path,'model_best.pth')
        torch.save(model_state,best_file)

def save_z(path,z1,z2):
    if not os.path.exists(path):
        os.makedirs(path)
    np.savez_compressed(path+'/latent_z.npz',Z_cvae_encoder=z1,Z_cvae_sencoder=z2)

# calculate the Euclidean distance for all points in the data , return matrix with size subjects*subjects
def data2cmat(data):
    return np.array([squareform(pdist(data[s,:,:],metric='euclidean')) for s in range(data.shape[0])])

# calculate RDM matrix of inVec
def calculate_RDM(inVec, data_scale='ratio', metric='euclidean'):
    vec = inVec
    vec = (vec - min(vec.flatten())) / (max(vec.flatten()) - min(vec.flatten()))

    if np.ndim(inVec) == 1:  # must be at least 2D
        vec = np.vstack((vec, np.zeros(vec.shape))).transpose()

    mat = squareform(pdist(vec, metric=metric).transpose())
    if data_scale == 'categorical':
        mat[mat != 0] = 1  # make into zeros and ones

    return mat

def RDM_to_vec(RDM):
    assert np.ndim(RDM)==2, 'error: not 2 dim'
    assert RDM.shape[0]==RDM.shape[1], 'error: not a square'

    n = RDM.shape[0]
    vec = RDM[np.triu_indices(n=n,k=1)]  # k: offset
    return vec


def slice_cmat(z,idx,patients):
    mat = z[patients,:][:,patients]  # choose patients, matrix size:patients*patients
    mat = mat[idx,:][:,idx]          # choose data if idx == True
    return mat

# calculate RDM difference between Z and Keys
def fit_rsa(z,key,model_idxs,model_rdms,patients,n_sample=10):
    corr = scipy.stats.kendalltau   # kendalltau as correlation measure
    r = np.array([corr(RDM_to_vec(slice_cmat(z[i,:,:],model_idxs[key],patients)),RDM_to_vec(model_rdms[key]))[0] for i in range(n_sample)])
    r = np.arctan(r)
    return r

# calculate index of LR regression
def calc_performance_statistics(y_pred, y):
    TN, FP, FN, TP = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    N = TN + TP + FN + FP
    S = (TP + FN) / N
    P = (TP + FP) / N
    acc = (TN + TP) / N
    sen = TP / (TP + FN)
    spc = TN / (TN + FP)
    prc = TP / (TP + FP)
    f1s = 2 * (prc * sen) / (prc + sen)
    try:
        mcc = (TP / N - S * P) / np.sqrt(P * S * (1 - S) * (1 - P))
        auc = roc_auc_score(y,y_pred)
    except ValueError:
        mcc=0
        auc=0
    return acc, auc, sen, spc, prc, f1s, mcc

# loss plot function
def plot_loss(path):
    csv = pd.read_csv(path,header=0,encoding='gbk')
    step=csv['Step']
    Value=csv['Value']

    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters',fontsize=15,fontweight='bold')
    plt.ylabel('Loss',fontsize=15,fontweight='bold')

    plt.xticks(fontsize=12,fontweight='bold')
    plt.yticks(fontsize=12,fontweight='bold')

    ax = plt.gca()
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    plt.plot(step, Value, linewidth=2, linestyle="solid")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Loss_curve.png", dpi=600)
    plt.show()


# plot the distribution of keys (Age, Education,HAMD-17)
def plot_distribution(path,title,color):
    csv = pd.read_csv(path, header=0, encoding='gbk')

    keys = ["Age", "Education", "HAMD-17"]
    for item in keys:
        x = csv[item].dropna().astype(int)  # delete NaN, float to int

        plt.hist(x, bins=range(min(x), max(x) + 1), color=color, edgecolor='black', alpha=0.75)

        plt.xticks([0, 20, 40, 60, 70], fontsize=12, fontweight='bold')
        plt.xticks([0,10,20,30,40,50], fontsize=12, fontweight='bold')
        plt.yticks(fontsize=12, fontweight='bold')

        ax = plt.gca()
        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(3)

        plt.title('Distribution on {}'.format(title), fontsize=18, fontweight='bold')
        plt.xlabel('{}'.format(item), fontsize=15, fontweight='bold')
        plt.ylabel('Numbers', fontsize=15, fontweight='bold')

        plt.tight_layout()
        plt.savefig("Distribution of {}.png".format(item), dpi=600)
        plt.show()

# bar plot function
def plot_nice_bar(key, rsa, ax=None, figsize=None, dpi=None, fontsize=None, fontsize_star=None, fontweight=None,
                  line_width=None, marker_size=None, title=None, report_t=False, do_pairwise_stars=False,
                  do_one_sample_stars=True):

    pallete = sns.color_palette()
    pallete_new = sns.color_palette()

    if not figsize:
        figsize = (5, 2)
    if not dpi:
        dpi = 300

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    pallete_new[1] = pallete[0]
    pallete_new[0] = pallete[1]
    pallete_new[0] = tuple(np.array((.5, .5, .5)))

    data = np.array(rsa[key])
    n = data.shape[0]
    c = data.shape[1]
    x = np.arange(c)

    if not fontsize:
        fontsize = 16

    if not fontsize_star:
        fontsize_star = 25
    if not fontweight:
        fontweight = 'bold'
    if not line_width:
        line_width = 2.5
    if not marker_size:
        marker_size = .1

    for i in range(c):
        plot_data = np.zeros(data.shape)
        plot_data[:, i] = data[:, i]

        xs = np.repeat(i, n) + (np.random.rand(n) - .5) * .25
        sc = plt.scatter(xs, data[:, i], c='k', s=marker_size)
        # b = sns.barplot(data=plot_data, err_kws={'color': 'r', 'linewidth': line_width}, linewidth=2.5,
        #                 facecolor=np.hstack((np.array(pallete_new[i]), .3)),
        #                 edgecolor=np.hstack((np.array(pallete_new[i]), 1)))  #remote environment
        b = sns.barplot(data=plot_data, errcolor='r', linewidth=line_width, errwidth=line_width,
                        facecolor=np.hstack((np.array(pallete_new[i]), .3)),
                        edgecolor=np.hstack((np.array(pallete_new[i]), 1)))

    locs, labels = plt.yticks()
    new_y = np.linspace(locs[0], locs[-1], 6)
    plt.yticks(new_y, labels=[f'{yy:.2f}' for yy in new_y], fontsize=fontsize, fontweight=fontweight)
    plt.ylabel('model fit (r)', fontsize=fontsize, fontweight=fontweight)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(line_width)

    xlbls = ['Shared', 'Specific']
    plt.xticks(np.arange(2), labels=xlbls, fontsize=fontsize, fontweight=fontweight)

    if do_one_sample_stars:
        one_sample = np.array([ttest_1samp(data[:, i], 0) for i in range(2)])
        one_sample_thresh = np.array((1, .05, .001, .0001))
        one_sample_stars = np.array(('n.s.', '*', '**', '***'))
        for i in range(c):
            these_stars = one_sample_stars[max(np.nonzero(one_sample[i, 1] < one_sample_thresh)[0])]
            xlbls[i] = f'{xlbls[i]}\n({these_stars})'
        plt.xticks(np.arange(2), labels=xlbls, fontsize=fontsize, fontweight=fontweight, horizontalalignment='center',
                   multialignment='center')

    pairwise_t = np.zeros((3, 3))
    pairwise_p = np.zeros((3, 3))

    pairwise_sample_thresh = np.array((1, .05, .001, .0001))
    pairwise_sample_stars = np.array(('n.s.', '*', '**', '***'))

    if report_t:
        for i in range(c):
            for j in range(c):
                t, p = ttest(data[:, i], data[:, j])
                if p > .001:
                    print(f'{key} {xlbls[i]} >  {xlbls[j]} | t({data.shape[0] - 1}) = {t:.2f} p = {p:.2f}')
                else:
                    print(f'{key} {xlbls[i]} >  {xlbls[j]} | t({data.shape[0] - 1}) = {t:.2f} p $<$ .001')
                pairwise_t[i, j] = t
                pairwise_p[i, j] = p

    comps = [[1, 2]]
    if do_pairwise_stars:
        for comp_idx in range(len(comps)):
            this_comp = comps[comp_idx]
            sig_idx = max(np.nonzero(pairwise_p[this_comp[0], this_comp[1]] < pairwise_sample_thresh)[0])
            max_y = new_y[-1] + comp_idx * .05
            xs = np.array(this_comp)
            stars = pairwise_sample_stars[sig_idx]
            plt.plot(xs, [max_y, max_y], 'k', linewidth=line_width)
            plt.text(xs.mean(), max_y, stars, fontsize=fontsize_star, horizontalalignment='center',
                     fontweight=fontweight)

    ylim = plt.ylim()
    plt.ylim(np.array(ylim) * (1, 1.1))

    if not title:
        plt.title(key, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)
    else:
        plt.title(title, fontsize=fontsize * 1.5, pad=2, fontweight=fontweight)