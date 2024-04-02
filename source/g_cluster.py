"""MDD cluster based on feature Z"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from c_load_data import patients

"K-means cluster"
result_path= '../d_results_of_z/latent_z_train.npz'
encoder='Z_cvae_sencoder'
latent_z=np.load(result_path)
n_clusters = 2
data=latent_z[encoder][1,:][patients,:]
kmeans = KMeans(n_clusters=n_clusters,n_init='auto')
kmeans.fit(data)
label=kmeans.labels_

"Plot K-means result"
plt.figure(figsize=(6,6))
color1=(0.6,0.6,0.6)
color2=(0.35,0.6,0.6)
colors = [color1, color2]
handles=[]
for id, label in enumerate(kmeans.labels_):
    lab="subgroup1" if label==1 else "subgroup2"
    handle=plt.scatter(data[id][0], data[id][1], color = colors[label],marker='o', s=15)

ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(3) # bold border

plt.xlabel('X',fontsize=25,fontweight='bold')
plt.ylabel("Y",fontsize=25,fontweight='bold')

plt.xticks([-0.5,0,0.5],fontsize=20, fontweight='bold')
plt.yticks([-1.0,0,1.0],fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig("Cluster.png", dpi=600)
plt.show()

"Plot average distortion in elbow method"
plt.figure(figsize=(4,5))
C=range(1,10)
sse_result=[]
for k in C:
    kmeans=KMeans(n_clusters=k,n_init='auto')
    kmeans.fit(data)
    sse_result.append(sum(np.min(cdist(data,kmeans.cluster_centers_,'euclidean'),axis=1))/data.shape[0])
plt.plot(C,sse_result,'k-',linewidth=3)

colors = ['black', 'red', 'black', 'black', 'black','black', 'black', 'black', 'black']
sizes=[60,80,60,60,60,60,60,60,60]
plt.scatter(C, sse_result, marker='^',c=colors,s=sizes) # plot scatter point

ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(3)

plt.xlabel('C',fontsize=25,fontweight='bold')
plt.ylabel("Average Distortion",fontsize=25,fontweight='bold')

plt.xticks([2,5,10],fontsize=20, fontweight='bold')
plt.yticks([1.5,2.0,2.5],fontsize=20, fontweight='bold')
plt.tight_layout()
fig_size = plt.gcf().get_size_inches()
plt.savefig("elbow.png", dpi=600)
plt.show()
