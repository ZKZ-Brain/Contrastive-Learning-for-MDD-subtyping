"""Evaluation of Clustering Agreement and Performance Metrics"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score, fowlkes_mallows_score, adjusted_rand_score

file_path = r"C:\Users\10945\Desktop\Cluster_total.xlsx"
sheet_name = "Total"

df = pd.read_excel(file_path, sheet_name=sheet_name)
df_clean = df.dropna()

# Extract clustering labels
labels_base = df_clean["base_Cluster"].astype(int)
labels_kmeans = df_clean["K-Means"].astype(int)
labels_hydra = df_clean["HYDRA"].astype(int)
labels_sustain = df_clean["Sustain"].astype(int)

group_true = labels_base
group_pred = labels_sustain

# 1. Calculate Cohen's Kappa
kappa = cohen_kappa_score(group_true, group_pred)
print(f"Cohen's Kappa: {kappa:.4f}")

# 2. Calculate Fowlkes-Mallows Index (FMI)
fmi_value = fowlkes_mallows_score(group_true, group_pred)
print(f"Fowlkes-Mallows Index (FMI): {fmi_value:.4f}")

# 3. Calculate Adjusted Rand Index (ARI)
ari_value = adjusted_rand_score(group_true, group_pred)
print(f"Adjusted Rand Index (ARI): {ari_value:.4f}")