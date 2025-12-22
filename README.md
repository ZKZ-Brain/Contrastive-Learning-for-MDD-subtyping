# Brain-CVAE

This repository provides the implementation code for the paper:  
**"Brain Contrastive Modeling Reveals Depression Subtypes with Distinct Treatment Response and Progression"**

## Pipeline

```
Step 1: Data Preprocessing
        format_conversion.py
        (NIfTI -> NPZ)
              |
              v
Step 2: Model Training
        train_CVAE.py
        (Train Brain-CVAE)
              |
              v
Step 3: Feature Extraction
        extract_latent_feature.py
        (Extract latent Z)
              |
    +---------+---------+------------+------------------------+
    |         |         |            |                        |
    v         v         v            v                        v
Step 4: Downstream Analysis
  RSA.py  cluster.py  medical_     rTMS_             symptom_network_analysis.R
  (RSA)   (Subtype)   classifi-    regression.py        (Symptom Network)   
                      cation.py    (Prediction)
```

## Project Structure

```
.
|-- a_raw_data/                       # Raw fMRI image data (.nii format)
|   |-- fALFF_FunImgARCW_2/           # 10 sample subjects from training set
|   |-- fALFF_FunVoloWC_1/            # 10 sample subjects from test set
|   |-- rTMS/                         # 10 sample subjects from SAINT dataset
|   |-- Xi_jing_KG_QG/                # 10 sample subjects from XJ_KG and XJ_QG dataset
|   +-- Xiangya/                      # 10 sample subjects from Xiangya dataset
|
|-- b_data/                           # Statistical tables and preprocessed data
|   |-- total_table.xlsx              # Consolidated metadata and labels
|   |-- S2898.npz                     # Preprocessed training set
|   |-- S2336_test.npz                # Preprocessed test set
|   |-- S26_rTMS.npz                  # Preprocessed rTMS set
|   |-- S60_xiangya.npz               # Preprocessed Xiangya dataset
|   |-- S73_XJ_KG.npz                 # Preprocessed XJ_KG dataset
|   +-- S44_XJ_QG.npz                 # Preprocessed XJ_QG dataset
|
|-- c_torch_weight/                   # Saved model weights
|   |-- model_best_train.pth          # Weights trained on training set
|   |-- model_best_test.pth           # Weights trained on test set
|   +-- model_best_512.pth            # Weights with 512-dim latent output
|
|-- d_results_of_z/                   # Extracted latent features for all datasets
|   +-- latent_z_train.npz
|
|-- e_prognosis/                      # Prognosis analysis data
|   |-- drug_data_512.npz             # 512-dim latent Z for medication samples
|   |-- rTMS_data_512.npz             # 512-dim latent Z for rTMS samples
|   +-- table_7_medication.csv        # Medication group metadata
|
|-- results/                          # Analysis results
|   |-- pattern_figure/               # Brain pattern visualization
|   |   |-- sub1_pattern.nii          # Subtype 1 brain pattern
|   |   +-- sub2_pattern.nii          # Subtype 2 brain pattern
|   +-- HYDRA_results_Cluster_is_3.mat  # HYDRA clustering results
|
|-- environment.yaml                  # Replicate analysis environment
|
+-- code/                             # Source code
    |-- 1 Data Preprocessing/        
    |   +-- format_conversion.py      # Convert NIfTI (.nii) files to compressed NPZ format
    |-- 2 Model Training/
    |   |-- b_model.py                # Brain-CVAE model architecture (VAE & CVAE classes)
    |   |-- HSIC.py                   # Hilbert-Schmidt Independence Criterion for disentanglement
    |   |-- load_data.py              # Dataset loading and preprocessing utilities
    |   |-- auxiliary_functions.py    # Helper functions (RDM calculation, metrics, visualization)
    |   +-- train_CVAE.py             # Main training script with contrastive learning
    |-- 3 Feature Extraction/
    |   +-- extract_latent_feature.py # Extract latent representations from trained model
    +-- 4 Downstream Analysis/
        |-- RSA.py                    # Representational Similarity Analysis
        |-- cluster.py                # K-Means clustering with automatic K selection
        |-- cluster_comparison.py     # Comparison of three clustering methods
        |-- medical_classification.py # Treatment response classification (6 ML models)
        |-- rTMS_regression.py        # rTMS outcome prediction (8 regression models)
        +-- symptom_network_analysis.R# Symptom network analysis for two subtypes
```

## Generated Outputs Summary

| Script | Output File | Description |
|--------|-------------|-------------|
| `RSA.py` | `RSA_results.png` | RSA bar plots for clinical variables |
| `cluster.py` | `elbow_auto.png` | Elbow method for cluster selection |
| `cluster.py` | `Cluster_auto.png` | t-SNE cluster visualization |
| `medical_classification.py` | `console output` | Classification accuracy for each model |
| `rTMS_regression.py` | `model_pca_comparison_heatmap.png` | Regression performance heatmap |
| `rTMS_regression.py` | `model_comparison_results.csv` | Detailed regression results |
| `symptom_network_analysis.R` | `network_comparison.png` | Subtype network comparison |
| `symptom_network_analysis.R` | `betweenness_comparison.png` | Subtype betweenness comparison |

## External Tools

For additional analyses mentioned in the paper:

- **HYDRA** (Heterogeneity through Discriminative Analysis): https://github.com/evarol/HYDRA
- **SuStaIn** (Subtype and Stage Inference): https://github.com/ucl-pond/pySuStaIn

## License

This project is provided for research purposes. Please contact the corresponding author for data access requests.
