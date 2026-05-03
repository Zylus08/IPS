import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer

# ==========================
# GLOBAL STYLE
# ==========================
mpl.rcParams.update({
    "font.size": 22,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

# ==========================
# LOAD DATA
# ==========================
csv_path = "time_series_features_all_48_datasets.csv"
df = pd.read_csv(csv_path, index_col=0)

print(f"Original datasets: {len(df)}")

# ---------------------------------------
# REMOVE CLASSIFICATION DATASETS (.ts)
# ---------------------------------------
df = df[~df.index.str.endswith(".ts")]

print(f"Forecasting datasets retained: {len(df)}")

print("\nMissing values before imputation:")
print(df.isnull().sum())

# ==========================
# KNN IMPUTATION
# ==========================
imputer = KNNImputer(
    n_neighbors=5,
    weights='distance'
)

df_imputed_array = imputer.fit_transform(df)

df_imputed = pd.DataFrame(
    df_imputed_array,
    index=df.index,
    columns=df.columns
)

print(f"\nMissing values after imputation: {df_imputed.isnull().sum().sum()}")

# ==========================
# LOG TRANSFORM ONLY HEAVY FEATURES
# ==========================
heavy_cols = ["Variance", "Energy"]

for col in heavy_cols:
    if col in df_imputed.columns:
        df_imputed[col] = np.log1p(df_imputed[col])

print("\nApplied log transform on Variance + Energy")

# ==========================
# DROP WEAK FEATURES
# ==========================
# ==========================
# DROP WEAK / LOW INFORMATION FEATURES
# ==========================
drop_cols = [
    "Mean",
    "Energy",
    "Peak-to-Peak",
    "Stationarity (ADF Test)",
    "Trend Strength",
    "Seasonality Strength"
]

existing_drop_cols = [
    col for col in drop_cols
    if col in df_imputed.columns
]

df_imputed = df_imputed.drop(
    columns=existing_drop_cols
)

print("\nDropped weak / low-information features:")
print(existing_drop_cols)

print("\nRemaining clustering features:")
print(df_imputed.columns.tolist())

# ==========================
# ROBUST SCALING
# ==========================
scaler = RobustScaler()
scaled_features = scaler.fit_transform(df_imputed)

print("\nRobust scaling complete.")

# ==========================
# FIXED K SELECTION
# ==========================
k = 4   # change to 4 if needed

print(f"\nUsing fixed K = {k}")

# ==========================
# FINAL CLUSTERING
# IMPORTANT: FULL FEATURE SPACE
# ==========================
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=100
)

labels = kmeans.fit_predict(scaled_features)

df_imputed["Cluster"] = labels

# ==========================
# PRINT DATASETS PER CLUSTER
# ==========================
print("\n===== DATASETS PER CLUSTER =====")

for cluster in sorted(df_imputed["Cluster"].unique()):
    print(f"\nCluster {cluster}:")
    
    cluster_datasets = df_imputed[
        df_imputed["Cluster"] == cluster
    ].index.tolist()
    
    for ds in cluster_datasets:
        print(f" - {ds}")

# ==========================
# PCA ONLY FOR VISUALIZATION
# ==========================
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

print("\nPCA explained variance ratio:")
print(pca.explained_variance_ratio_)

# ==========================
# VISUALIZATION
# ==========================
plt.figure(figsize=(18, 11))

# Create mesh grid first
x_min, x_max = principal_components[:, 0].min() - 1, principal_components[:, 0].max() + 1
y_min, y_max = principal_components[:, 1].min() - 1, principal_components[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

# Convert PCA grid points back to original feature space
grid_points_pca = np.c_[xx.ravel(), yy.ravel()]

grid_points_original = pca.inverse_transform(
    grid_points_pca
)

# Use ORIGINAL KMeans model
z = kmeans.predict(
    grid_points_original
)

z = z.reshape(xx.shape)

# Plot decision boundaries
plt.contourf(
    xx,
    yy,
    z,
    alpha=0.25,
    cmap="Set1"
)

# Scatter actual points
scatter = plt.scatter(
    principal_components[:, 0],
    principal_components[:, 1],
    c=labels,
    cmap="tab10",
    s=250,
    edgecolor="black",
    linewidth=1.8
)

# Annotate names
for i, dataset in enumerate(df_imputed.index):
    plt.annotate(
        dataset,
        (
            principal_components[i, 0],
            principal_components[i, 1]
        ),
        fontsize=8
    )

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(f"K-Means Clustering (K={k}) on Forecasting Datasets")

cbar = plt.colorbar(scatter)
cbar.set_label("Cluster")

plt.tight_layout()
plt.savefig(
    f"final_clustering_k{k}.pdf",
    dpi=300
)

plt.show()
# ==========================
# FINAL OUTPUT
# ==========================
print("\n===== FINAL CLUSTER SIZES =====")
print(df_imputed["Cluster"].value_counts().sort_index())

print("\n===== FINAL DATAFRAME =====")
print(df_imputed.to_string())

df_imputed.to_csv(
    "final_clustered_forecasting_datasets.csv"
)

# =====================================
# FIND FARTHEST POINTS FROM CLUSTER 0
# =====================================
from scipy.spatial.distance import cdist

print("\n===== CLOSEST DATASET TO CENTROID (PER CLUSTER) =====")

for cluster_id in np.unique(labels):
    
    # indices of datasets in this cluster
    cluster_indices = np.where(labels == cluster_id)[0]

    # feature vectors of that cluster
    cluster_points = scaled_features[cluster_indices]

    # centroid of this cluster
    centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)

    # distances from centroid
    distances = cdist(cluster_points, centroid).flatten()

    # dataset names
    dataset_names = df_imputed.index[cluster_indices]

    distance_df = pd.DataFrame({
        "Dataset": dataset_names,
        "Distance_From_Centroid": distances
    })

    # sort ascending → closest first
    distance_df = distance_df.sort_values(
        by="Distance_From_Centroid",
        ascending=True
    )

    print(f"\nCluster {cluster_id}:")
    print("Closest dataset:")
    print(distance_df.head(1))

# print("\nTop 5 candidates to remove:")
# print(distance_df.head(5))