import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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
# LOAD DATA (ONCE)
# ==========================
csv_path = "time_series_features_all_39_datasets.csv"
df = pd.read_csv(csv_path, index_col=0)
print(f"Loaded {df.shape[0]} datasets with {df.shape[1]} features.")
print(f"Missing values before imputation:\n{df.isnull().sum()}")

# ==========================
# KNN IMPUTATION
# ==========================
imputer = KNNImputer(n_neighbors=5, weights='distance')
df_imputed_array = imputer.fit_transform(df)
df_imputed = pd.DataFrame(df_imputed_array, index=df.index, columns=df.columns)
print(f"\nMissing values after imputation: {df_imputed.isnull().sum().sum()} (should be 0)")

# ==========================
# SCALE FEATURES
# ==========================
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_imputed)

# ==========================
# PCA REDUCTION (2 components)
# ==========================
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

# ==========================
# SILHOUETTE & ELBOW (reference)
# ==========================
k_values = range(2, 11)
inertia_values = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(principal_components)
    inertia_values.append(kmeans.inertia_)
    sil = silhouette_score(principal_components, labels)
    silhouette_scores.append(sil)

print("\n===== OPTIMAL K DETERMINATION (reference) =====")
for k, inertia, sil in zip(k_values, inertia_values, silhouette_scores):
    print(f"K={k}: Inertia={inertia:.2f}, Silhouette={sil:.3f}")
best_k_sil = k_values[np.argmax(silhouette_scores)]
print(f"\nSilhouette suggests K={best_k_sil}")

# ==========================
# MANUAL K SELECTION
# ==========================
k = 3   # change as needed
print(f"\nUsing manually selected K = {k} for final clustering.")

# ==========================
# FINAL CLUSTERING
# ==========================
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(principal_components)

# Add Cluster column to the imputed DataFrame (index already has dataset names)
df_imputed["Cluster"] = labels

# ==========================
# 2D PLOT (decision boundary)
# ==========================
x_min, x_max = principal_components[:, 0].min() - 1, principal_components[:, 0].max() + 1
y_min, y_max = principal_components[:, 1].min() - 1, principal_components[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.figure(figsize=(18, 11))
plt.contourf(xx, yy, z, alpha=0.25, cmap="Set1")

scatter = plt.scatter(
    principal_components[:, 0],
    principal_components[:, 1],
    c=labels,
    cmap="tab10",
    s=250,
    edgecolor="black",
    linewidth=1.8
)

# Annotate with dataset names from the index
for i, dataset in enumerate(df_imputed.index):
    plt.annotate(dataset,
                 (principal_components[i, 0],
                  principal_components[i, 1]),
                 fontsize=16,
                 fontweight='bold')

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title(f"K-Means Clustering (K={k}) on Time Series Features (KNN imputed)")
cbar = plt.colorbar(scatter)
cbar.ax.tick_params(labelsize=18)
cbar.set_label("Cluster", fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig(f"clustering_results_k{k}_knn_imputed.pdf", dpi=300)
plt.show()

# ==========================
# DISPLAY IMPUTED DATAFRAME WITH CLUSTERS
# ==========================
print("\n===== IMPUTED DATAFRAME WITH CLUSTERS =====")
# To match the original display style, use to_string() – no extra "Dataset" column
print(df_imputed.to_string())

# Save to CSV (index will be the dataset names, no extra index column)
df_imputed.to_csv("imputed_clustered_data.csv")

print("\nCluster sizes:")
print(df_imputed["Cluster"].value_counts().sort_index())