import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


df = pd.read_csv("archive\\ev_battery_health.csv")

# choosen columns
cols = [
    "cycle_count",
    "fast_charge_ratio",
    "avg_depth_of_discharge_pct",
    "avg_ambient_temp_c",
    "internal_resistance_mohm",
    "total_energy_throughput_kwh"
]

X = df[cols].copy()
true_labels = df["degradation_category"] # Kept strictly for post-cluster validation!

# 3. Handle missing values (impute with median)
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 4. Scale features (crucial for distance-based algorithms like K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 5. Determine optimal K using Silhouette Score
print("Evaluating cluster counts (K = 2 to 6)...")
best_k = 3
best_score = -1

for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    print(f" -> K={k} | Silhouette Score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k

print(f"\nOptimal K selected: {best_k} based on highest silhouette score.")

# 6. Run final KMeans model with the optimal K
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster_id'] = kmeans_final.fit_predict(X_scaled)

# 7. Validate: Cross-tabulate unsupervised clusters with true degradation categories
print("\n Validation: Clusters vs. True Degradation Categories")
crosstab = pd.crosstab(df['cluster_id'], true_labels, margins=True)
print(crosstab)