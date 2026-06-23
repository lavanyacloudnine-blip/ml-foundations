#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insurance Customer Segmentation — K-Means Clustering
Author: Lavanya Siripuram

GOAL
----
Segment insurance customers into meaningful groups based on their
premiums, age, renewal behavior, claims history, and income — with no
labeled target. This is unsupervised learning: there's no "correct"
answer to predict, only patterns to discover.

DATA
----
100 insurance customers, 5 continuous attributes:
  - Premiums Paid
  - Age
  - Days to Renew (days until policy renewal)
  - Claims made (cumulative claim amount)
  - Income

No missing values, no duplicates, no categorical encoding needed.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/cluster.csv")
print(f"Dataset: {df.shape[0]} customers, {df.shape[1]} attributes\n")
print(df.describe().round(1))

# ---------------------------------------------------------------------------
# 2. Standardize features
# ---------------------------------------------------------------------------
# K-Means uses distance between points to form clusters. Without scaling,
# Income (tens of thousands) and Claims made would completely dominate the
# distance calculation over Age (tens) — the clusters would really just be
# "income clusters" wearing a disguise. Standardizing puts every feature on
# equal footing (mean 0, std 1).
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df)

# ---------------------------------------------------------------------------
# 3. Elbow method to choose K
# ---------------------------------------------------------------------------
inertias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(scaled_features)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(k_range), inertias, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.xticks(list(k_range))
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=150)
print("\nSaved elbow plot to elbow_method.png")

# ---------------------------------------------------------------------------
# 3b. Silhouette score — a second, more rigorous check on K
# ---------------------------------------------------------------------------
# The elbow bend at K=3 is visible but gradual, not sharp — worth checking
# against a second method rather than taking it at face value. Silhouette
# score measures how well-separated clusters actually are (range -1 to 1,
# higher is better), independent of the inertia curve's shape.
print("\n--- Silhouette scores by K ---")
for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(scaled_features)
    score = silhouette_score(scaled_features, km.labels_)
    print(f"K={k}: silhouette={score:.3f}")
print("Note: silhouette favors K=2 here, even though K=3 is used below for "
      "its more actionable business segmentation — see README for discussion.")

# ---------------------------------------------------------------------------
# 4. Fit final model with chosen K
# ---------------------------------------------------------------------------
# The elbow visibly bends around K=3 — inertia keeps dropping after that,
# but at a much slower rate, meaning additional clusters buy increasingly
# little separation for the added complexity.
K = 3
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
cluster_labels = kmeans.fit_predict(scaled_features)

df["cluster"] = cluster_labels

# ---------------------------------------------------------------------------
# 5. Profile each cluster using ORIGINAL (unscaled) values
# ---------------------------------------------------------------------------
# Important: profiling clusters using the standardized values (means near
# 0, values like -1.44) tells you nothing a business stakeholder can use.
# Group by cluster on the original dataframe instead, so the profile shows
# real premiums, real ages, real income.
print(f"\n--- Cluster Profile (K={K}, original units) ---")
profile = df.groupby("cluster").agg(
    customers=("Age", "count"),
    avg_premium=("Premiums Paid", "mean"),
    avg_age=("Age", "mean"),
    avg_days_to_renew=("Days to Renew", "mean"),
    avg_claims=("Claims made", "mean"),
    avg_income=("Income", "mean"),
).round(1)
print(profile.to_string())

# ---------------------------------------------------------------------------
# 6. Visualize clusters (Income vs Premiums Paid, colored by cluster)
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 6))
colors = ["#2E5C8A", "#C0392B", "#27AE60"]
for c in range(K):
    subset = df[df["cluster"] == c]
    plt.scatter(subset["Income"], subset["Premiums Paid"],
                label=f"Cluster {c} (n={len(subset)})",
                color=colors[c], alpha=0.7, s=60)
plt.xlabel("Income")
plt.ylabel("Premiums Paid")
plt.title("Customer Segments: Income vs Premiums Paid")
plt.legend()
plt.tight_layout()
plt.savefig("cluster_scatter.png", dpi=150)
print("\nSaved cluster scatter plot to cluster_scatter.png")
