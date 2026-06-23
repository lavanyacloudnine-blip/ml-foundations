#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Telecom Customer Churn Prediction — Decision Tree Classifier
Author: Lavanya Siripuram

GOAL
----
Predict whether a telecom customer will churn (leave the service) using
their usage patterns (call minutes, charges, customer service calls) and
plan details (international plan, voicemail plan).

DATA
----
3,333 customers, 19 columns. Target is `churn` (yes/no). `state` and
`area_code` are dropped as unrelated to usage behavior, per domain
judgment — they describe where the customer lives, not how they use the
service, so including them risks the model leaning on a region
proxy rather than genuine churn drivers.

IMPORTANT: THIS DATASET IS IMBALANCED
---------------------------------------
Only 14.5% of customers churn (483 out of 3,333). This matters a lot:
a model that always predicts "no churn" gets ~85.5% accuracy automatically,
without learning anything useful. Accuracy alone is therefore a misleading
metric here — precision, recall, and F1 on the "churn" class are what
actually indicate whether the model is useful for a retention team trying
to catch customers before they leave.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, balanced_accuracy_score)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/churn.csv")
print(f"Dataset: {df.shape[0]} customers, {df.shape[1]} columns")

churn_rate = (df["churn"] == "yes").mean()
print(f"Churn rate: {churn_rate:.1%} "
      f"({(df['churn']=='yes').sum()} churned / {len(df)} total)")
print(f"A model that always predicts 'no churn' would score "
      f"{1 - churn_rate:.1%} accuracy without learning anything — "
      f"keep this in mind when reading the results below.\n")

# ---------------------------------------------------------------------------
# 2. Drop unwanted columns, encode categoricals
# ---------------------------------------------------------------------------
df = df.drop(["state", "area_code"], axis=1)

encoder = LabelEncoder()
df = df.assign(churn=encoder.fit_transform(df["churn"]))                     # no=0, yes=1
df = df.assign(international_plan=encoder.fit_transform(df["international_plan"]))
df = df.assign(voice_mail_plan=encoder.fit_transform(df["voice_mail_plan"]))

# ---------------------------------------------------------------------------
# 3. Split dependent/independent variables
# ---------------------------------------------------------------------------
y = df["churn"]
X = df.drop(["churn"], axis=1)

# ---------------------------------------------------------------------------
# 4. Train/test split (stratified — important given the class imbalance,
#    otherwise the test set could end up with very few churn cases)
# ---------------------------------------------------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 5. Build decision tree
# ---------------------------------------------------------------------------
DT_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6)
DT_model.fit(x_train, y_train)

pred = DT_model.predict(x_test)

# ---------------------------------------------------------------------------
# 6. Evaluate — accuracy AND the metrics that actually matter here
# ---------------------------------------------------------------------------
cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()

print("--- Confusion Matrix ---")
print(cm)
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n")

acc = accuracy_score(y_test, pred)
bal_acc = balanced_accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.1%}")
print(f"Balanced accuracy (accounts for class imbalance): {bal_acc:.1%}\n")

print("--- Classification Report ---")
print(classification_report(y_test, pred, target_names=["No Churn", "Churn"]))

# ---------------------------------------------------------------------------
# 7. 5-fold cross-validation for a more stable estimate
# ---------------------------------------------------------------------------
cv_scores = cross_val_score(DT_model, X, y, cv=5, scoring="balanced_accuracy")
print(f"5-fold CV balanced accuracy: {cv_scores.mean():.1%} "
      f"(+/- {cv_scores.std():.1%})\n")

# ---------------------------------------------------------------------------
# 8. Feature importance
# ---------------------------------------------------------------------------
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": DT_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("--- Top 8 Feature Importances ---")
print(importance_df.head(8).to_string(index=False))

# ---------------------------------------------------------------------------
# 9. Plot: feature importance + confusion matrix
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

top8 = importance_df.head(8).iloc[::-1]
axes[0].barh(top8["Feature"], top8["Importance"], color="#2E5C8A")
axes[0].set_xlabel("Importance")
axes[0].set_title("Top 8 Feature Importances")

im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["No Churn", "Churn"])
axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["No Churn", "Churn"])
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
axes[1].set_title(f"Confusion Matrix (Test Set, n={len(y_test)})")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black",
                      fontsize=13)
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("churn_results.png", dpi=150)
print("\nSaved plot to churn_results.png")
