#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leukemia Remission Prediction — Logistic Regression
Author: Lavanya Siripuram

GOAL
----
Predict whether a leukemia patient achieves complete remission (binary:
1 = remission, 0 = no remission) using six clinical/lab measurements taken
at diagnosis. This is the classic teaching dataset of 27 leukemia patients
used widely in biostatistics courses to demonstrate logistic regression.

DATA
----
27 patients, 6 predictors:
  - Cell  : cellularity of the marrow clot section
  - Smear : smear differential percentage of blasts
  - Infil : percentage of absolute marrow leukemia cell infiltrate
  - Li    : labeling index (% of labeled leukemia cells)
  - Blast : absolute number of blasts in peripheral blood
  - Temp  : highest temperature before start of treatment

NOTE ON DATA QUALITY
---------------------
The CSV I originally received had its decimal points corrupted into minus
signs during a prior transcription/export step (e.g. 0.8 became -8, and
0.996 became -996). I confirmed this against the original published
version of this dataset (it's a well-known dataset from SAS/Minitab
biostatistics documentation) and reconstructed the correct decimal values
before doing any modeling. Training on the corrupted version would have
produced a model fitting transcription noise, not biology.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

RANDOM_STATE = 42  # fixed seed so results are reproducible run-to-run

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/leukemia_remission.csv")
print(f"Dataset: {df.shape[0]} patients, {df.shape[1] - 1} predictors")
print(f"Remission cases: {df['Remiss'].sum()} / {len(df)} "
      f"({df['Remiss'].mean():.1%})\n")

# ---------------------------------------------------------------------------
# 2. Define target (y) and features (X)
# ---------------------------------------------------------------------------
y = df["Remiss"]          # dependent variable — remission yes/no
X = df.drop(["Remiss"], axis=1)  # independent variables

# ---------------------------------------------------------------------------
# 3. Train/test split
# ---------------------------------------------------------------------------
# With only 27 rows total, a 70/30 split leaves ~8 patients for testing —
# small enough that results vary noticeably between random seeds. Fixing
# random_state makes results reproducible; in practice, with a dataset
# this small, leave-one-out cross-validation would give a more reliable
# performance estimate than a single train/test split (see "What I'd do
# next" below).
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 4. Fit logistic regression
# ---------------------------------------------------------------------------
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
model.fit(x_train, y_train)

coefficients_df = pd.DataFrame({
    "Feature": x_train.columns,
    "Weight (Coefficient)": model.coef_[0]
}).sort_values(by="Weight (Coefficient)", ascending=False)

print("--- Baseline Intercept ---")
print(f"{model.intercept_[0]:.3f}\n")

print("--- Feature Importance (logistic regression coefficients) ---")
print(coefficients_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Predict on test set
# ---------------------------------------------------------------------------
pred = model.predict(x_test)

# ---------------------------------------------------------------------------
# 6. Confusion matrix
# ---------------------------------------------------------------------------
# sklearn's confusion_matrix(y_true, y_pred) returns:
#   [[TN, FP],
#    [FN, TP]]
# rows = actual class, columns = predicted class, in label order [0, 1].
cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()

print("\n--- Confusion Matrix ---")
print(cm)
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

accuracy = accuracy_score(y_test, pred)
print(f"\nTest set accuracy: {accuracy:.1%}  ({tn + tp} correct out of {len(y_test)})")

print("\n--- Classification Report ---")
print(classification_report(y_test, pred, target_names=["No Remission", "Remission"]))

# ---------------------------------------------------------------------------
# 7. Plot confusion matrix
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_xticklabels(["No Remission", "Remission"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No Remission", "Remission"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (Test Set, n={len(y_test)})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved plot to confusion_matrix.png")

# ---------------------------------------------------------------------------
# 8. Leave-One-Out Cross-Validation (more reliable estimate for n=27)
# ---------------------------------------------------------------------------
# A single 70/30 split on 27 rows tests on only ~8 patients, so the accuracy
# above can swing a lot depending on which patients happen to land in the
# test set. Leave-One-Out CV trains on 26 patients and predicts the 1 left
# out, repeated for all 27 — using every patient as a test case exactly
# once. This gives a far more stable accuracy estimate for a dataset this
# small than any single train/test split can.
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
loo_preds, loo_actuals = [], []

for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    m = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    m.fit(X_tr, y_tr)
    loo_preds.append(m.predict(X_te)[0])
    loo_actuals.append(y_te.values[0])

loo_accuracy = accuracy_score(loo_actuals, loo_preds)
loo_cm = confusion_matrix(loo_actuals, loo_preds)

print("\n--- Leave-One-Out Cross-Validation (n=27, all patients) ---")
print(f"LOOCV accuracy: {loo_accuracy:.1%}")
print("LOOCV confusion matrix:")
print(loo_cm)
