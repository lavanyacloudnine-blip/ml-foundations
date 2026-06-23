#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Invoice Overdue-Days Prediction — Multiple Linear Regression
Author: Lavanya Siripuram

GOAL
----
Predict how many days overdue an invoice will end up (New_Due_Stage) using
the customer's payment history and complaint behavior, captured at the time
the invoice was issued. This is a classic accounts-receivable / collections
risk problem: flagging invoices likely to go badly overdue early lets a
collections team prioritize outreach before the debt ages.

DATA
----
5,000 invoice records from a B2B billing dataset (142 raw columns covering
invoice details, monthly account status snapshots, and historical payment/
complaint percentages). Target and features are derived as described below.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/invoice_data.csv")

# ---------------------------------------------------------------------------
# 2. Define target and features
# ---------------------------------------------------------------------------
# Target: New_Due_Stage = number of days overdue the invoice ultimately
# reached (0 = paid on time; higher = more overdue).
target = "New_Due_Stage"

# Features: invoice amount, the customer's historical days-to-pay, and
# their 12-month complaint/payment-behavior percentages. These are all
# known at (or close to) invoice issue time, so they're valid predictors
# rather than information leaking in from the future.
#
# Note: this dataset stores several groups of percentage columns that are
# shares of the same whole and sum to 1 for every row (e.g. Pct_Complains /
# Pct_Com_Sugg / Pct_Request, and similarly the four Pct_*Num12 columns).
# Within each group only one column is kept — including all of them causes
# severe multicollinearity and produces unstable, misleadingly large
# coefficients.
features = [
    "Invoice_Amount",
    "Days_To_Pay",
    "Pct_Complains",   # one representative of the complaint-type share group
    "Num_Tran_12",
    "Dol_Tran_12",
    "Pct_age_Num12",   # one representative of the payment-status share group
]

X = df[features]
y = df[target]

print(f"Dataset: {df.shape[0]} invoices, {len(features)} features")
print(f"Target '{target}' range: {y.min():.0f} to {y.max():.0f} days overdue")
print(f"Target mean: {y.mean():.1f} days, median: {y.median():.1f} days\n")

# ---------------------------------------------------------------------------
# 3. Train/test split (70/30)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)

# Scale features — helps interpret coefficient magnitudes on a level
# playing field since Invoice_Amount and Dol_Tran_12 are on very
# different scales than the percentage columns.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Model performance on test set (30% holdout) ---")
print(f"RMSE: {rmse:.1f} days")
print(f"MAE:  {mae:.1f} days")
print(f"R^2:  {r2:.3f}")

# ---------------------------------------------------------------------------
# 6. Feature importance (standardized coefficients)
# ---------------------------------------------------------------------------
coef_df = pd.DataFrame({
    "feature": features,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=False)

print("\n--- Feature impact (standardized coefficients) ---")
print(coef_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Plot: predicted vs actual
# ---------------------------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.4, s=15)
lims = [0, max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual days overdue")
plt.ylabel("Predicted days overdue")
plt.title("Predicted vs Actual Overdue Days (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig("predicted_vs_actual.png", dpi=150)
print("\nSaved plot to predicted_vs_actual.png")
