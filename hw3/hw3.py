# boston_housing_regression.py
# Boston Housing: OLS (subset), Ridge, LASSO with plots and stats

import os, io, urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from itertools import cycle

# ---------- 0) Load data (UCI Boston Housing) ----------
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
COLS = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX",
        "PTRATIO","B","LSTAT","MEDV"]

def load_boston_uci():
    with urllib.request.urlopen(URL) as f:
        raw = f.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=COLS, engine="python")
    return df

df = load_boston_uci()

# Optional: drop capped labels at 50 to reduce distortion
df = df[df["MEDV"] < 50].reset_index(drop=True)

# ---------- 1) Quick EDA ----------
os.makedirs("figs", exist_ok=True)

# Histograms
ax = df.hist(figsize=(14,10), bins=20)
plt.tight_layout(); plt.savefig("figs/histograms.png"); plt.close()

# Correlation heatmap (matplotlib only)
corr = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(9,7))
im = ax.imshow(corr, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns)
fig.colorbar(im, ax=ax); plt.tight_layout(); plt.savefig("figs/corr_heatmap.png"); plt.close()

# ---------- 2) Train/test split ----------
X_full = df.drop(columns=["MEDV"])
y = df["MEDV"].values
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# Subset for OLS (chosen from EDA & VIF reasoning)
ols_subset = ["RM","LSTAT","PTRATIO","NOX","DIS","CHAS"]
X_train_ols = X_train[ols_subset].copy()
X_test_ols  = X_test[ols_subset].copy()

# ---------- 3) OLS on subset ----------
ols = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("lr", LinearRegression())
])
ols.fit(X_train_ols, y_train)
y_pred_train_ols = ols.predict(X_train_ols)
y_pred_test_ols  = ols.predict(X_test_ols)

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

m_tr_ols = metrics(y_train, y_pred_train_ols)
m_te_ols = metrics(y_test,  y_pred_test_ols)

# ---------- 4) Ridge & LASSO (all features, CV) ----------
alphas = np.logspace(-3, 3, 61)
cv = KFold(n_splits=10, shuffle=True, random_state=42)

ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=alphas, cv=cv, scoring="neg_mean_squared_error"))
])
ridge.fit(X_train, y_train)
y_pred_train_r = ridge.predict(X_train)
y_pred_test_r  = ridge.predict(X_test)
m_tr_r = metrics(y_train, y_pred_train_r)
m_te_r = metrics(y_test,  y_pred_test_r)
best_alpha_ridge = ridge.named_steps["ridge"].alpha_

lasso = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(alphas=None, cv=cv, random_state=42, max_iter=20000))
])
lasso.fit(X_train, y_train)
y_pred_train_l = lasso.predict(X_train)
y_pred_test_l  = lasso.predict(X_test)
m_tr_l = metrics(y_train, y_pred_train_l)
m_te_l = metrics(y_test,  y_pred_test_l)
best_alpha_lasso = lasso.named_steps["lasso"].alpha_

# ---------- 5) Coefficients & plots ----------
def coef_from_pipeline(pipe, columns):
    scaler = pipe.named_steps["scaler"]
    coef   = (pipe.named_steps["ridge"].coef_
              if "ridge" in pipe.named_steps
              else pipe.named_steps["lasso"].coef_)
    # map back to original feature order
    # (scaler preserves order; coef aligns with columns)
    return pd.Series(coef, index=columns).sort_values(key=np.abs, ascending=False)

coef_ridge = coef_from_pipeline(ridge, X_full.columns)
coef_lasso = coef_from_pipeline(lasso, X_full.columns)

coef_ridge.sort_values().plot(kind="barh", figsize=(7,6))
plt.title(f"Ridge Coefficients (alpha={best_alpha_ridge:.4g})")
plt.tight_layout(); plt.savefig("figs/ridge_coefs.png"); plt.close()

coef_lasso.sort_values().plot(kind="barh", figsize=(7,6))
plt.title(f"LASSO Coefficients (alpha={best_alpha_lasso:.4g})")
plt.tight_layout(); plt.savefig("figs/lasso_coefs.png"); plt.close()

# Predicted vs Actual plots
def pred_plot(y_true, y_pred, title, path):
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, s=25, alpha=0.7)
    plt.plot(lims, lims, lw=2)
    plt.xlabel("Actual MEDV"); plt.ylabel("Predicted MEDV"); plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

pred_plot(y_test, y_pred_test_ols,  "OLS (subset) — Test",  "figs/ols_pred.png")
pred_plot(y_test, y_pred_test_r,    "Ridge — Test",        "figs/ridge_pred.png")
pred_plot(y_test, y_pred_test_l,    "LASSO — Test",        "figs/lasso_pred.png")

# Residual plot (test)
def resid_plot(y_true, y_pred, title, path):
    res = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, res, s=20, alpha=0.7)
    plt.axhline(0, lw=2)
    plt.xlabel("Predicted MEDV"); plt.ylabel("Residual")
    plt.title(title); plt.tight_layout(); plt.savefig(path); plt.close()

resid_plot(y_test, y_pred_test_ols, "OLS Residuals — Test",   "figs/ols_resid.png")
resid_plot(y_test, y_pred_test_r,   "Ridge Residuals — Test", "figs/ridge_resid.png")
resid_plot(y_test, y_pred_test_l,   "LASSO Residuals — Test", "figs/lasso_resid.png")

# ---------- 6) Print summary table ----------
def row(name, m_tr, m_te):
    return {
        "Model": name,
        "Train RMSE": round(m_tr[0], 3),
        "Train MAE":  round(m_tr[1], 3),
        "Train R2":   round(m_tr[2], 3),
        "Test RMSE":  round(m_te[0], 3),
        "Test MAE":   round(m_te[1], 3),
        "Test R2":    round(m_te[2], 3),
    }

summary = pd.DataFrame([
    row("OLS (subset)", m_tr_ols, m_te_ols),
    row(f"Ridge (alpha={best_alpha_ridge:.3g})", m_tr_r, m_te_r),
    row(f"LASSO (alpha={best_alpha_lasso:.3g})", m_tr_l, m_te_l),
])

print("\n=== Boston Housing: Results (train/test) ===")
print(summary.to_string(index=False))

# Save to CSV for easy copy into Word
summary.to_csv("figs/results.csv", index=False)
print("\nSaved figures to ./figs and table to figs/results.csv")
