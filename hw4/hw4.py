# hw4_metro_local.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, f1_score, roc_auc_score, confusion_matrix)

# Optional XGBoost (skip if not installed)
HAS_XGB = True
try:
    from xgboost import XGBRegressor, XGBClassifier
except Exception:
    HAS_XGB = False
    print("[Info] xgboost not installed; will skip XGB models.")

# ---- path to your local CSV ----
DATA_PATH = "Metro_Interstate_Traffic_Volume.csv"  # ← 如需绝对路径：r"G:\urban\hw4\Metro_Interstate_Traffic_Volume.csv"

def ensure_figdir(): os.makedirs("figs", exist_ok=True)

def regress_metrics(y, yhat):
    return (float(np.sqrt(mean_squared_error(y, yhat))),
            float(mean_absolute_error(y, yhat)),
            float(r2_score(y, yhat)))

def classify_metrics(y, yhat, proba):
    acc = float(accuracy_score(y, yhat))
    f1  = float(f1_score(y, yhat, average="macro"))
    try:
        auc = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except Exception:
        auc = np.nan
    return acc, f1, auc

def scatter_pred_actual(y, yhat, title, fname):
    lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    plt.figure(figsize=(5,5))
    plt.scatter(y, yhat, s=12, alpha=0.6)
    plt.plot(lims, lims, lw=2)
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(title)
    plt.tight_layout(); plt.savefig(f"figs/{fname}", dpi=140); plt.close()

def residual_plot(yhat, resid, title, fname):
    plt.figure(figsize=(6,4))
    plt.scatter(yhat, resid, s=12, alpha=0.6)
    plt.axhline(0, lw=2)
    plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.title(title)
    plt.tight_layout(); plt.savefig(f"figs/{fname}", dpi=140); plt.close()

def corr_heatmap(df, cols, fname):
    C = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9,7))
    im = ax.imshow(C.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=90)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title("Correlation heatmap"); plt.tight_layout()
    plt.savefig(f"figs/{fname}", dpi=140); plt.close()

def line_by_hour_weektype(df, fname):
    g = df.groupby(["is_weekend","hour"])["traffic_volume"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7,4))
    for wk, lab in [(0,"Weekday"), (1,"Weekend")]:
        sub = g[g.is_weekend==wk]
        ax.plot(sub["hour"], sub["traffic_volume"], label=lab)
    ax.legend(); ax.set_xlabel("Hour"); ax.set_ylabel("Mean traffic_volume")
    ax.set_title("Traffic by hour (weekday vs weekend)")
    plt.tight_layout(); plt.savefig(f"figs/{fname}", dpi=140); plt.close()

def bar_by_cat(df, cat, fname, top_k=10):
    top = df[cat].value_counts().nlargest(top_k).index
    tmp = df[df[cat].isin(top)].groupby(cat)["traffic_volume"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8,5))
    tmp.plot(kind="barh", ax=ax)
    ax.set_xlabel("Mean traffic_volume"); ax.set_ylabel(cat)
    ax.set_title(f"Mean traffic by {cat}")
    plt.tight_layout(); plt.savefig(f"figs/{fname}", dpi=140); plt.close()

def get_feature_names(pre):
    names = []
    for name, trans, cols in pre.transformers_:
        if name == "num": names.extend(cols)
        elif name == "cat": names.extend(trans.get_feature_names_out(cols))
    return names

def main():
    ensure_figdir()
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found: {os.path.abspath(DATA_PATH)}")
    df = pd.read_csv(DATA_PATH)

    # --- Engineer features ---
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    df["hour"] = df["date_time"].dt.hour
    df["dow"]  = df["date_time"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["sin_hour"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["cos_hour"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["is_holiday"] = (df["holiday"].astype(str) != "None").astype(int)

    # --- EDA figures ---
    num_hist = [c for c in ["traffic_volume","temp","rain_1h","snow_1h","clouds_all"] if c in df.columns]
    if num_hist:
        df[num_hist].hist(figsize=(12,8), bins=24)
        plt.tight_layout(); plt.savefig("figs/eda_hists.png", dpi=140); plt.close()
    corr_cols = [c for c in ["traffic_volume","temp","rain_1h","snow_1h","clouds_all",
                             "sin_hour","cos_hour","is_weekend","is_holiday"] if c in df.columns]
    if corr_cols: corr_heatmap(df, corr_cols, "eda_corr.png")
    line_by_hour_weektype(df, "eda_hour_weektype.png")
    bar_by_cat(df, "weather_main", "eda_weather_main.png")

    # --- chronological split 80/20 ---
    split_idx = int(len(df)*0.8)
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    y_tr = train_df["traffic_volume"].values
    y_te = test_df["traffic_volume"].values

    # Preprocess (scaler + one-hot)
    numeric = [c for c in ["temp","rain_1h","snow_1h","clouds_all",
                           "sin_hour","cos_hour","is_weekend","is_holiday"] if c in df.columns]
    categorical = [c for c in ["weather_main","dow"] if c in df.columns]
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ], remainder="drop")

    # ================= Regression =================
    print("\n[Regression] OLS / RF / (optional) XGB")
    # OLS
    ols = Pipeline([("pre", pre), ("lr", LinearRegression())]).fit(train_df, y_tr)
    yhat_tr_ols = ols.predict(train_df); yhat_te_ols = ols.predict(test_df)
    mtr_ols = regress_metrics(y_tr, yhat_tr_ols); mte_ols = regress_metrics(y_te, yhat_te_ols)
    scatter_pred_actual(y_te, yhat_te_ols, "OLS — Pred vs Actual (Test)", "reg_ols_pred.png")

    # RF with time-aware CV
    tscv = TimeSeriesSplit(n_splits=5)
    rf = Pipeline([("pre", pre), ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))])
    grid_rf = {"rf__n_estimators":[300,600], "rf__max_depth":[None,10,16], "rf__min_samples_leaf":[1,5,10]}
    rf_cv = GridSearchCV(rf, grid_rf, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(train_df, y_tr)
    rf_best = rf_cv.best_estimator_
    yhat_tr_rf = rf_best.predict(train_df); yhat_te_rf = rf_best.predict(test_df)
    mtr_rf = regress_metrics(y_tr, yhat_tr_rf); mte_rf = regress_metrics(y_te, yhat_te_rf)
    scatter_pred_actual(y_te, yhat_te_rf, "RF — Pred vs Actual (Test)", "reg_rf_pred.png")

    # XGB (optional)
    if HAS_XGB:
        xgb = Pipeline([("pre", pre), ("xgb", XGBRegressor(objective="reg:squarederror",
                                                           random_state=42, n_jobs=-1,
                                                           tree_method="hist", eval_metric="rmse", verbosity=0))])
        grid_xgb = {"xgb__n_estimators":[400,800], "xgb__learning_rate":[0.05,0.1],
                    "xgb__max_depth":[4,6,8], "xgb__subsample":[0.8,1.0], "xgb__colsample_bytree":[0.8,1.0]}
        xgb_cv = GridSearchCV(xgb, grid_xgb, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1).fit(train_df, y_tr)
        xgb_best = xgb_cv.best_estimator_
        yhat_tr_xgb = xgb_best.predict(train_df); yhat_te_xgb = xgb_best.predict(test_df)
        mtr_xgb = regress_metrics(y_tr, yhat_tr_xgb); mte_xgb = regress_metrics(y_te, yhat_te_xgb)
        scatter_pred_actual(y_te, yhat_te_xgb, "XGB — Pred vs Actual (Test)", "reg_xgb_pred.png")
    else:
        mtr_xgb = mte_xgb = None

    # residuals of best test RMSE
    candidates = [("OLS", yhat_te_ols, mte_ols[0]), ("RF", yhat_te_rf, mte_rf[0])]
    if mte_xgb is not None: candidates.append(("XGB", yhat_te_xgb, mte_xgb[0]))
    best_name, best_pred, _ = sorted(candidates, key=lambda x: x[2])[0]
    residual_plot(best_pred, y_te - best_pred, f"{best_name} — Residuals (Test)", "reg_best_resid.png")

    # save regression table
    rows = [
        {"Model":"OLS", "Train RMSE":mtr_ols[0], "Train MAE":mtr_ols[1], "Train R2":mtr_ols[2],
         "Test RMSE":mte_ols[0], "Test MAE":mte_ols[1], "Test R2":mte_ols[2]},
        {"Model":f"RF {rf_cv.best_params_}", "Train RMSE":mtr_rf[0], "Train MAE":mtr_rf[1], "Train R2":mtr_rf[2],
         "Test RMSE":mte_rf[0], "Test MAE":mte_rf[1], "Test R2":mte_rf[2]},
    ]
    if mtr_xgb is not None:
        rows.append({"Model":f"XGB {xgb_cv.best_params_}", "Train RMSE":mtr_xgb[0], "Train MAE":mtr_xgb[1], "Train R2":mtr_xgb[2],
                     "Test RMSE":mte_xgb[0], "Test MAE":mte_xgb[1], "Test R2":mte_xgb[2]})
    pd.DataFrame(rows).to_csv("figs/results_regression.csv", index=False)

    # ================= Classification =================
    print("\n[Classification] LogReg / RF-C / (optional) XGB-C")
    # bin by train quantiles (keep thresholds for test)
    q1, q2 = np.quantile(y_tr, [0.33, 0.66])
    biny = lambda y: np.where(y <= q1, 0, np.where(y <= q2, 1, 2))
    y_tr_c, y_te_c = biny(y_tr), biny(y_te)

    # same preprocessor
    pre_c = pre

    # LogReg
    lr = Pipeline([("pre", pre_c),
                   ("lr", LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs"))]).fit(train_df, y_tr_c)
    yhat_tr_lr = lr.predict(train_df); yhat_te_lr = lr.predict(test_df)
    proba_tr_lr = lr.predict_proba(train_df); proba_te_lr = lr.predict_proba(test_df)
    mtr_lr = classify_metrics(y_tr_c, yhat_tr_lr, proba_tr_lr)
    mte_lr = classify_metrics(y_te_c, yhat_te_lr, proba_te_lr)

    # RF-C
    rfc = Pipeline([("pre", pre_c), ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))]).fit(train_df, y_tr_c)
    yhat_tr_rfc = rfc.predict(train_df); yhat_te_rfc = rfc.predict(test_df)
    proba_tr_rfc = rfc.predict_proba(train_df); proba_te_rfc = rfc.predict_proba(test_df)
    mtr_rfc = classify_metrics(y_tr_c, yhat_tr_rfc, proba_tr_rfc)
    mte_rfc = classify_metrics(y_te_c, yhat_te_rfc, proba_te_rfc)

    # XGB-C (optional)
    if HAS_XGB:
        xgbc = Pipeline([("pre", pre_c),
                         ("xgbc", XGBClassifier(objective="multi:softprob", num_class=3,
                                                random_state=42, n_jobs=-1, tree_method="hist",
                                                eval_metric="mlogloss", verbosity=0))]).fit(train_df, y_tr_c)
        yhat_tr_xgbc = xgbc.predict(train_df); yhat_te_xgbc = xgbc.predict(test_df)
        proba_tr_xgbc = xgbc.predict_proba(train_df); proba_te_xgbc = xgbc.predict_proba(test_df)
        mtr_xgbc = classify_metrics(y_tr_c, yhat_tr_xgbc, proba_tr_xgbc)
        mte_xgbc = classify_metrics(y_te_c, yhat_te_xgbc, proba_te_xgbc)
    else:
        mtr_xgbc = mte_xgbc = None

    # confusion matrix of best Macro-F1
    cands = [("LogReg", yhat_te_lr, mte_lr[1]), ("RF-C", yhat_te_rfc, mte_rfc[1])]
    if mte_xgbc is not None: cands.append(("XGB-C", yhat_te_xgbc, mte_xgbc[1]))
    best_name, best_pred, _ = sorted(cands, key=lambda x: (x[2] if not np.isnan(x[2]) else -1))[-1]
    cm = confusion_matrix(y_te_c, best_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["Low","Med","High"]); ax.set_yticklabels(["Low","Med","High"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i,j]), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{best_name} — Confusion Matrix (Test)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig("figs/cls_best_confmat.png", dpi=140); plt.close()

    # save classification table
    rows_c = [
        {"Model":"LogReg", "Train Acc":mtr_lr[0], "Train Macro-F1":mtr_lr[1], "Train ROC-AUC":mtr_lr[2],
         "Test Acc":mte_lr[0], "Test Macro-F1":mte_lr[1], "Test ROC-AUC":mte_lr[2]},
        {"Model":"RF-C",   "Train Acc":mtr_rfc[0], "Train Macro-F1":mtr_rfc[1], "Train ROC-AUC":mtr_rfc[2],
         "Test Acc":mte_rfc[0], "Test Macro-F1":mte_rfc[1], "Test ROC-AUC":mte_rfc[2]},
    ]
    if mtr_xgbc is not None:
        rows_c.append({"Model":"XGB-C", "Train Acc":mtr_xgbc[0], "Train Macro-F1":mtr_xgbc[1], "Train ROC-AUC":mtr_xgbc[2],
                       "Test Acc":mte_xgbc[0], "Test Macro-F1":mte_xgbc[1], "Test ROC-AUC":mte_xgbc[2]})
    pd.DataFrame(rows_c).to_csv("figs/results_classification.csv", index=False)

    print("\nSaved: figs/ (all plots), results_regression.csv, results_classification.csv")
    print("Split: train/test = 80% / 20%, random_state = 42")

if __name__ == "__main__":
    main()
