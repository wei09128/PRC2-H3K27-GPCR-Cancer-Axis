#!/usr/bin/env python3
"""
ML with proper cross-validation (no data leakage).
Feature selection happens INSIDE each CV fold via Pipeline.
"""
import matplotlib
matplotlib.use('Agg')
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict

from scipy.stats import pearsonr  # for PCC + p-values

# -----------------------
# Custom Correlation Pruning Transformer
# -----------------------

class CorrelationPruner(BaseEstimator, TransformerMixin):
    """
    Feature selection via correlation pruning.
    Fits on training data only (no leakage).
    """
    def __init__(self, max_features=10, corr_threshold=0.5):
        self.max_features = max_features
        self.corr_threshold = corr_threshold
        self.selected_indices_ = None
        
    def fit(self, X, y):
        X_df = pd.DataFrame(X)
    
        # Rank by F-score
        F, _ = f_classif(X, y)
        rank_idx = np.argsort(F)[::-1]
    
        # Correlation matrix (handle constant cols)
        corr = X_df.corr(method="pearson").abs().fillna(0).values
    
        selected = []
        for idx in rank_idx:
            if len(selected) >= self.max_features:
                break
            if all(corr[idx, s] < self.corr_threshold for s in selected):
                selected.append(idx)
    
        self.selected_indices_ = np.array(selected, dtype=int)
        return self

    
    def transform(self, X):
        return X[:, self.selected_indices_]
    
    def get_support(self):
        return self.selected_indices_


# -----------------------
# Data loading
# -----------------------

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Loaded {df.shape[0]} patients x {df.shape[1]} columns")
    
    # Filter labels
    df = df[df["Aggressive_Label"].isin(["Aggressive", "LessAggressive"])].copy()
    
    # Separate metadata and features
    non_feature_cols = ["Sample_ID", "Cancer_Type", "Response_Status", "Aggressive_Label"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    
    # Drop columns with >60% missing
    missing_pct = df[feature_cols].isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > 0.6].index.tolist()
    if cols_to_drop:
        print(f"Dropping {len(cols_to_drop)} columns with >60% missing")
        feature_cols = [c for c in feature_cols if c not in cols_to_drop]
    
    # Drop rows with >80% missing (in features only)
    row_missing = df[feature_cols].isnull().sum(axis=1) / len(feature_cols)
    # after cols_to_drop update feature_cols
    if len(feature_cols) == 0:
        raise ValueError("All features dropped by missingness filter.")

    rows_to_drop = row_missing[row_missing > 0.8].index
    if len(rows_to_drop) > 0:
        print(f"Dropping {len(rows_to_drop)} rows with >80% missing")
        df = df.drop(rows_to_drop)
    
    y = (df["Aggressive_Label"] == "Aggressive").astype(int).values
    X_df = df[feature_cols].copy()
    
    X = X_df.values
    
    print(f"Final: {X.shape[0]} patients x {X.shape[1]} features")
    return X, y, feature_cols, df


# -----------------------
# Model + param grids
# -----------------------

def build_models_and_grids():
    models = {}
    param_grids = {}

    # 1. Logistic Regression (L2) with feature selection in pipeline
    models["LR_L2"] = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("prune", CorrelationPruner(max_features=12, corr_threshold=0.5)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000, random_state=42)),
    ])
    param_grids["LR_L2"] = {
        "prune__max_features": [7,8, 9,10,14, 15],
        "clf__C": [0.000001,0.00001,0.0001, 0.001, 0.003, 0.01, 0.1, 1.0, 9,10,11],
    }

    # 2. Logistic Regression (L1)
    models["LR_L1"] = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("prune", CorrelationPruner(max_features=10, corr_threshold=0.5)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, random_state=42)),
    ])
    param_grids["LR_L1"] = {
        "prune__max_features": [8,9,10,12, 13, 14,15],
        "clf__C": [8 ,9, 10, 11, 12, 15],
    }

    # 3. Random Forest (smaller grid to avoid getting stuck)
    models["RF"] = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("prune", CorrelationPruner(max_features=15, corr_threshold=0.5)),
        ("clf", RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)),
    ])
    param_grids["RF"] = {
        "prune__max_features": [10, 15],
        "clf__n_estimators": [30,40,50, 100, 150,200],
        "clf__max_depth": [2,3,4, 5, None],
        "clf__min_samples_leaf": [1, 2,3],
    }

    # 4. Linear SVM
    models["SVM_linear"] = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("prune", CorrelationPruner(max_features=10, corr_threshold=0.5)),
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(dual=False, max_iter=5000, random_state=42)),
    ])
    param_grids["SVM_linear"] = {
        "prune__max_features": [8,9, 10,11, 14,15],
        "clf__C": [0.000001,0.00001,0.0001, 0.001, 0.01, 0.1, 1.0],
    }

    return models, param_grids


# -----------------------
# Feature importance
# -----------------------

def get_top_features(model_name, fitted_model, feature_names, k=5):
    """Extract top-k features, mapping through feature selection."""
    
    # Get selected feature indices from pruner
    pruner = fitted_model.named_steps["prune"]
    selected_idx = pruner.get_support()
    selected_names = [feature_names[i] for i in selected_idx]
    
    clf = fitted_model.named_steps["clf"]
    
    if model_name == "RF":
        importances = clf.feature_importances_
    elif model_name in ["LR_L2", "LR_L1", "SVM_linear"]:
        importances = np.abs(clf.coef_[0])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Sort and get top k
    idx = np.argsort(importances)[::-1][:k]
    top_feats = [(selected_names[i], importances[i]) for i in idx]
    return top_feats


def plot_top_features(model_name, top_features, outdir="."):
    feat_names = [f for (f, v) in top_features]
    values = [v for (f, v) in top_features]

    plt.figure(figsize=(8, 4))
    y_pos = np.arange(len(feat_names))
    plt.barh(y_pos, values, color='steelblue')
    plt.yticks(y_pos, feat_names)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance (|coef| or Gini)")
    plt.title(f"Top 5 features – {model_name}")

    out_path = os.path.join(outdir, f"top5_{model_name}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot: {out_path}")

# -----------------------
# ROC (Cross-validated)
# -----------------------

def _cv_decision_scores(estimator, X, y, cv):
    """
    Get cross-validated decision scores for ROC.
    Uses predict_proba if available, else decision_function.
    """
    if hasattr(estimator, "predict_proba"):
        probs = cross_val_predict(
            estimator, X, y, cv=cv, method="predict_proba", n_jobs=-1
        )
        return probs[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = cross_val_predict(
            estimator, X, y, cv=cv, method="decision_function", n_jobs=-1
        )
        return scores
    else:
        # fallback (not ideal for ROC)
        preds = cross_val_predict(
            estimator, X, y, cv=cv, method="predict", n_jobs=-1
        )
        return preds


def plot_cv_roc(estimator, X, y, cv, out_path, title="Model"):
    """
    Plot a single cross-validated ROC curve for a given estimator.
    Feature selection stays inside folds because estimator is a Pipeline.
    """
    scores = _cv_decision_scores(estimator, X, y, cv)
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"{title} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random Classifier")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Cross-Validated): {title}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved ROC: {out_path}")
    
def plot_selected_features_heatmap(best_models, feature_cols, out_path="selected_features_heatmap.png"):
    """
    Binary heatmap of which features were selected by each model's pruner.
    NOT importance. 1 = selected, 0 = not selected.
    """
    model_names = list(best_models.keys())
    mat = np.zeros((len(feature_cols), len(model_names)), dtype=int)

    for j, name in enumerate(model_names):
        pruner = best_models[name].named_steps["prune"]
        sel_idx = pruner.get_support()
        mat[sel_idx, j] = 1

    sel_df = pd.DataFrame(mat, index=feature_cols, columns=model_names)
    sel_df["n_models"] = sel_df.sum(axis=1)
    sel_df = sel_df.sort_values("n_models", ascending=False)

    data = sel_df[model_names].values

    # dynamic height so long feature lists stay readable when zoomed
    h = max(6, 0.18 * data.shape[0])
    plt.figure(figsize=(6.5, h))
    plt.imshow(data, aspect="auto", interpolation="nearest")
    plt.xticks(np.arange(len(model_names)), model_names, rotation=30, ha="right")
    plt.yticks(np.arange(len(sel_df.index)), sel_df.index, fontsize=6)
    plt.xlabel("Model")
    plt.title("Selected Features per Model (1=selected)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved feature selection heatmap: {out_path}")


# -----------------------
# Nested CV metrics + F1 bar plot
# -----------------------

def nested_cv_metrics(base_model, param_grid, X, y, outer_cv, inner_cv):
    """
    Proper nested CV:
      - inner GridSearchCV tunes hyperparams
      - outer fold evaluates on held-out data
    Returns foldwise AUC/F1/ACC lists.
    """
    aucs, f1s, accs = [], [], []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X_tr, y_tr)
        best = grid.best_estimator_

        # scores for AUC
        if hasattr(best, "predict_proba"):
            scores = best.predict_proba(X_te)[:, 1]
        elif hasattr(best, "decision_function"):
            scores = best.decision_function(X_te)
        else:
            scores = best.predict(X_te)

        fpr, tpr, _ = roc_curve(y_te, scores)
        aucs.append(auc(fpr, tpr))

        preds = (scores > 0.5).astype(int) if scores.ndim == 1 else scores
        f1s.append(f1_score(y_te, preds))
        accs.append(accuracy_score(y_te, preds))

    return {"auc": aucs, "f1": f1s, "acc": accs}

# -----------------------
# PCC diagnostics plots
# -----------------------

def compute_feature_pcc_table(df, feature_cols, label_col="Aggressive_Label"):
    """
    Compute PCC (r), p-value, and valid sample size for each feature vs label.
    Label is binary Aggressive=1, LessAggressive=0.
    """
    y_all = (df[label_col] == "Aggressive").astype(int).values
    rows = []

    for col in feature_cols:
        x_raw = pd.to_numeric(df[col], errors="coerce")
        mask = x_raw.notna().values
        x = x_raw.values[mask]
        y = y_all[mask]
        n_valid = len(x)

        if n_valid < 3 or np.all(x == x[0]):
            r, p = 0.0, 1.0
        else:
            r, p = pearsonr(x, y)

        rows.append({
            "feature": col,
            "r": r,
            "abs_r": abs(r),
            "p": p,
            "n_valid": n_valid
        })

    corr_df = pd.DataFrame(rows).sort_values("abs_r", ascending=False)
    return corr_df


def plot_pcc_diagnostics(corr_df, outdir=".", top_k=15):
    """
    Make:
      1) histogram of PCC values
      2) top-|PCC| bar plot (green if p<0.05)
      3) volcano: |PCC| vs -log10(p)
      4) n_valid vs |PCC|
    """
    # 1) Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(corr_df["r"], bins=25, edgecolor="black", alpha=0.8)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Pearson Correlation Coefficient (r)")
    plt.ylabel("Frequency")
    plt.title("Distribution of PCC Values")
    plt.tight_layout()
    path1 = os.path.join(outdir, "pcc_histogram.png")
    plt.savefig(path1, dpi=300); plt.close()
    print(f"Saved: {path1}")

    # 2) Top-k |PCC|
    top = corr_df.head(top_k).copy()
    colors = ["green" if p < 0.05 else "gray" for p in top["p"]]

    plt.figure(figsize=(7, 4.5))
    y_pos = np.arange(len(top))
    plt.barh(y_pos, top["abs_r"], color=colors)
    plt.yticks(y_pos, top["feature"])
    plt.gca().invert_yaxis()
    plt.xlabel("Absolute PCC")
    plt.title(f"Top {top_k} Features by |PCC| (p<0.05)")
    plt.tight_layout()
    path2 = os.path.join(outdir, f"top{top_k}_abs_pcc.png")
    plt.savefig(path2, dpi=300); plt.close()
    print(f"Saved: {path2}")

    # 3) Volcano plot
    plt.figure(figsize=(6, 5))
    neglogp = -np.log10(corr_df["p"].clip(lower=1e-300))
    sig = corr_df["p"] < 0.05
    plt.scatter(corr_df["abs_r"], neglogp, alpha=0.7)
    plt.axhline(-np.log10(0.05), linestyle="--", label="p=0.05")
    plt.xlabel("Absolute PCC")
    plt.ylabel("-log10(p-value)")
    plt.title("Volcano Plot: Correlation vs Significance")
    plt.legend()
    plt.tight_layout()
    path3 = os.path.join(outdir, "pcc_volcano.png")
    plt.savefig(path3, dpi=300); plt.close()
    print(f"Saved: {path3}")

    # 4) Sample size vs |PCC|
    plt.figure(figsize=(6, 5))
    plt.scatter(corr_df["n_valid"], corr_df["abs_r"], alpha=0.7)
    plt.xlabel("Number of Valid Samples")
    plt.ylabel("Absolute PCC")
    plt.title("Sample Size vs Correlation Strength")
    plt.tight_layout()
    path4 = os.path.join(outdir, "nvalid_vs_abs_pcc.png")
    plt.savefig(path4, dpi=300); plt.close()
    print(f"Saved: {path4}")
    
def plot_selected_features_heatmap(
    best_models,
    feature_cols,
    out_path="selected_features_heatmap.png",
    top_n=None,          # e.g., 30 to show top 30 only
    min_models=1,        # keep features selected by at least this many models
    sort_by="n_models"   # or "alphabetical"
):
    """
    Binary heatmap of feature selection per model.
    1 = selected by model's pruner, 0 = not selected.

    Filtering:
      - min_models: remove features selected by fewer than this many models
      - top_n: after filtering, keep only top N by selection count
    """
    model_names = list(best_models.keys())
    mat = np.zeros((len(feature_cols), len(model_names)), dtype=int)

    for j, name in enumerate(model_names):
        pruner = best_models[name].named_steps["prune"]
        sel_idx = pruner.get_support()
        mat[sel_idx, j] = 1

    sel_df = pd.DataFrame(mat, index=feature_cols, columns=model_names)
    sel_df["n_models"] = sel_df.sum(axis=1)

    # ---- FILTER 1: drop features nobody selected (or below min_models) ----
    sel_df = sel_df[sel_df["n_models"] >= min_models]

    # ---- SORT ----
    if sort_by == "n_models":
        sel_df = sel_df.sort_values("n_models", ascending=False)
    elif sort_by == "alphabetical":
        sel_df = sel_df.sort_index()

    # ---- FILTER 2: keep only top_n if requested ----
    if top_n is not None:
        sel_df = sel_df.head(top_n)

    data = sel_df[model_names].values

    # dynamic size so labels stay readable
    h = max(3.5, 0.22 * data.shape[0])
    plt.figure(figsize=(6.5, h))
    plt.imshow(data, aspect="auto", interpolation="nearest")
    plt.xticks(np.arange(len(model_names)), model_names, rotation=30, ha="right")
    plt.yticks(np.arange(len(sel_df.index)), sel_df.index, fontsize=7)
    plt.xlabel("Model")
    plt.title(f"Selected Features – PRC2/H3K27 Model") # PRC2/H3K27 Model
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved feature selection heatmap: {out_path}")

    
def plot_all_models_cv_roc(best_models, X, y, cv, out_path="roc_all_models.png"):
    """
    One ROC plot with all 4 tuned models overlaid (cross-validated).
    """
    plt.figure(figsize=(6.5, 6.5))

    for name, model in best_models.items():
        scores = _cv_decision_scores(model, X, y, cv)
        fpr, tpr, _ = roc_curve(y, scores)
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={model_auc:.3f})")

    plt.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random Classifier")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (CV) – PRC2/H3K27 Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved combined ROC: {out_path}")

def plot_nested_f1_bar(nested_results, out_path="nested_f1_bar.png"):
    """
    Bar plot: mean F1 per model with std across outer folds.
    """
    model_names = list(nested_results.keys())
    means = [np.mean(nested_results[m]["f1"]) for m in model_names]
    stds  = [np.std(nested_results[m]["f1"]) for m in model_names]

    plt.figure(figsize=(7, 4))
    x = np.arange(len(model_names))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, model_names, rotation=30, ha="right")
    plt.ylabel("F1 Score (Outer CV)")
    plt.title("Nested CV F1 – PRC2/H3K27 Model")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved nested F1 bar plot: {out_path}")

# -----------------------
# Main
# -----------------------

def main(csv_path):
    X, y, feature_cols, df = load_data(csv_path)

    models, param_grids = build_models_and_grids()

    cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    print("\nUsing 7-fold Stratified CV (GridSearchCV, scoring = AUC)")
    print("Feature selection happens INSIDE each fold (no data leakage)\n")

    results = []
    best_models = {}

    for name, base_model in models.items():
        print("=" * 50)
        print(f"Model: {name}")
        print("=" * 50)

        param_grid = param_grids[name]
        n_combinations = 1
        for v in param_grid.values():
            n_combinations *= len(v)
        print(f"Grid search: {n_combinations} param combinations x 7 folds = {n_combinations * 7} fits")

        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        grid.fit(X, y)

        print(f"\nBest CV AUC: {grid.best_score_:.3f}")
        print(f"Best params: {grid.best_params_}")

        best_model = grid.best_estimator_
        best_models[name] = best_model

        # Print selected features to console (no plots)
        pruner = best_model.named_steps["prune"]
        sel_idx = pruner.get_support()
        sel_names = [feature_cols[i] for i in sel_idx]
        print(f"Selected features ({len(sel_names)}):")
        for f in sel_names:
            print("  ", f)

        results.append({
            "Model": name,
            "Best_AUC": grid.best_score_,
            "Best_Params": str(grid.best_params_),
        })
        print()

    # -----------------------
    # NESTED CV BLOCK (honest evaluation)
    # -----------------------
    outer_cv = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    nested_results = {}
    print("\nRunning nested CV for honest AUC/F1/ACC (outer=7, inner=5)...")
    for name, base_model in models.items():
        nested_results[name] = nested_cv_metrics(
            base_model=base_model,
            param_grid=param_grids[name],
            X=X,
            y=y,
            outer_cv=outer_cv,
            inner_cv=inner_cv
        )
        print(
            f"{name}: "
            f"mean AUC={np.mean(nested_results[name]['auc']):.3f} ± {np.std(nested_results[name]['auc']):.3f}, "
            f"mean F1={np.mean(nested_results[name]['f1']):.3f} ± {np.std(nested_results[name]['f1']):.3f}, "
            f"mean ACC={np.mean(nested_results[name]['acc']):.3f} ± {np.std(nested_results[name]['acc']):.3f}"
        )

    # If you kept plot_nested_f1_bar(), this makes your Panel A figure:
    try:
        plot_nested_f1_bar(nested_results, out_path="nested_f1_bar.png")
    except NameError:
        print("plot_nested_f1_bar() not found (skipping bar plot).")

    # -----------------------
    # Plots you want
    # -----------------------
    plot_all_models_cv_roc(best_models, X, y, cv=cv, out_path="roc_all_models.png")

    plot_selected_features_heatmap(
        best_models, feature_cols,
        out_path="selected_features_heatmap.png",
        top_n=15
    )

    # -----------------------
    # Summary table
    # -----------------------
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv("model_comparison.csv", index=False)
    print("\nSaved: model_comparison.csv")

    # Save nested CV summary too (nice for the paper)
    nested_df = pd.DataFrame({
        "Model": list(nested_results.keys()),
        "Nested_AUC_mean": [np.mean(nested_results[m]["auc"]) for m in nested_results],
        "Nested_AUC_sd":   [np.std(nested_results[m]["auc"]) for m in nested_results],
        "Nested_F1_mean":  [np.mean(nested_results[m]["f1"]) for m in nested_results],
        "Nested_F1_sd":    [np.std(nested_results[m]["f1"]) for m in nested_results],
        "Nested_ACC_mean": [np.mean(nested_results[m]["acc"]) for m in nested_results],
        "Nested_ACC_sd":   [np.std(nested_results[m]["acc"]) for m in nested_results],
    })
    nested_df.to_csv("nested_cv_summary.csv", index=False)
    print("Saved: nested_cv_summary.csv")

    print("Done.")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ml.py PRC2_patient_table.csv")
        sys.exit(1)

    main(sys.argv[1])