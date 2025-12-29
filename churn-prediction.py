import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

import mlflow
import mlflow.sklearn


warnings.filterwarnings("ignore")


# =========================
# CONFIG
# =========================
DATA_PATH = "BankChurners.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
K_CLUSTERS = 4

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

# MLflow
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT = "Credit_Card_Churn"


# =========================
# UTILITIES
# =========================
def save_fig(filename: str):
    path = os.path.join(FIG_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def build_preprocess(X_df):
    cat_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_df.select_dtypes(exclude=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop"
    )


def evaluate_pipeline(pipe, X_te, y_te):
    y_pred = pipe.predict(X_te)

    if hasattr(pipe[-1], "predict_proba"):
        y_prob = pipe.predict_proba(X_te)[:, 1]
    else:
        y_prob = None

    out = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred, zero_division=0),
        "recall": recall_score(y_te, y_pred, zero_division=0),
        "f1": f1_score(y_te, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_te, y_prob) if y_prob is not None else np.nan,
        "cm": confusion_matrix(y_te, y_pred),
        "y_prob": y_prob  # ROC için lazım
    }
    return out


def log_model_run(run_name, setting, model_name, pipe, metrics, extra_params=None):
    """
    MLflow'a bir model run'ı loglar:
    - params (model.get_params())
    - metrics
    - confusion matrix json
    - model artifact
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"setting": setting, "model": model_name})

        # Params
        params = {}
        try:
            params.update(pipe.named_steps["model"].get_params())
        except Exception:
            pass
        if extra_params:
            params.update(extra_params)

        # Çok uzun parametreleri kısaltalım (MLflow bazen şişiyor)
        params_clean = {k: str(v)[:250] for k, v in params.items()}
        mlflow.log_params(params_clean)

        # Metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])
        if not np.isnan(metrics["roc_auc"]):
            mlflow.log_metric("roc_auc", metrics["roc_auc"])

        # Confusion matrix as artifact (json)
        cm_path = os.path.join(OUT_DIR, f"cm_{run_name}.json")
        with open(cm_path, "w", encoding="utf-8") as f:
            json.dump(metrics["cm"].tolist(), f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(cm_path)

        # Model artifact
        mlflow.sklearn.log_model(pipe, artifact_path="model")


# =========================
# EDA PLOTS
# =========================
def plot_target_distribution(df):
    counts = df["churn"].value_counts()
    plt.figure()
    plt.bar(["Existing(0)", "Attrited(1)"], [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Churn Dağılımı")
    plt.ylabel("Count")
    return save_fig("eda_target_distribution.png")


def plot_categorical_by_churn(df, cat_col):
    # churn kırılımlı kategori dağılımı
    ctab = pd.crosstab(df[cat_col], df["churn"], normalize="index")
    plt.figure()
    plt.bar(ctab.index.astype(str), ctab.get(0, pd.Series(0, index=ctab.index)).values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{cat_col} - Existing Oranı (churn kırılımlı)")
    plt.ylabel("Oran")
    return save_fig(f"eda_cat_{cat_col}_existing_ratio.png")


def plot_numeric_hist(df, num_col):
    plt.figure()
    plt.hist(df[num_col].dropna(), bins=30)
    plt.title(f"{num_col} Histogram")
    plt.xlabel(num_col)
    plt.ylabel("Frekans")
    return save_fig(f"eda_hist_{num_col}.png")


def plot_numeric_box_by_churn(df, num_col):
    # churn=0 ve churn=1 için boxplot
    data0 = df[df["churn"] == 0][num_col].dropna()
    data1 = df[df["churn"] == 1][num_col].dropna()
    plt.figure()
    plt.boxplot([data0, data1], labels=["Existing(0)", "Churn(1)"])
    plt.title(f"{num_col} - churn kırılımlı boxplot")
    plt.ylabel(num_col)
    return save_fig(f"eda_box_{num_col}_by_churn.png")


def plot_corr_heatmap(df, numeric_cols):
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title("Korelasyon Isı Haritası (Numerik)")
    return save_fig("eda_corr_heatmap.png")


# =========================
# RESULT PLOTS
# =========================
def plot_auc_comparison(df_results, title, filename):
    """
    df_results: columns -> model, setting, roc_auc
    """
    pivot = df_results.pivot(index="model", columns="setting", values="roc_auc")
    pivot = pivot.dropna(how="all")
    # iki kolon varsa yan yana bar basit şekilde
    pivot = pivot.sort_values(by=pivot.columns.tolist()[-1], ascending=False)

    # 1. kolon
    plt.figure()
    plt.bar(pivot.index, pivot.iloc[:, 0].values)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"{title} - {pivot.columns[0]}")
    plt.ylabel("ROC-AUC")
    save_fig(f"{filename}_1.png")

    # 2. kolon varsa
    if pivot.shape[1] > 1:
        plt.figure()
        plt.bar(pivot.index, pivot.iloc[:, 1].values)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{title} - {pivot.columns[1]}")
        plt.ylabel("ROC-AUC")
        save_fig(f"{filename}_2.png")

    return pivot


def plot_roc_curve_for_best(best_name, best_pipe, X_test, y_test, filename):
    # ROC curve
    if not hasattr(best_pipe[-1], "predict_proba"):
        return None

    y_prob = best_pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - {best_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    return save_fig(filename)


# =========================
# MAIN
# =========================
def main():
    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ============
    # 1) LOAD + CLEAN
    # ============
    df_raw = pd.read_csv(DATA_PATH)
    print("Raw shape:", df_raw.shape)

    # Drop leakage columns
    leak_cols = [c for c in df_raw.columns if ("Naive_Bayes_Classifier" in c) or ("Naive_Bayes" in c)]
    df_raw = df_raw.drop(columns=leak_cols, errors="ignore")
    print("Dropped leakage cols:", leak_cols)

    # Drop ID
    df = df_raw.drop(columns=["CLIENTNUM"], errors="ignore").copy()

    # Create churn
    df["churn"] = df["Attrition_Flag"].map({"Existing Customer": 0, "Attrited Customer": 1})
    df = df.drop(columns=["Attrition_Flag"], errors="ignore")

    print("Clean shape:", df.shape)
    print("Churn counts:\n", df["churn"].value_counts())
    print("Churn ratios:\n", df["churn"].value_counts(normalize=True))

    # ============
    # 1b) EDA (save figures + log to MLflow)
    # ============
    with mlflow.start_run(run_name="EDA"):
        mlflow.set_tags({"stage": "EDA"})

        # basic dataset stats
        mlflow.log_param("n_rows", df.shape[0])
        mlflow.log_param("n_cols", df.shape[1])
        mlflow.log_param("leak_cols_dropped", ",".join(leak_cols) if leak_cols else "None")

        # target distribution
        p1 = plot_target_distribution(df)
        mlflow.log_artifact(p1)

        # choose some categorical vars if exist
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for c in cat_cols:
            pc = plot_categorical_by_churn(df, c)
            mlflow.log_artifact(pc)

        # choose some numeric vars
        num_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
        num_cols = [c for c in num_cols if c != "churn"]

        # histogram + boxplot for key numeric vars if present
        key_nums = [c for c in ["Customer_Age", "Credit_Limit", "Total_Trans_Amt", "Total_Trans_Ct", "Months_Inactive_12_mon"] if c in df.columns]
        for n in key_nums:
            ph = plot_numeric_hist(df, n)
            pb = plot_numeric_box_by_churn(df, n)
            mlflow.log_artifact(ph)
            mlflow.log_artifact(pb)

        # correlation heatmap (numeric)
        if len(num_cols) >= 2:
            pcorr = plot_corr_heatmap(df, num_cols[:20])  # çok kalabalık olmasın diye ilk 20
            mlflow.log_artifact(pcorr)

        # save basic describe table
        desc_path = os.path.join(TAB_DIR, "describe_numeric.csv")
        df[num_cols].describe().to_csv(desc_path)
        mlflow.log_artifact(desc_path)

    # ============
    # 2) SPLIT
    # ============
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("\nTRAIN churn counts:\n", y_train.value_counts())
    print("TEST churn counts:\n", y_test.value_counts())

    # ============
    # 3) SEGMENTATION (train only)
    # ============
    seg_features = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Trans_Amt",
        "Total_Trans_Ct"
    ]
    seg_features = [c for c in seg_features if c in X_train.columns]  # güvenlik

    seg_scaler = StandardScaler()
    Xseg_train_scaled = seg_scaler.fit_transform(X_train[seg_features])

    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=RANDOM_STATE, n_init=10).fit(Xseg_train_scaled)

    X_train_seg = X_train.copy()
    X_test_seg = X_test.copy()

    X_train_seg["segment"] = kmeans.predict(seg_scaler.transform(X_train_seg[seg_features]))
    X_test_seg["segment"] = kmeans.predict(seg_scaler.transform(X_test_seg[seg_features]))

    # Make segment categorical
    X_train_seg["segment"] = X_train_seg["segment"].astype(str)
    X_test_seg["segment"] = X_test_seg["segment"].astype(str)

    # Segment churn rate (train)
    tmp = X_train_seg.copy()
    tmp["churn"] = y_train.values
    seg_rate = tmp.groupby("segment")["churn"].mean().sort_values(ascending=False)
    print("\nChurn rate by segment (train):\n", seg_rate)

    # log segmentation stats
    with mlflow.start_run(run_name="Segmentation"):
        mlflow.set_tags({"stage": "Segmentation"})
        mlflow.log_param("k_clusters", K_CLUSTERS)
        mlflow.log_param("seg_features", ",".join(seg_features))
        seg_path = os.path.join(TAB_DIR, "segment_churn_rate.csv")
        seg_rate.to_csv(seg_path)
        mlflow.log_artifact(seg_path)

    # ============
    # 4) MODELS
    # ============
    models = {
        "LogReg_balanced": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "RandomForest_balanced": RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample"
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM_RBF_balanced": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "MLP_NN": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=1200,
            random_state=RANDOM_STATE
        )
    }

    # ============
    # 5) FULL SETTING: no segment vs segment
    # ============
    print("\n======================")
    print("A) FULL SETTING")
    print("======================")

    # No segment
    preprocess_no = build_preprocess(X_train)
    results_no = []
    trained_no = {}

    for name, mdl in models.items():
        pipe = Pipeline(steps=[("prep", preprocess_no), ("model", clone(mdl))])
        pipe.fit(X_train, y_train)
        m = evaluate_pipeline(pipe, X_test, y_test)
        trained_no[name] = pipe

        results_no.append({
            "model": name, "setting": "no_segment_full",
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "roc_auc": m["roc_auc"]
        })

        print(f"\n{name} | no_segment_full")
        print("CM:\n", m["cm"])
        print(f"AUC={m['roc_auc']:.4f} Recall={m['recall']:.4f} F1={m['f1']:.4f} Acc={m['accuracy']:.4f}")

        log_model_run(
            run_name=f"{name}_no_segment_full",
            setting="no_segment_full",
            model_name=name,
            pipe=pipe,
            metrics=m
        )

    results_no_full = pd.DataFrame(results_no)

    # With segment
    preprocess_seg = build_preprocess(X_train_seg)
    results_seg = []
    trained_seg = {}

    for name, mdl in models.items():
        pipe = Pipeline(steps=[("prep", preprocess_seg), ("model", clone(mdl))])
        pipe.fit(X_train_seg, y_train)
        m = evaluate_pipeline(pipe, X_test_seg, y_test)
        trained_seg[name] = pipe

        results_seg.append({
            "model": name, "setting": "with_segment_full",
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "roc_auc": m["roc_auc"]
        })

        print(f"\n{name} | with_segment_full")
        print("CM:\n", m["cm"])
        print(f"AUC={m['roc_auc']:.4f} Recall={m['recall']:.4f} F1={m['f1']:.4f} Acc={m['accuracy']:.4f}")

        log_model_run(
            run_name=f"{name}_with_segment_full",
            setting="with_segment_full",
            model_name=name,
            pipe=pipe,
            metrics=m
        )

    results_seg_full = pd.DataFrame(results_seg)

    full_compare = pd.concat([results_no_full, results_seg_full], ignore_index=True)
    full_csv = os.path.join(TAB_DIR, "full_compare.csv")
    full_compare.to_csv(full_csv, index=False)

    # ============
    # 6) REDUCED SETTING: drop seg_features
    # ============
    print("\n======================")
    print("B) REDUCED SETTING (drop seg_features)")
    print("======================")

    X_train_red = X_train.drop(columns=seg_features)
    X_test_red = X_test.drop(columns=seg_features)

    X_train_seg_red = X_train_seg.drop(columns=seg_features)
    X_test_seg_red = X_test_seg.drop(columns=seg_features)

    # No segment reduced
    preprocess_no_red = build_preprocess(X_train_red)
    res_no_red = []
    trained_no_red = {}

    for name, mdl in models.items():
        pipe = Pipeline(steps=[("prep", preprocess_no_red), ("model", clone(mdl))])
        pipe.fit(X_train_red, y_train)
        m = evaluate_pipeline(pipe, X_test_red, y_test)
        trained_no_red[name] = pipe

        res_no_red.append({
            "model": name, "setting": "no_segment_reduced",
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "roc_auc": m["roc_auc"]
        })

        print(f"\n{name} | no_segment_reduced")
        print("CM:\n", m["cm"])
        print(f"AUC={m['roc_auc']:.4f} Recall={m['recall']:.4f} F1={m['f1']:.4f} Acc={m['accuracy']:.4f}")

        log_model_run(
            run_name=f"{name}_no_segment_reduced",
            setting="no_segment_reduced",
            model_name=name,
            pipe=pipe,
            metrics=m
        )

    results_no_red = pd.DataFrame(res_no_red)

    # With segment reduced
    preprocess_seg_red = build_preprocess(X_train_seg_red)
    res_seg_red = []
    trained_seg_red = {}

    for name, mdl in models.items():
        pipe = Pipeline(steps=[("prep", preprocess_seg_red), ("model", clone(mdl))])
        pipe.fit(X_train_seg_red, y_train)
        m = evaluate_pipeline(pipe, X_test_seg_red, y_test)
        trained_seg_red[name] = pipe

        res_seg_red.append({
            "model": name, "setting": "with_segment_reduced",
            "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "roc_auc": m["roc_auc"]
        })

        print(f"\n{name} | with_segment_reduced")
        print("CM:\n", m["cm"])
        print(f"AUC={m['roc_auc']:.4f} Recall={m['recall']:.4f} F1={m['f1']:.4f} Acc={m['accuracy']:.4f}")

        log_model_run(
            run_name=f"{name}_with_segment_reduced",
            setting="with_segment_reduced",
            model_name=name,
            pipe=pipe,
            metrics=m
        )

    results_seg_red = pd.DataFrame(res_seg_red)

    red_compare = pd.concat([results_no_red, results_seg_red], ignore_index=True)
    red_csv = os.path.join(TAB_DIR, "reduced_compare.csv")
    red_compare.to_csv(red_csv, index=False)

    # ============
    # 7) SUMMARY PLOTS + log artifacts to MLflow
    # ============
    with mlflow.start_run(run_name="Summary"):
        mlflow.set_tags({"stage": "Summary"})

        # tables
        mlflow.log_artifact(full_csv)
        mlflow.log_artifact(red_csv)

        # AUC comparison plots
        pivot_full = plot_auc_comparison(
            full_compare[["model", "setting", "roc_auc"]],
            title="ROC-AUC Karşılaştırma (FULL)",
            filename="compare_full_auc"
        )

        pivot_red = plot_auc_comparison(
            red_compare[["model", "setting", "roc_auc"]],
            title="ROC-AUC Karşılaştırma (REDUCED)",
            filename="compare_reduced_auc"
        )

        # log all figures in FIG_DIR
        for fn in os.listdir(FIG_DIR):
            mlflow.log_artifact(os.path.join(FIG_DIR, fn))

        # choose best model by FULL AUC (segmentli veya segmentsiz en iyi)
        best_row = full_compare.sort_values("roc_auc", ascending=False).iloc[0]
        best_model_name = best_row["model"]
        best_setting = best_row["setting"]

        if best_setting == "no_segment_full":
            best_pipe = trained_no[best_model_name]
            roc_path = plot_roc_curve_for_best(best_model_name, best_pipe, X_test, y_test, "best_model_roc_full.png")
        else:
            best_pipe = trained_seg[best_model_name]
            roc_path = plot_roc_curve_for_best(best_model_name, best_pipe, X_test_seg, y_test, "best_model_roc_full.png")

        if roc_path:
            mlflow.log_artifact(roc_path)

        # log best info
        mlflow.log_param("best_model_full", f"{best_model_name}::{best_setting}")
        mlflow.log_metric("best_full_auc", float(best_row["roc_auc"]))

    # ============
    # 8) GRID SEARCH (Selected Models)
    # ============
    print("\n======================")
    print("C) GRID SEARCH (Selected Models)")
    print("======================")

    param_grids = {
        "LogReg_balanced": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"]
        },
        "RandomForest_balanced": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 15],
            "model__min_samples_split": [2, 10]
        },
        "MLP_NN": {
            "model__hidden_layer_sizes": [(64, 32), (128, 64)],
            "model__alpha": [0.0001, 0.001],
            "model__learning_rate_init": [0.001, 0.01]
        }
    }

    preprocess_gs = build_preprocess(X_train)
    grid_summary = []

    for model_name in ["LogReg_balanced", "RandomForest_balanced", "MLP_NN"]:
        print(f"\nGridSearchCV: {model_name} | NO SEGMENT FULL")

        pipe = Pipeline(steps=[
            ("prep", preprocess_gs),
            ("model", clone(models[model_name]))
        ])

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grids[model_name],
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X_train, y_train)

        best_pipe = grid.best_estimator_
        test_m = evaluate_pipeline(best_pipe, X_test, y_test)

        grid_summary.append({
            "model": model_name,
            "best_params": grid.best_params_,
            "cv_best_auc": grid.best_score_,
            "test_auc": test_m["roc_auc"],
            "test_recall": test_m["recall"],
            "test_f1": test_m["f1"]
        })

        print("Best params:", grid.best_params_)
        print(f"CV best AUC={grid.best_score_:.4f} | Test AUC={test_m['roc_auc']:.4f} Recall={test_m['recall']:.4f} F1={test_m['f1']:.4f}")

        # MLflow run for each grid
        with mlflow.start_run(run_name=f"GridSearch_{model_name}"):
            mlflow.set_tags({"stage": "GridSearch", "model": model_name})
            mlflow.log_params({k: str(v) for k, v in grid.best_params_.items()})
            mlflow.log_metric("cv_best_auc", grid.best_score_)
            mlflow.log_metric("test_auc", test_m["roc_auc"])
            mlflow.log_metric("test_recall", test_m["recall"])
            mlflow.log_metric("test_f1", test_m["f1"])
            mlflow.sklearn.log_model(best_pipe, artifact_path="best_model")

    grid_df = pd.DataFrame(grid_summary).sort_values("test_auc", ascending=False)
    grid_path = os.path.join(TAB_DIR, "grid_search_summary.csv")
    grid_df.to_csv(grid_path, index=False)
    print("\nGRID SEARCH SUMMARY:\n", grid_df)
    print("\nSaved outputs to:", OUT_DIR)
    print("MLflow UI: run -> mlflow ui")


if __name__ == "__main__":
    main()
