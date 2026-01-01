import os
import optuna
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from joblib import dump
from src.logger import get_logger
from rich.table import Table
from rich.console import Console
from imblearn.over_sampling import SMOTE


# ===================== PATH =====================
os.chdir(r'C:\SML_Projects\SML_hotelBooking_cancelling_prediction')

logger = get_logger('use_training', 'training.log')

# ===================== DATA LOAD =====================
x_train = pd.read_csv('data/preprocessed/preprocessed_x_train.csv')
x_test  = pd.read_csv('data/preprocessed/preprocessed_x_test.csv')

y_train = pd.read_csv('data/split/y_train.csv').values.ravel()
y_test  = pd.read_csv('data/split/y_test.csv').values.ravel()

smote = SMOTE(random_state=42, k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train)

logger.info("Data loaded and balanced successfully")

kf = KFold(n_splits=3, shuffle=True, random_state=42)


# ===================== OPTUNA OBJECTIVE =====================
def objective(trial):

    rf = RandomForestClassifier(
        n_estimators=trial.suggest_int("rf_n_estimators", 50, 500),
        max_depth=trial.suggest_int("rf_max_depth", 5, 50),
        min_samples_split=trial.suggest_int("rf_min_samples_split", 2, 10),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    xgb = XGBClassifier(
        n_estimators=trial.suggest_int("xgb_n_estimators", 50, 500),
        max_depth=trial.suggest_int("xgb_max_depth", 3, 20),
        learning_rate=trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    lgbm = LGBMClassifier(
        n_estimators=trial.suggest_int("lgbm_n_estimators", 50, 500),
        max_depth=trial.suggest_int("lgbm_max_depth", -1, 20),
        learning_rate=trial.suggest_float("lgbm_learning_rate", 0.01, 0.3, log=True),
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    final_estimator = BaggingClassifier(
        estimator=LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),
        n_estimators=trial.suggest_int("bag_n_estimators", 5, 50),
        random_state=42,
        n_jobs=-1
    )

    model = StackingClassifier(
        estimators=[
            ("rf", rf),
            ("xgb", xgb),
            ("lgbm", lgbm)
        ],
        final_estimator=final_estimator,
        n_jobs=-1
    )

    threshold = trial.suggest_float("threshold", 0.2, 0.5)

    model.fit(x_train, y_train)
    y_proba = model.predict_proba(x_train)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    score = recall_score(y_train, y_pred)

    return score


# ===================== OPTUNA RUN =====================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30, show_progress_bar=True)

logger.info(f"Best Recall: {study.best_value}")
logger.info(f"Best params: {study.best_params}")

best = study.best_params
best_threshold = best["threshold"]


# ===================== TRAIN BEST MODEL =====================
rf = RandomForestClassifier(
    n_estimators=best["rf_n_estimators"],
    max_depth=best["rf_max_depth"],
    min_samples_split=best["rf_min_samples_split"],
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

xgb = XGBClassifier(
    n_estimators=best["xgb_n_estimators"],
    max_depth=best["xgb_max_depth"],
    learning_rate=best["xgb_learning_rate"],
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

lgbm = LGBMClassifier(
    n_estimators=best["lgbm_n_estimators"],
    max_depth=best["lgbm_max_depth"],
    learning_rate=best["lgbm_learning_rate"],
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

final_estimator = BaggingClassifier(
    estimator=LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    n_estimators=best["bag_n_estimators"],
    random_state=42,
    n_jobs=-1
)

model = StackingClassifier(
    estimators=[
        ("rf", rf),
        ("xgb", xgb),
        ("lgbm", lgbm)
    ],
    final_estimator=final_estimator,
    n_jobs=-1
)

model.fit(x_train, y_train)
logger.info("Best Stacking model trained")


# ===================== EVALUATION WITH THRESHOLD =====================
y_proba_test = model.predict_proba(x_test)[:, 1]
y_pred_test = (y_proba_test >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

scores = cross_val_score(model, x_train, y_train, cv=kf, scoring='f1_macro')
kf_mean = scores.mean()
kf_std = scores.std()

# ===================== SAVE MODEL =====================
os.makedirs('pipeline', exist_ok=True)
dump(model, 'pipeline/final_pipeline.joblib', compress=3)

logger.info(f"Final model saved with threshold={best_threshold:.2f}")


results = []
console = Console()

results.append(["Stacking", accuracy, precision, recall, f1, kf_mean, kf_std,])
results_sorted = sorted(results, key=lambda x: x[-1], reverse=True)

table = Table(title="Stacking Results", show_lines=True)
table.add_column("Algorithm")
table.add_column("Accuracy")
table.add_column("Precision")
table.add_column("Recall")
table.add_column("F1-score")
table.add_column("K-Fold mean")
table.add_column("K-Fold std")

table.add_row(
    "Stacking",
    f"{accuracy:.2f}",
    f"{precision:.2f}",
    f"{recall:.2f}",
    f"{f1:.2f}",
    f"{kf_mean:.2f}",
    f"{kf_std:.2f}",
)

temp_console = Console(record=True)
temp_console.print(table)

text = temp_console.export_text()
with open("results/final_results.txt", "w", encoding="utf-8") as f:
    f.write(text)
logger.info("Comparison table saved at results/final_results.txt")

print("Results saved to results/final_results.txt")