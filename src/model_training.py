import os
import pandas as pd
from joblib import dump
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

from skopt import BayesSearchCV
from skopt.space import Real, Integer

from rich.console import Console
from rich.table import Table

from src.logger import get_logger


class Trainer:
    def __init__(self, df_path, target, save_path="pipeline/final_pipeline.joblib"):
        self.df_path = df_path
        self.target = target
        self.save_path = save_path

        self.logger = get_logger("trainer", "training.log")
        self.console = Console()

        self.model = CatBoostRegressor(verbose=0, random_state=42)

    # =====================================================
    # Load Data
    # =====================================================
    def load_data(self):
        self.logger.info(f"Loading dataset → {self.df_path}")
        df = pd.read_csv(self.df_path)

        X = df.drop(columns=self.target, errors='ignore')
        y = df[self.target]

        self.logger.info(f"Dataset Loaded. Shape = {df.shape}")
        return X, y

    # =====================================================
    # TRAIN
    # =====================================================
    def train(self):
        X, y = self.load_data()

        self.logger.info("Starting Train/Test Split…")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.logger.info("Training model…")
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.r2 = r2_score(y_test, y_pred)
        self.mae = mean_absolute_error(y_test, y_pred)

        self.logger.info(f"Train Completed. R2={self.r2:.4f}, MAE={self.mae:.4f}")

    # =====================================================
    # EVALUATE — KFold cross validation
    # =====================================================
    def evaluate(self):
        X, y = self.load_data()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        self.logger.info("Running K-Fold evaluation…")
        scores = cross_val_score(self.model, X, y, cv=kf, scoring="r2")

        self.kf_mean = scores.mean()
        self.kf_std = scores.std()

        self.logger.info(
            f"KFold Mean={self.kf_mean:.4f}, Std={self.kf_std:.4f}"
        )

    # =====================================================
    # TUNING — Bayesian Optimization
    # =====================================================
    def tuning(self):
        X, y = self.load_data()

        params = {
            "depth": Integer(3, 10),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "l2_leaf_reg": Real(1, 10)
        }

        self.logger.info("Starting Bayesian Optimization…")

        bayes = BayesSearchCV(
            estimator=self.model,
            search_spaces=params,
            n_iter=20,
            random_state=42,
            cv=3,
            scoring="r2",
            n_jobs=-1,
            verbose=0
        )

        bayes.fit(X, y)
        self.model = bayes.best_estimator_

        self.logger.info(f"Best Params → {bayes.best_params_}")

    # =====================================================
    # Save Model Pipeline
    # =====================================================
    def save_pipeline(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        dump(self.model, self.save_path, compress=3)

        self.logger.info(f"Model saved at {self.save_path}")

    # =====================================================
    # Save Results as TXT Table
    # =====================================================
    def save_dataset(self):
        table = Table(title="Model Evaluation Results", show_lines=True)

        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("R2 Score", f"{self.r2:.4f}")
        table.add_row("MAE", f"{self.mae:.4f}")
        table.add_row("K-Fold Mean", f"{self.kf_mean:.4f}")
        table.add_row("K-Fold Std", f"{self.kf_std:.4f}")

        temp_console = Console(record=True)
        temp_console.print(table)
        text = temp_console.export_text()

        os.makedirs("results", exist_ok=True)
        with open("results/final_results.txt", "w", encoding="utf-8") as f:
            f.write(text)

        self.logger.info("Results saved to results/final_results.txt")
        self.console.print(table)