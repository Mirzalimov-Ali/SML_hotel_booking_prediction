import os
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from joblib import dump
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from src.logger import get_logger
from rich.table import Table
from rich.console import Console
from skopt.space import Real, Integer, Categorical

os.chdir(r'C:\Users\User\Desktop\ML_Lesson\Projects\hotelBooking_cancelling_prediction')

logger = get_logger('use_training', 'training.log')

# ============ DATA LOAD ============
x_train = pd.read_csv('data/preprocessed/preprocessed_x_train.csv')
x_test  = pd.read_csv('data/preprocessed/preprocessed_x_test.csv')

y_train = pd.read_csv('data/split/y_train.csv').values.ravel()
y_test  = pd.read_csv('data/split/y_test.csv').values.ravel()

logger.info("Data loaded successfully")


# df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

# drop_cols = [
#     "car_age",
#     "age_in_month",
#     "reg_date_score",
#     "is_manual",
#     "model_popularity",
#     "notRepairedDamage",
#     "is_automatic",
#     "is_diesel",
#     "has_damage",
#     "model_length"
# ]

# df = df.drop(drop_cols, errors='ignore')

# X = df.drop(columns="kilometer", errors='ignore')
# y = df['kilometer']

kf = KFold(n_splits=3, shuffle=True, random_state=42)

# ===================== TRAIN ==========================
model = LogisticRegression(max_iter=300)

model.fit(x_train, y_train)
logger.info("Model training completed")

# ===================== EVALUEATION ==========================
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred_test)

precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1  = f1_score(y_test, y_pred_test)

scores = cross_val_score(model, x_test, y_test, cv=kf, scoring='f1_macro')

kf_mean = scores.mean()
kf_std = scores.std()

# ===================== SAVE MODEL ==========================
os.makedirs('pipeline', exist_ok=True)

dump(model, 'pipeline/final_pipeline.joblib', compress=3)
logger.info("Final model pipeline saved.")

# ===================== SAVE RESULTS TABLE ==========================
results = []
console = Console()

results.append(["LogisticRegression", accuracy, precision, recall, f1, kf_mean, kf_std,])
results_sorted = sorted(results, key=lambda x: x[-1], reverse=True)

table = Table(title="LogisticRegression Results", show_lines=True)
table.add_column("Algorithm")
table.add_column("Accuracy")
table.add_column("Precision")
table.add_column("Recall")
table.add_column("F1-score")
table.add_column("K-Fold mean")
table.add_column("K-Fold std")

table.add_row(
    "LogisticRegression",
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