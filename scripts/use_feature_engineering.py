import pandas as pd
import os
from src.feature_engineering import FeatureEngineering
from joblib import dump
from src.logger import get_logger

logger = get_logger('use_feature_engineering', 'feature_engineering.log')

# ============ DATA LOAD ============
filled_x_train = pd.read_csv('data/filled/filled_x_train.csv')
filled_x_test = pd.read_csv('data/filled/filled_x_test.csv')

# ============ TRAIN ============
feature_engineering = FeatureEngineering()
engineered_x_train = feature_engineering.fit_transform(filled_x_train)
logger.info("Feature engineering completed for TRAIN dataset.")

# ============ SAVE PIPELINE ============
os.makedirs('pipeline', exist_ok=True)

dump(feature_engineering, 'pipeline/engineered_pipeline.joblib')
logger.info("Feature engineering pipeline saved to pipeline/feature_engineering.joblib")

# ============ TEST ============
engineered_x_test = feature_engineering.transform(filled_x_test)
logger.info("Feature engineering completed for TEST dataset.")

# ============ SAVE DATASET ============
os.makedirs('data/engineered', exist_ok=True)

engineered_x_train.to_csv('data/engineered/engineered_x_train.csv', index=False)
logger.info("Engineered dataset saved to data/engineered/engineered_x_train.csv")

engineered_x_test.to_csv('data/engineered/engineered_x_test.csv', index=False)
logger.info("Engineered dataset saved to data/engineered/engineered_x_test.csv")

logger.info("Successfully engineered the dataset and saved all outputs!")