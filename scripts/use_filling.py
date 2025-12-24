import pandas as pd
import os
from joblib import dump
from src.train_preprocessing import PreprocessingTrain
from src.test_preprocessing import PreprocessingTest
from src.logger import get_logger

logger = get_logger('use_filling', 'preprocessing.log')

# ============ DATA LOAD ============
x_train = pd.read_csv('data/split/x_train.csv')
x_test = pd.read_csv('data/split/x_test.csv')

# ============ TRAIN ============
preprocessing_train = PreprocessingTrain(x_train, target='is_canceled')

filled_x_train = preprocessing_train.fillingTrain().getDataset()
logger.info("Preprocessing steps completed successfully.")

# ============ SAVE PIPELINE ============
os.makedirs('pipeline', exist_ok=True)

dump(preprocessing_train, 'pipeline/filled_pipeline.joblib')
logger.info("Preprocessing pipeline saved to pipeline/filled_x_train.joblib")

# ============ TEST ============
preprocessing_test = PreprocessingTest(preprocessing_train)

filled_x_test = preprocessing_test.fillingTest(x_test)
logger.info("Preprocessing steps completed successfully.")

# ============ SAVE DATASET ============
os.makedirs('data/filled', exist_ok=True)

filled_x_train.to_csv('data/filled/filled_x_train.csv', index=False)
logger.info("Filled dataset saved to data/filled/filled_x_train.csv")

filled_x_test.to_csv('data/filled/filled_x_test.csv', index=False)
logger.info("Filled dataset saved to data/filled/filled_x_test.csv")

logger.info("Successfully filled the dataset and saved all outputs!")