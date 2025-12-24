import pandas as pd
import os
from joblib import dump
from src.train_preprocessing import PreprocessingTrain
from src.test_preprocessing import PreprocessingTest
from src.logger import get_logger

logger = get_logger('use_preprocessing', 'preprocessing.log')

# ============ DATA LOAD ============
engineered_x_train = pd.read_csv('data/engineered/engineered_x_train.csv')
engineered_x_test = pd.read_csv('data/engineered/engineered_x_test.csv')

# ============ TRAIN ============
preprocessing_train = PreprocessingTrain(engineered_x_train, target='is_canceled')

preprocessed_x_train = (
    preprocessing_train
    .encodingTrain()
    .logTransformationTrain()
    .scalingTrain()
    .getDataset()
)
logger.info("Preprocessing steps completed successfully.")

# ============ SAVE PIPELINE ============
os.makedirs('pipeline', exist_ok=True)

dump(preprocessing_train, 'pipeline/preprocessed_pipeline.joblib')
logger.info("Preprocessing pipeline saved to pipeline/preprodessed_pipeline.joblib")

# ============ TEST ============
preprocessing_test = PreprocessingTest(preprocessing_train)

preprocessed_x_test = preprocessing_test.transform(engineered_x_test, filled=True)
logger.info("Preprocessing steps completed successfully.")

# ============ SAVE DATASET ============
os.makedirs('data/preprocessed', exist_ok=True)

preprocessed_x_train.to_csv('data/preprocessed/preprocessed_x_train.csv', index=False)
logger.info("reprocessed dataset saved to data/preprocessed/preprocessed_x_train.csv")

preprocessed_x_test.to_csv('data/preprocessed/preprocessed_x_test.csv', index=False)
logger.info("Preprocessed dataset saved to data/preprocessed/preprocessed_x_test.csv")

logger.info("Successfully preprocessed the dataset and saved all outputs!")