import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger('preprocessing TEST', 'preprocessing.log')

class PreprocessingTest:
    def __init__(self, preprocessing_train):
        self.imputers = preprocessing_train.imputers
        self.label_encoders = preprocessing_train.label_encoders
        self.scalers = preprocessing_train.scalers
        self.log_cols = preprocessing_train.log_cols
        self.target = preprocessing_train.target

    # ===== MISSING VALUES =====
    def fillingTest(self, df):
        df = df.copy()

        if "knn" in self.imputers:
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[num_cols] = self.imputers["knn"].transform(df[num_cols])

        for col, val in self.imputers.items():
            if col != "knn" and col in df.columns:
                df[col] = df[col].fillna(val)

        return df


    # ===== ENCODING =====
    def encodingTest(self, df):
        df = df.copy()

        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[[col]] = encoder.transform(df[[col]])
                logger.info(f"[TEST] Applied Ordinal Encoding to column: {col}")

        return df

    # ===== LOG TRANSFORMATION =====
    def logTransformationTest(self, df):
        df = df.copy()

        for col in self.log_cols:
            if col in df.columns and (df[col] > 0).all():
                df[col] = np.log1p(df[col])
                logger.info(f"[TEST] Applied log1p to column: {col}")

        return df

    # ===== SCALING =====
    def scalingTest(self, df: pd.DataFrame):
        df = df.copy()

        if "numeric" in self.scalers:
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            num_cols = [c for c in num_cols if c not in self.target]

            df[num_cols] = self.scalers["numeric"].transform(df[num_cols])
            logger.info("[TEST] Scaled numeric columns using StandardScaler")

        return df

    
    # ===== TRANSFORM =====
    def transform(self, df, filled=False):
        if not filled:
            df = self.fillingTest(df)

        df = self.encodingTest(df)
        df = self.logTransformationTest(df)
        df = self.scalingTest(df)
        return df

