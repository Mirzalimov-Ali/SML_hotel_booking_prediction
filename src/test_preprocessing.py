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

    def fillingTest(self, df: pd.DataFrame):
        df = df.copy()

        # ===== MISSING VALUES =====
        for col, val in self.imputers.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
                logger.info(f"[TEST] Filled missing values in column: {col}")

        return df

    # ===== ENCODING =====
    def encodingTest(self, df: pd.DataFrame):
        df = df.copy()

        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                # agar testda train’da yo‘q label bo‘lsa → error bo‘lmasligi uchun
                df[col] = df[col].astype(str)

                known_classes = set(encoder.classes_)
                default_class = encoder.classes_[0]

                df[col] = df[col].apply(
                    lambda x: x if x in known_classes else default_class
                )

                df[col] = encoder.transform(df[col])
                logger.info(f"[TEST] Applied Label Encoding to column: {col}")

        return df

    # ===== LOG TRANSFORMATION =====
    def logTransformationTest(self, df: pd.DataFrame):
        df = df.copy()

        for col in self.log_cols:
            if col in df.columns and (df[col] > 0).all():
                df[col] = np.log1p(df[col])
                logger.info(f"[TEST] Applied log1p to column: {col}")

        return df

    # ===== SCALING =====
    def scalingTest(self, df: pd.DataFrame):
        df = df.copy()

        for col, scaler in self.scalers.items():
            if col in df.columns and col not in self.target:
                df[col] = scaler.transform(df[[col]])
                logger.info(f"[TEST] Scaled column: {col}")

        logger.info("TEST preprocessing finished successfully")
        return df
    
    def transform(self, df, filled=False):
        if not filled:
            df = self.fillingTest(df)

        df = self.encodingTest(df)
        df = self.logTransformationTest(df)
        df = self.scalingTest(df)
        return df

