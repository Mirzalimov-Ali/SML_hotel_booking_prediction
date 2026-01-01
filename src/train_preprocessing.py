import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import KNNImputer
from src.logger import get_logger

logger = get_logger('preprocessing TRAIN', 'preprocessing.log')

class PreprocessingTrain(BaseEstimator, TransformerMixin):
    def __init__(self, df: pd.DataFrame = None, target=None, log_transform=True):
        self.df = df.copy() if df is not None else None
        self.target = target if isinstance(target, list) else [target] if target else []
        self.log_transform = log_transform
        
        self.imputers = {}
        self.label_encoders = {}
        self.scalers = {}
        self.log_cols = []

    # ===== MISSING VALUES =====
    def fillingTrain(self):
        self.imputers = {}

        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        cat_cols = self.df.select_dtypes(include=['object']).columns

        # ------ Numerical → KNN IMPUTER ------
        knn_imputer = KNNImputer(n_neighbors=5)
        self.df[num_cols] = knn_imputer.fit_transform(self.df[num_cols])
        self.imputers["knn"] = knn_imputer

        logger.info(f"Filled NUMERIC columns using KNNImputer: {list(num_cols)}")

        # ------ Categorical → MODE ------
        for col in cat_cols:
            val = self.df[col].mode()[0]
            self.imputers[col] = val          
            self.df[col] = self.df[col].fillna(val)

            logger.info(f"Filled CATEGORICAL column '{col}' with MODE: {self.df[col].mode()[0]}")

        logger.info("Missing values filled successfully (TRAIN)")
        return self

    # ===== ENCODING =====
    def encodingTrain(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                self.df[[col]] = le.fit_transform(self.df[[col]])
                self.label_encoders[col] = le
                logger.info(f"Applied Ordinal Encoding to column: {col}")

        logger.info("All categorical features encoded using Ordinal Encoding.")
        return self
    
    # ===== LOG TRANSFORMATION =====
    def logTransformationTrain(self):
        if not self.log_transform:
            return self

        skewness = self.df.select_dtypes(include=['int64', 'float64']).skew()
        self.log_cols = skewness[skewness >= 0.5].index.tolist()

        for col in self.log_cols:
            if (self.df[col] > 0).all():
                self.df[col] = np.log1p(self.df[col])

        logger.info(f"Log transformation applied to: {self.log_cols}")
        return self
    
    # ===== SCALING =====
    def scalingTrain(self):
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        numeric_cols = [c for c in numeric_cols if c not in self.target]

        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        self.scalers["numeric"] = scaler

        logger.info(f"Numerical scaling applied to {len(numeric_cols)} columns using StandardScaler()")
        return self

    # ===== FIT =====
    def fit(self, X, y=None):
        self.df = X.copy()

        self.fillingTrain()
        self.encodingTrain()
        self.logTransformationTrain()
        self.scalingTrain()

        return self

    # # ===== TRANSFORM =====
    def transform(self, X: pd.DataFrame):
        df = X.copy()

        # missing values
        # ------ Numerical → KNN ------
        if "knn" in self.imputers:
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            df[num_cols] = self.imputers["knn"].transform(df[num_cols])
            logger.info("Applied KNNImputer to numerical columns")

        # ------ Categorical → MODE ------
        for col, val in self.imputers.items():
            if col != "knn" and col in df.columns:
                df[col] = df[col].fillna(val)
                logger.info(f"Applied MODE imputer to column: {col}")

        # label encoding
        for col, enc in self.label_encoders.items():
            if col in df.columns:
                df[[col]] = enc.transform(df[[col]])
                logger.info(f"Applied Ordinal Encoding to column: {col}")

        # log transform
        for col in self.log_cols:
            if col in df.columns and (df[col] > 0).all():
                df[col] = np.log1p(df[col])
                logger.info(f"Applied log1p transformation to: {col}")

        # scaling
        if "numeric" in self.scalers:
            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            num_cols = [c for c in num_cols if c not in self.target]

            df[num_cols] = self.scalers["numeric"].transform(df[num_cols])
            logger.info("Scaled numerical columns using StandardScaler")

        return df

    
    # ===== RETURN DATASET =====
    def getDataset(self):
        return self.df