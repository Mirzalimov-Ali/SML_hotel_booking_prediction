import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class Preprocessing:
    def __init__(self, csv_path, target):
        self.csv_path = csv_path
        self.target = target

        self.fill_values = {}
        self.label_encoders = {}
        self.onehot_cols = {}
        self.scaler = None
        self.num_cols = None

    # =====================
    # LOAD & SPLIT
    # =====================
    def load_data(self):
        df = pd.read_csv(self.csv_path)

        df.drop(['reservation_status', 'reservation_status_date', 'company'], axis=1, inplace=True)

        X = df.drop(self.target, axis=1)
        y = df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
        )

    # =====================
    # MISSING VALUES
    # =====================
    def filling_train(self):
        for col in self.X_train.columns:
            if self.X_train[col].dtype == 'object':
                val = self.X_train[col].mode()[0]
            else:
                val = self.X_train[col].median()

            self.fill_values[col] = val
            self.X_train[col] = self.X_train[col].fillna(val)

    def filling_test(self):
        for col, val in self.fill_values.items():
            self.X_test[col] = self.X_test[col].fillna(val)

    # =====================
    # ENCODING
    # =====================
    def encode_train(self, onehot_threshold=5):
        for col in self.X_train.columns:
            if self.X_train[col].dtype == 'object':
                if self.X_train[col].nunique() <= onehot_threshold:
                    self.onehot_cols[col] = self.X_train[col].unique()
                else:
                    le = LabelEncoder()
                    self.X_train[col] = le.fit_transform(self.X_train[col])
                    self.label_encoders[col] = le

        self.X_train = pd.get_dummies(
            self.X_train,
            columns=self.onehot_cols.keys(),
            drop_first=True
        )

    def encode_test(self):
        for col, le in self.label_encoders.items():
            self.X_test[col] = self.X_test[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

        self.X_test = pd.get_dummies(
            self.X_test,
            columns=self.onehot_cols.keys(),
            drop_first=True
        )

        self.X_test = self.X_test.reindex(
            columns=self.X_train.columns,
            fill_value=0
        )

    # =====================
    # SCALING
    # =====================
    def scale_train(self):
        self.num_cols = self.X_train.select_dtypes(
            include=['int64', 'float64']
        ).columns

        self.scaler = StandardScaler()
        self.X_train[self.num_cols] = self.scaler.fit_transform(
            self.X_train[self.num_cols]
        )

    def scale_test(self):
        self.X_test[self.num_cols] = self.scaler.transform(
            self.X_test[self.num_cols]
        )

    # =====================
    # FULL RUN
    # =====================
    def run_all(self):
        self.load_data()

        self.filling_train()
        self.filling_test()

        self.encode_train()
        self.encode_test()

        self.scale_train()
        self.scale_test()