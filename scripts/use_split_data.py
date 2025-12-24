import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/raw/hotel_bookings_updated_2024.csv")

drop_cols = [
    "is_canceled",
    "arrival_date_year",
    "reservation_status",
    "reservation_status_date",
    "company"
]

x = df.drop(drop_cols, axis=1)
y = df["is_canceled"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

os.makedirs("data/split", exist_ok=True)

x_train.to_csv("data/split/x_train.csv", index=False)
x_test.to_csv("data/split/x_test.csv", index=False)
y_train.to_csv("data/split/y_train.csv", index=False)
y_test.to_csv("data/split/y_test.csv", index=False)