import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

MODEL_PATH = Path("pipeline/full_pipeline.joblib")

app = FastAPI(
    title="Hotel Booking Cancellation Prediction API",
    version="1.0"
)

pipeline = None  # important


# --------------------------------------------------
# Load model ON STARTUP
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")


# --------------------------------------------------
# Schemas
# --------------------------------------------------
class BookingInput(BaseModel):
    hotel: str
    lead_time: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: int
    babies: int
    meal: str
    country: str
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: int
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int
    city: str


class PredictionOutput(BaseModel):
    prediction: str
    cancellation_probability: float
    risk_level: str


# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Predict
# --------------------------------------------------
@app.post("/predict", response_model=PredictionOutput)
def predict(data: BookingInput):
    if pipeline is None:
        raise RuntimeError("Model not loaded")

    df = pd.DataFrame([data.model_dump()])

    pred = int(pipeline.predict(df)[0])
    proba = pipeline.predict_proba(df)[0][1]

    label = "Cancelled ❌" if pred == 1 else "Not Cancelled ✅"

    if proba >= 0.8:
        risk = "HIGH 🔴"
    elif proba >= 0.5:
        risk = "MEDIUM 🟠"
    else:
        risk = "LOW 🟢"

    return PredictionOutput(
        prediction=label,
        cancellation_probability=round(proba, 4),
        risk_level=risk
    )
