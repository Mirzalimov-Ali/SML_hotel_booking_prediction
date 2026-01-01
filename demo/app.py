import pandas as pd
import gradio as gr
import requests

# ===============================
# FASTAPI BACKEND URL
# ===============================
API_URL = "http://127.0.0.1:8000/predict"


# ===============================
# PREDICT VIA API
# ===============================
def predict(
    hotel, lead_time, arrival_date_month, arrival_date_week_number,
    arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights,
    adults, children, babies, meal, country, market_segment, distribution_channel,
    is_repeated_guest, previous_cancellations, previous_bookings_not_canceled,
    reserved_room_type, assigned_room_type, booking_changes, deposit_type, agent,
    days_in_waiting_list, customer_type, adr, required_car_parking_spaces,
    total_of_special_requests, city
):
    payload = {
        "hotel": hotel,
        "lead_time": int(lead_time),
        "arrival_date_month": arrival_date_month,
        "arrival_date_week_number": int(arrival_date_week_number),
        "arrival_date_day_of_month": int(arrival_date_day_of_month),
        "stays_in_weekend_nights": int(stays_in_weekend_nights),
        "stays_in_week_nights": int(stays_in_week_nights),
        "adults": int(adults),
        "children": int(children),
        "babies": int(babies),
        "meal": meal,
        "country": country,
        "market_segment": market_segment,
        "distribution_channel": distribution_channel,
        "is_repeated_guest": int(is_repeated_guest),
        "previous_cancellations": int(previous_cancellations),
        "previous_bookings_not_canceled": int(previous_bookings_not_canceled),
        "reserved_room_type": reserved_room_type,
        "assigned_room_type": assigned_room_type,
        "booking_changes": int(booking_changes),
        "deposit_type": deposit_type,
        "agent": int(agent),
        "days_in_waiting_list": int(days_in_waiting_list),
        "customer_type": customer_type,
        "adr": float(adr),
        "required_car_parking_spaces": int(required_car_parking_spaces),
        "total_of_special_requests": int(total_of_special_requests),
        "city": city
    }

    response = requests.post(API_URL, json=payload, timeout=5)
    response.raise_for_status()
    result = response.json()

    return (
        result["prediction"],
        f"{result['cancellation_probability']:.2%}",
        result["risk_level"]
    )


# ===============================
# CUSTOM CSS
# ===============================
CUSTOM_CSS = """
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

.gr-box {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}

h1 {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
}

.gradient-text {
    background: linear-gradient(to right, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    color: transparent;
}

.emoji {
    color: initial !important;
    margin-right: 8px;
}

button {
    background: linear-gradient(135deg, #22c55e, #38bdf8) !important;
    color: black !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    height: 60px !important;
}

/* remove label background */
label span,
.gr-label span {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}
"""


# ===============================
# UI
# ===============================
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    <h1>
        <span class="emoji">🏨</span>
        <span class="gradient-text">Hotel Booking Cancellation Predictor</span>
    </h1>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📅 Booking Info")
            hotel = gr.Textbox(label="Hotel")
            lead_time = gr.Number(label="Lead Time")
            arrival_date_month = gr.Textbox(label="Arrival Month")
            arrival_date_week_number = gr.Number(label="Week Number")
            arrival_date_day_of_month = gr.Number(label="Day of Month")
            stays_in_weekend_nights = gr.Number(label="Weekend Nights")
            stays_in_week_nights = gr.Number(label="Week Nights")

        with gr.Column(scale=2):
            gr.Markdown("### 👥 Guests")
            adults = gr.Number(label="Adults")
            children = gr.Number(label="Children")
            babies = gr.Number(label="Babies")
            meal = gr.Textbox(label="Meal")
            country = gr.Textbox(label="Country")
            city = gr.Textbox(label="City")
            is_repeated_guest = gr.Number(label="Repeated Guest (0/1)")
            customer_type = gr.Textbox(label="Customer Type")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🛏 Room & Booking")
            reserved_room_type = gr.Textbox(label="Reserved Room")
            assigned_room_type = gr.Textbox(label="Assigned Room")
            booking_changes = gr.Number(label="Booking Changes")
            deposit_type = gr.Textbox(label="Deposit Type")
            agent = gr.Number(label="Agent")

        with gr.Column():
            gr.Markdown("### 💰 Pricing & History")
            adr = gr.Number(label="ADR")
            previous_cancellations = gr.Number(label="Previous Cancellations")
            previous_bookings_not_canceled = gr.Number(label="Prev. Not Canceled")
            days_in_waiting_list = gr.Number(label="Waiting List Days")
            required_car_parking_spaces = gr.Number(label="Parking Spaces")
            total_of_special_requests = gr.Number(label="Special Requests")

    # hidden but required by API
    market_segment = gr.Textbox(value="Online TA", visible=False)
    distribution_channel = gr.Textbox(value="TA/TO", visible=False)

    predict_btn = gr.Button("🚀 Predict Cancellation Risk")

    with gr.Row():
        prediction = gr.Textbox(label="Prediction")
        probability = gr.Textbox(label="Probability")
        risk = gr.Textbox(label="Risk Level")

    predict_btn.click(
        predict,
        inputs=[
            hotel, lead_time, arrival_date_month, arrival_date_week_number,
            arrival_date_day_of_month, stays_in_weekend_nights, stays_in_week_nights,
            adults, children, babies, meal, country, market_segment,
            distribution_channel, is_repeated_guest, previous_cancellations,
            previous_bookings_not_canceled, reserved_room_type, assigned_room_type,
            booking_changes, deposit_type, agent, days_in_waiting_list,
            customer_type, adr, required_car_parking_spaces,
            total_of_special_requests, city
        ],
        outputs=[prediction, probability, risk]
    )


if __name__ == "__main__":
    demo.launch()
