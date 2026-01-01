import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime
import numpy as np
from src.logger import get_logger

logger = get_logger('feature_engineering', 'feature_engineering.log')

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['total_nights'] = X['stays_in_weekend_nights'] + X['stays_in_week_nights']
        X['lead_time_log'] = np.log1p(X['lead_time'])
        X['is_extreme_lead'] = (X['lead_time'] > 365).astype(int)

        X['total_guests'] = X['adults'] + X['children'] + X['babies']

        X['is_free_booking'] = (X['adr'] == 0).astype(int)
        X['has_previous_cancel'] = (X['previous_cancellations'] > 0).astype(int)
        X['is_loyal_customer'] = ((X['previous_bookings_not_canceled'] > 0) |(X['is_repeated_guest'] == 1)).astype(int)

        X['room_changed'] = (X['reserved_room_type'] != X['assigned_room_type']).astype(int)
        X['has_agent'] = X['agent'].notnull().astype(int)

        X['is_short_stay'] = (X['total_nights'] <= 2).astype(int)
        X['is_long_stay'] = (X['total_nights'] >= 7).astype(int)
        X['weekend_ratio'] = X['stays_in_weekend_nights'] / (X['total_nights'] + 1)

        X['is_last_minute'] = (X['lead_time'] <= 7).astype(int)
        X['adr_x_lead_time'] = X['adr'] * X['lead_time_log']
        X['high_adr'] = (X['adr'] > X['adr'].median()).astype(int)

        X['adr_per_person'] = X['adr'] / (X['adults'] + X['children'] + 1)

        X['is_suspicious_price'] = (
            (X['adr'] == 0) & (X['total_of_special_requests'] == 0)
        ).astype(int)

        X['has_children'] = (X['children'] > 0).astype(int)
        X['has_babies'] = (X['babies'] > 0).astype(int)
        X['solo_traveler'] = (X['total_guests'] == 1).astype(int)

        X['many_booking_changes'] = (X['booking_changes'] >= 2).astype(int)
        X['was_waitlisted'] = (X['days_in_waiting_list'] > 0).astype(int)

        X['many_special_requests'] = (X['total_of_special_requests'] >= 2).astype(int)
        X['needs_parking'] = (X['required_car_parking_spaces'] > 0).astype(int)

        high_season_months = ['June', 'July', 'August', 'December']
        X['is_high_season'] = X['arrival_date_month'].isin(high_season_months).astype(int)

        X['online_booking'] = X['market_segment'].isin(
            ['Online TA', 'Offline TA/TO']
        ).astype(int)

        X['agent_booking'] = (
            X['agent'].notnull() & 
            X['market_segment'].isin(['Online TA', 'Offline TA/TO'])
        ).astype(int)

        X['risky_customer'] = (
            (X['previous_cancellations'] > 0) &
            (X['previous_bookings_not_canceled'] == 0)
        ).astype(int)

        X['room_mismatch_online'] = (
            (X['reserved_room_type'] != X['assigned_room_type']) &
            X['market_segment'].isin(['Online TA', 'Offline TA/TO'])
        ).astype(int)

        logger.info("New engineered features successfully added.")
        return X

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'logger' in state:
            del state['logger']
        return state