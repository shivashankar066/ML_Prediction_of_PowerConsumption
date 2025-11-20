import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
LOG_DIR = os.path.join(BASE_DIR, 'logs')


def ensure_dirs():
    for d in [DATA_DIR, OUTPUT_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

def get_last_7_days(df):
    """
    Extract last 7 days of data based on the max(Date).
    """
    max_date = df["Date"].max()
    seven_day_start = max_date - pd.Timedelta(days=7)

    last_7_df = df[df["Date"] >= seven_day_start].reset_index(drop=True)
    return last_7_df
