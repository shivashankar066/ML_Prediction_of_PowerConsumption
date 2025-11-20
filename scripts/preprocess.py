import pandas as pd
import numpy as np

def preprocess_data(df):
    print("Initial data shape:", df.shape)
    print("Initial columns:", df.columns.tolist())

    # --- Name filtering ---
    if 'Name' in df.columns:
        df['Name'] = df['Name'].replace('AHU1.1', 'AHU.1')
        df = df[df['Name'].str.contains('HVAC', na=False)]
        print(f"Shape after filtering 'HVAC' names: {df.shape}")
    else:
        print("⚠️ 'Name' column not found, skipping name-based filtering.")

    # --- Target variable check ---
    if 'Active_Energy_Delivered' not in df.columns:
        raise ValueError("Target column 'Active_Energy_Delivered' not found.")
    df['Active_Energy_Delivered'] = pd.to_numeric(df['Active_Energy_Delivered'], errors='coerce')
    df.dropna(subset=['Active_Energy_Delivered'], inplace=True)
    df = df[(df['Active_Energy_Delivered'] >= 0) & (df['Active_Energy_Delivered'] < 120)]
    print(f"Shape after filtering target: {df.shape}")

    # --- Date Handling ---
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        invalid = df['Date'].isna().sum()
        df.dropna(subset=['Date'], inplace=True)
        print(f"Dropped {invalid} rows due to invalid 'Date'.")
        df = df.sort_values(by='Date')
    else:
        raise ValueError("Column 'Date' not found in dataset.")

    # --- Jumbo Temperature Imputation ---
    temp_cols_exist = all(c in df.columns for c in ['Jumbo_Temp1', 'Jumbo_Temp2', 'Jumbo_Temp3'])
    if temp_cols_exist:
        for col in ['Jumbo_Temp1', 'Jumbo_Temp2', 'Jumbo_Temp3']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Jumbo_Temp1'] = df['Jumbo_Temp1'].fillna(df['Jumbo_Temp2']).fillna(df['Jumbo_Temp3'])
        df.drop(columns=['Jumbo_Temp2', 'Jumbo_Temp3'], inplace=True)
        df['Jumbo_Temp1'] = df['Jumbo_Temp1'].ffill().bfill()
        print("✅ Imputed Jumbo_Temp1")
    else:
        print("⚠️ Missing Jumbo_Temp columns, skipped imputation.")

    # --- Jumbo Humidity Imputation ---
    humidity_cols_exist = all(c in df.columns for c in ['Jumbo_Humidity', 'Jumbo_Humidity3'])
    if humidity_cols_exist:
        for col in ['Jumbo_Humidity', 'Jumbo_Humidity3']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['Jumbo_Humidity'] = df['Jumbo_Humidity'].fillna(df['Jumbo_Humidity3'])
        df.drop(columns=['Jumbo_Humidity3'], inplace=True)
        df['Jumbo_Humidity'] = df['Jumbo_Humidity'].ffill().bfill()
        print("✅ Imputed Jumbo_Humidity")
    else:
        print("⚠️ Missing Jumbo_Humidity columns, skipped imputation.")

    # --- Feature Engineering ---
    if {'Avg_Return_Water_Temp', 'Avg_Supply_water_Temp'} <= set(df.columns):
        df['Compressor_delta'] = (
            pd.to_numeric(df['Avg_Return_Water_Temp'], errors='coerce') -
            pd.to_numeric(df['Avg_Supply_water_Temp'], errors='coerce')
        )
        print("✅ Created 'Compressor_delta'")

    # --- Range Filters ---
    if 'T2M' in df.columns:
        df = df[(df['T2M'] > 0) & (df['T2M'] < 50)]
    if 'Operating_Hours' in df.columns:
        df = df[(df['Operating_Hours'] > 0) & (df['Operating_Hours'] < 1.1)]

    # --- Final Fill ---
    df = df.ffill().bfill()

    print("✅ Preprocessing complete.")
    print(f"Final cleaned data shape: {df.shape}")
    return df