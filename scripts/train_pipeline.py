import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
# from preprocess import preprocess_data
from scripts.models import get_models
from scripts.utils import DATA_DIR, OUTPUT_DIR
from sklearn.model_selection import train_test_split
import pandas as pd
from scripts.utils import get_last_7_days, OUTPUT_DIR
from scripts.explain_shap import generate_shap_report
from scripts.preprocess import preprocess_data

def load_and_prepare(path):
    # Auto-detect Excel or CSV
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
        df = df.fillna(0)
    else:
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore", parse_dates=["Date"])
    df = preprocess_data(df)
    # Encode Machine_Name (RandomForest cannot take text)
    le = LabelEncoder()
    df["Machine_Encoded"] = le.fit_transform(df["Name"])

    # Independent variables
    features = [
        "Machine_Encoded", "T2M", "Operating_Hours", "Jumbo_Temp1", "Jumbo_Humidity",
        "Average_Voltage_Line_to_Line", #"Average_Voltage_Line_to_Neutral",
        "Avg_Supply_water_Temp", "Avg_Return_Water_Temp", "Compressor_delta",
        "1st_Shift", "2nd_Shift", "common", "General", "hour", "Month", "Day"
    ]

    return df, features, le



def split_random(df):
    """
    Random 80/20 split for model training.
    """
    train, test = train_test_split(df, test_size=0.2)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------
# 3️⃣ Evaluate Model
# ---------------------------------------------

def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    metrics = {
        "Train": {
            "MSE": mse_train,
            "RMSE": rmse_train,
            "R2": r2_score(y_train, y_pred_train),
        },
        "Test": {
            "MSE": mse_test,
            "RMSE": rmse_test,
            "R2": r2_score(y_test, y_pred_test),
        },
        "OOB": {
            "R2": getattr(model, "oob_score_", None)
        }
    }
    return metrics, y_pred_train, y_pred_test


# ---------------------------------------------
# 5️⃣ Complete Pipeline (multiple models)
# ---------------------------------------------

def hvac_pipeline(data_path):
    df, features, le = load_and_prepare(data_path)
    train, test = split_random(df)
    X_train, y_train = train[features], train["Active_Energy_Delivered"]
    X_test, y_test = test[features], test["Active_Energy_Delivered"]
    print("training columns:", X_train.columns.tolist())
    print("testing columns:", X_test.columns.tolist())
    models = get_models()
    all_metrics = {}
    all_predictions = {}

    best_model = None
    best_r2 = -999

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics, y_pred_train, y_pred_test = evaluate_model(model, X_train, y_train, X_test, y_test)
        all_metrics[name] = metrics
        all_predictions[name] = (y_pred_test, model)

        test_r2 = metrics['Test']['R2']
        if test_r2 > best_r2:
            best_r2 = test_r2
            best_model = model

    # choose best_model for SHAP/exports
    chosen_preds, _ = all_predictions[list(all_predictions.keys())[0]]
    # create a base test_result for the best_model
    y_pred_best = all_predictions[[k for k,v in all_predictions.items() if getattr(v[1], 'feature_importances_', None) is not None][0]][0] if True else all_predictions[list(all_predictions.keys())[0]][0]

    # Use best_model for creating shap and final excel
    best_model_name = None
    for k, v in all_metrics.items():
        if v['Test']['R2'] == best_r2:
            best_model_name = k
            break

    print(f"Best model by Test R2: {best_model_name} (R2={best_r2})")

    # Evaluate chosen best model predictions
    chosen_pred_array, chosen_model = all_predictions[best_model_name]

    test_result = test.copy().reset_index(drop=True)
    test_result["Predicted_Power"] = chosen_pred_array
    test_result["Error_%"] = ((test_result["Predicted_Power"] - test_result["Active_Energy_Delivered"]) / test_result["Active_Energy_Delivered"]) * 100

    test_result["Anomaly_Flag"] = np.where(
        np.abs(test_result["Error_%"]) > 20,
        np.where(test_result["Error_%"] > 0, "Overconsumption", "Underconsumption"),
        "Normal"
    )

    last_7_days = get_last_7_days(test)
    generate_shap_report(chosen_model, last_7_days, features)

    # Save metrics summary
    metrics_summary = []
    for name, met in all_metrics.items():
        metrics_summary.append({
            'Model': name,
            'Train_RMSE': met['Train']['RMSE'],
            'Train_R2': met['Train']['R2'],
            'Test_RMSE': met['Test']['RMSE'],
            'Test_R2': met['Test']['R2']
        })

    metrics_df = pd.DataFrame(metrics_summary)
    metrics_out = os.path.join(OUTPUT_DIR, 'metrics.csv')
    metrics_df.to_csv(metrics_out, index=False)
    print(f"Saved metrics -> {metrics_out}")

    # Save a CSV of test (used by SHAP script)
    test_csv_path = os.path.join(OUTPUT_DIR, 'test_for_shap.csv')
    test_result.to_csv(test_csv_path, index=False)

    return all_metrics, test_result, chosen_model, features