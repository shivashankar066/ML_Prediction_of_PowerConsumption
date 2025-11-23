import shap
import numpy as np
import pandas as pd
import os
from scripts.utils import OUTPUT_DIR


def shap_feature_importance(model, X, top_n=5):
    """
    Returns a dataframe with Top-N most impactful SHAP features per row.
    """
    top_n = min(top_n, X.shape[1])
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)

    feature_names = X.columns

    top_features = []
    for row in np.abs(shap_values):
        top_idx = np.argsort(row)[::-1][:top_n]
        top_feat_names = [feature_names[i] for i in top_idx]
        top_features.append(top_feat_names)

    top_features_df = pd.DataFrame(
        top_features,
        columns=[f"Top_Feature_{i+1}" for i in range(top_n)]
    )

    return top_features_df


def generate_shap_report(model, full_test, full_target, last7, features, top_n=20):
    """
    Generates two XLSX files:
    1. test_predictions_with_shap.xlsx → full test dataset
    2. test_for_shap.xlsx → latest 7-days only
    """

    # -----------------------------
    # 1️⃣  FULL TEST DATA SHAP REPORT
    # -----------------------------

    X_test_full = full_test[features].reset_index(drop=True)
    preds_full = model.predict(X_test_full)

    full_df = full_test.reset_index(drop=True).copy()
    full_df["Predicted_Power"] = preds_full
    full_df["Actual_Power"] = full_target.values

    full_df["Error_%"] = (
        (full_df["Predicted_Power"] - full_df["Actual_Power"])
        / full_df["Actual_Power"]
    ) * 100

    # Anomaly flag
    full_df["Anomaly_Flag"] = np.where(
        np.abs(full_df["Error_%"]) > 20,
        np.where(full_df["Error_%"] > 0, "Overconsumption", "Underconsumption"),
        "Normal"
    )

    # SHAP for full test
    top_features_full = shap_feature_importance(model, X_test_full, top_n=top_n)

    full_final = pd.concat([full_df, top_features_full], axis=1)

    # Save file
    full_out_path = os.path.join(OUTPUT_DIR, "test_predictions_with_shap.xlsx")
    full_final.to_excel(full_out_path, index=False)

    print(f"✔ Full test SHAP file saved: {full_out_path}")

    # -----------------------------
    # 2️⃣  LAST 7 DAYS SHAP REPORT
    # -----------------------------

    last7_X = last7[features].reset_index(drop=True)
    preds_last7 = model.predict(last7_X)

    last7_df = last7.reset_index(drop=True).copy()
    last7_df["Predicted_Power"] = preds_last7

    # SHAP for last 7 days
    top7 = shap_feature_importance(model, last7_X, top_n=top_n)

    last7_final = pd.concat([last7_df, top7], axis=1)

    last7_out_path = os.path.join(OUTPUT_DIR, "test_for_shap.xlsx")
    last7_final.to_excel(last7_out_path, index=False)

    print(f"✔ Last 7-days SHAP file saved: {last7_out_path}")

    return full_final, last7_final
