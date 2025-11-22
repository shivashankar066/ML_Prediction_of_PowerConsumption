import shap
import numpy as np
import pandas as pd
import os
from scripts.utils import OUTPUT_DIR


def shap_feature_importance(model, X, top_n=5):
    """
    Returns a dataframe with Top-N most impactful SHAP features per row.
    """
    # explainer = shap.TreeExplainer(model)
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)

    feature_names = X.columns

    # Extract Top-N SHAP features per row
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


def generate_shap_report(model, test_df, features, out_path=None, top_n=5):
    """
    Generates an XLSX file including:
    - Actual values
    - Predicted values
    - Error percentage
    - Anomaly detection flag
    - Top-N SHAP impactful features per row
    """

    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, "test_predictions_with_shap.xlsx")

    X_test = test_df[features].reset_index(drop=True)
    preds = model.predict(X_test)

    # Core results dataframe
    result = test_df.reset_index(drop=True).copy()
    result["Predicted_Power"] = preds

    result["Error_%"] = (
        (result["Predicted_Power"] - result["Active_Energy_Delivered"])
        / result["Active_Energy_Delivered"]
    ) * 100

    # Flag anomalies if error > ±20%
    result["Anomaly_Flag"] = np.where(
        np.abs(result["Error_%"]) > 20,
        np.where(result["Error_%"] > 0, "Overconsumption", "Underconsumption"),
        "Normal"
    )

    # Compute SHAP top features
    top_features_df = shap_feature_importance(model, X_test, top_n=top_n)

    # Merge both
    final_df = pd.concat([result, top_features_df], axis=1)

    # Save excel
    final_df.to_excel(out_path, index=False)

    print("\n✔ Test predictions with SHAP saved to:")
    print(out_path)

    return final_df
