from scripts.utils import ensure_dirs, DATA_DIR, get_last_7_days
from scripts.train_pipeline import hvac_pipeline
from scripts.explain_shap import generate_shap_report
import os

ensure_dirs()
DATA_PATH = os.path.join(DATA_DIR, 'HVAC_data.xlsx')

if __name__ == '__main__':
    print('Running HVAC pipeline...')

    # Run pipeline
    all_metrics, test_result, best_model, features = hvac_pipeline(DATA_PATH)

    # Get last 7 days from test_result
    last_7_days = get_last_7_days(test_result)

    # Generate SHAP report using updated function
    generate_shap_report(
        model=best_model,
        full_test=test_result,
        full_target=test_result["Active_Energy_Delivered"],
        last7=last_7_days,
        features=features
    )

    print('Done.')
