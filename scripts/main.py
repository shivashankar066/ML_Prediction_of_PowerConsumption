from utils import ensure_dirs, DATA_DIR
from train_pipeline import hvac_pipeline
from explain_shap import generate_shap_report
import os

ensure_dirs()
DATA_PATH = os.path.join(DATA_DIR, 'HVAC_data.xlsx')
if __name__ == '__main__':
    print('Running HVAC pipeline...')
    all_metrics, test_result, best_model, features = hvac_pipeline(DATA_PATH)
# Generate SHAP report using the chosen best model
    generate_shap_report(best_model, test_result, features)
    print('Done.')