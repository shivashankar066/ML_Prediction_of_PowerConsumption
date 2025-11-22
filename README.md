📘 HVAC Power Consumption Prediction – ML Pipeline

This project builds an end-to-end Machine Learning pipeline to predict power consumption for an HVAC system using historical meter data.
It includes data preprocessing, model training, evaluation, SHAP explainability, and automated CI/CD using GitHub Actions.
🚀 Features
✅ Complete ML Pipeline

Data preprocessing

Feature engineering

Train/test split

Model training (Random Forest, XGBoost, etc.)

Model selection

Metrics generation (RMSE, R², MAE, MAPE)

SHAP explainability

✅ Automated CI/CD Pipeline

Uses GitHub Actions to run the pipeline:

On every push to main

On manual trigger

Automatically every day at 2 AM UTC

Outputs include:

Trained model files

SHAP plots

Metrics CSV

Saved inside outputs/ and uploaded as GitHub Artifacts

Project Structure:

ML_Prediction_of_PowerConsumption/
│
├── scripts/
│   ├── main.py               # Main execution script
│   ├── utils.py              # Path & directory utilities
│   ├── train_pipeline.py     # Full model training pipeline
│   └── explain_shap.py       # SHAP explainability generator
│
├── outputs/                  # Generated models, SHAP plots, metrics
│   ├── shap_summary_plot.png
│   ├── metrics.csv
│   └── best_model.pkl
│
├── data/
│   └── HVAC_data.xlsx        # Input dataset
│
├── requirements.txt          # Python dependencies
├── .github/
│   └── workflows/
│       └── ci_cd.yml         # CI/CD pipeline definition
│
└── README.md                 # Project documentation


🧠 How This Works
1️⃣ Data is read from data/HVAC_data.xlsx

Daily-appending machine-meter data can be plugged in directly.

2️⃣ ML pipeline trains and evaluates models

The pipeline:

Cleans data

Encodes/normalizes features

Tests multiple ML models

Selects the best model

Saves results into outputs/

3️⃣ SHAP report is automatically generated

Explains model behavior feature-wise.


CI/CD Pipeline (GitHub Actions)

Your pipeline performs:

Install dependencies

Run the HVAC pipeline

Save SHAP plots + metrics

Upload outputs as artifacts

Users can download artifacts directly from the Actions → Workflow Run → Artifacts section.