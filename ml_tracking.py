import os
import sys
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow.tracking._tracking_service.client as ml_client
os.environ["NO_EMOJI"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "en_US.utf-8"


def no_emoji_print(self, run_id):
    try:
        sys.stdout.write(f"View run details in MLflow UI.\n")
    except:
        pass

ml_client.TrackingServiceClient._log_url = no_emoji_print

# Set MLflow to local folder (VERY IMPORTANT)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment name (CREATE IF NOT EXISTS)
mlflow.set_experiment("Student_Grade_Project")

# Paths
MODELS_DIR = "models"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"

# Load test dataset
if not os.path.exists(PROCESSED_DATA_PATH):
    raise FileNotFoundError(f"Test data not found: {PROCESSED_DATA_PATH}")

df_test = pd.read_csv(PROCESSED_DATA_PATH)
print(f" Loaded test data with {len(df_test)} rows")

# Ensure Expected column exists for evaluation
if "Grade" not in df_test.columns:
    raise ValueError("'Grade' column missing in test data! Cannot compute metrics.")

# Iterate through all trained models
model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]

if not model_files:
    raise FileNotFoundError(f"No models found in {MODELS_DIR}")

for file in model_files:
    model_path = os.path.join(MODELS_DIR, file)
    print(f"\n Loading model: {file}")

    # Load model dictionary
    model_data = joblib.load(model_path)
    model = model_data["model"]
    label_encoder = model_data["label_encoder"]
    feature_cols = model_data["feature_cols"]

    # Prepare data
    X_test = df_test[feature_cols]
    #actual grade
    y_true = df_test["Grade"]

    # Predictions
    y_pred_encoded = model.predict(X_test)
    #predicted grade
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    print( acc)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"{file} — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    model_name = file.replace(".joblib", "")

    # Start MLflow run
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({
            "model_name": model_name,
            "feature_count": len(feature_cols)
        })

        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        # Log model locally
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print(f" Logged model to MLflow: {model_name}")

print("\n All models logged successfully to MLflow!")