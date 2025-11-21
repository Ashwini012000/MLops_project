import mlflow
best_run_id = "7b53ff1296594401b596133dcfd7908c"
mlflow.set_tracking_uri("file:///C:/Users/SKAs328/Desktop/MLOps_project/mlruns")

model_uri = f"runs:/{best_run_id}/RandomForest_model"
mlflow.register_model(
    model_uri=model_uri,
    name="Student_Grade_Models"
)
