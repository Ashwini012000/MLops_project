import mlflow
import joblib

def load_model_and_encoder():

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    model_name = "Student_Grade_Models"

    print(" Loading MLflow model from Registry (staging)…")
    model = mlflow.pyfunc.load_model(f"models:/{model_name}@staging")

    print(" Loading Label Encoder…")
    encoder = joblib.load("label_encoder.pkl")

    return model, encoder
