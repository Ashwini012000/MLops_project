import mlflow
import pandas as pd
import numpy as np
import joblib

# 1. Load model from MLflow Registry
model_name = "Student_Grade_Models"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@staging")

print("Model loaded successfully from MLflow Registry!")

# 2. New Data
new_data = {
    "Math": [0],
    "Science": [0],
    "English": [0],
    "History": [0],
    "Total": [0],
    "Percentage": [0]
}

df = pd.DataFrame(new_data)
print("\nInput Data:\n", df)

# 3. Predict (numeric)
numeric_pred = model.predict(df)
numeric_pred = np.array(numeric_pred).flatten()  # ensure array
print("\nNumeric Prediction:", numeric_pred)

# 4. Load Label Encoder
le = joblib.load("label_encoder.pkl")
print("Loaded object:", type(le))

# 5. Convert numeric → actual Grade
grade_string = le.inverse_transform([numeric_pred[0]])[0]

print("\nFinal Predicted Grade:", grade_string)
