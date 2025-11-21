# import pandas as pd
# import joblib
# from src.data_processing import DataProcessor   # if you have this class

# def main():
#     print("Loading best model...")
#     model = joblib.load("models/best_model/best_model.joblib")
#     print("Model loaded!")
#     save_path= "data/precited.csv"

#     # Load raw prediction input
#     input_path = "data/prediction_input.csv"
#     df = pd.read_csv(input_path)
#     print("\nLoaded columns:", df.columns.tolist())


#     print("Raw input:")
#     print(df.head())

#     # Process input data
#     processor = DataProcessor(df,save_path)
#     #print(processor)
#     df_processed = processor.clean_data()

#     # Predict
#     predictions = model.predict(df_processed)

#     df["Predicted_Grade"] = predictions

#     # Save predictions
#     output_path = "data/predictions_output.csv"
#     df.to_csv(output_path, index=False)

#     print(f"Prediction completed! Saved at: {output_path}")

# if __name__ == "__main__":
#     main()

import os
import pandas as pd
import joblib
from src.data_processing import DataProcessor


BEST_MODEL_PATH = "best_model\RandomForest_model.joblib"
PREDICTION_INPUT_PATH = "data/prediction_input.csv"
PREDICTION_OUTPUT_PATH = "data/predictions/predicted_grades.csv"


def main():

    print("\nStarting Prediction Pipeline...\n")

    # Step 1: Load prediction input CSV
    if not os.path.exists(PREDICTION_INPUT_PATH):
        raise FileNotFoundError(f"Prediction input file not found at: {PREDICTION_INPUT_PATH}")

    df = pd.read_csv(PREDICTION_INPUT_PATH)
    print(f"Loaded {len(df)} rows for prediction")

    # Step 2: Clean and process data
    processor = DataProcessor(df,PREDICTION_INPUT_PATH)
    df_processed = processor.clean_data()
    

    # Step 3: Load trained best model
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at: {BEST_MODEL_PATH}")

    saved_model = joblib.load(BEST_MODEL_PATH)

    model = saved_model["model"]
    label_encoder = saved_model["label_encoder"]
    feature_cols = saved_model["feature_cols"]

    print(f"Loaded Best Model from: {BEST_MODEL_PATH}")
    print(f"Features used for prediction: {feature_cols}")

    # Step 4: Prepare input features
    X_new = df_processed[feature_cols]

    # Step 5: Make predictions
    encoded_predictions = model.predict(X_new)
    predictions = label_encoder.inverse_transform(encoded_predictions)

    df_processed["Predicted_Grade"] = predictions

    # Step 6: Save output
    os.makedirs(os.path.dirname(PREDICTION_OUTPUT_PATH), exist_ok=True)
    df_processed.to_csv(PREDICTION_OUTPUT_PATH, index=False)

    print("\nPrediction Completed Successfully!")
    print(f"Saved predictions at: {PREDICTION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
