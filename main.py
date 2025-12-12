# import os
# import pandas as pd
# from src.data_processing import DataProcessor
# from src.model_training import ModelTrainer
# def main():
#     print(" Starting Student Grade Prediction Pipeline...")

#     # Define paths
#     raw_data_path = "data/raw/raw_data.csv"
#     processed_data_path = "data/processed/processed_data.csv"
#     model_save_path = "models/student_grade_model.joblib"

#     # Step 1: Load raw data
#     if not os.path.exists(raw_data_path):
#         raise FileNotFoundError(f"Raw data not found at: {raw_data_path}")
#     print(" Loading raw data...")
#     df_raw = pd.read_csv(raw_data_path)
#     print(f" Loaded {len(df_raw)} rows from {raw_data_path}")

#     # Step 2: Process data
#     print(" Cleaning and processing data...")
#     processor = DataProcessor(df_raw, save_path=processed_data_path)
#     processed_df = processor.run_pipeline()
#     print(f"Processed and saved data:{processed_df}")
    
#      # Step 4: Train the model
#     print(" Training model...")
#     trainer = ModelTrainer(processed_df)
#     trainer.train_models()
#    # trainer.save_best_model()
#     print(f" Model saved successfully at: {model_save_path}")

#     print(" Pipeline execution completed successfully!")

# if __name__ == "__main__":
#     main()

import os
import pandas as pd
from src.data_processing import DataProcessor
from src.model_training import ModelTrainer

def main():
    print("Starting Student Grade Prediction Pipeline...")

    # Define paths
    raw_data_path = "data/raw/raw_data.csv"
    processed_data_path = "data/processed/processed_data.csv"
    model_save_path = "models/student_grade_model.joblib"

    # Step 0: Ensure folders exist
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Step 1: Load raw data
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f"Raw data not found at: {raw_data_path}")
    print("Loading raw data...")
    df_raw = pd.read_csv(raw_data_path)
    print(f"Loaded {len(df_raw)} rows from {raw_data_path}")

    # Step 2: Process data
    print("Cleaning and processing data...")
    processor = DataProcessor(df_raw, save_path=processed_data_path)
    processed_df = processor.run_pipeline()
    print(f"Processed data saved at: {processed_data_path}")
    print(f"data saved")

    # # Step 3: Train the model
    # print("Training model...")
    # trainer = ModelTrainer(processed_df)
    # trainer.train_models()
    # # Optional: save best model locally
    # trainer.save_best_model(model_save_path)
    # print(f"Model saved successfully at: {model_save_path}")

    # print("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
