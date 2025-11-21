# import joblib
# import os
# import shutil

# # Path where your models are stored
# models_dir = "models"

# # Decision Tree model path
# decision_tree_path = os.path.join(models_dir, "DecisionTree_model.joblib")

# # Confirm file exists
# if not os.path.exists(decision_tree_path):
#     raise FileNotFoundError("DecisionTree_model.joblib not found in models folder")

# # Folder to save best model
# best_model_dir = os.path.join(models_dir, "best_model")
# #print(best_model_dir)
# os.makedirs(best_model_dir, exist_ok=True)

# # Save best model as best_model.joblib
# best_model_save_path = os.path.join(best_model_dir, "best_model.joblib")

# # Copy the model file
# shutil.copy(decision_tree_path, best_model_save_path)

# print("Best model saved successfully!")
# print(f"Best model path: {best_model_save_path}")

import joblib
import os

MODELS_DIR = "models"
BEST_MODEL_DIR = "best_model"
BEST_MODEL_NAME = "RandomForest_model.joblib"

# path of the trained model
randomforest_model_path = os.path.join(MODELS_DIR, BEST_MODEL_NAME)
print(randomforest_model_path)

if not os.path.exists(randomforest_model_path):
    raise FileNotFoundError(f"Randomforest model not found at {randomforest_model_path}")

# Load trained DecisionTree model (this is a dictionary)
saved_model = joblib.load(randomforest_model_path)
print(saved_model)

# Create best_model directory
print("creating dir")
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

best_model_path = os.path.join(BEST_MODEL_DIR, BEST_MODEL_NAME)

# Save it exactly as it was
joblib.dump(saved_model, best_model_path)
print("model saved at saved_model")

print(f" Best model copied successfully to: {best_model_path}")
