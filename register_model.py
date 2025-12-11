
import mlflow
from mlflow.tracking import MlflowClient

# Your best model run-id
best_run_id = "mlruns\517387353629759616\46cd37d6501b4ed2ba3748bfd8624469"

# Model URI inside the run's artifacts
model_uri = f"runs:/{best_run_id}/model"

# Model registry name
model_name = "Student_Grade_Models"

# Register a new version of the model
model_details = mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
    
)
#Add Alias (staging or production)
client = MlflowClient()

client.set_registered_model_alias(
    name=model_name,
    alias="staging",              # alias
    version=model_details.version
)

print("Alias 'staging' set successfully!")

# Print details
print(f"Model registered successfully!")
print(f"Model Name: {model_details.name}")
print(f"Model Version: {model_details.version}")
