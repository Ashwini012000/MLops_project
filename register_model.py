import mlflow
<<<<<<< HEAD
best_run_id = "7b53ff1296594401b596133dcfd7908c"

=======
best_run_id = "483edabbb16546a8883ba3553819281e"
>>>>>>> 08f27187bb00986ae3034340cec852a5941135c9

model_uri = f"runs:/{best_run_id}/model"
model_details= mlflow.register_model(
    model_uri=model_uri,
    name="Student_Grade_Models"
)

print(f"Model registered with name: {model_details.name} and version: {model_details.version}")