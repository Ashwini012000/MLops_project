from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import mlflow

app = FastAPI()
templates = Jinja2Templates(directory="deployment/templates")

# Load MLflow Model
model_name = "Student_Grade_Models"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@staging")

# Load Label Encoder
encoder = joblib.load("label_encoder.pkl")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(
    request: Request,
    Math: int = Form(...),
    Science: int = Form(...),
    English: int = Form(...),
    History: int = Form(...)
):
    total = Math + Science + English + History
    percentage = total / 4

    df = pd.DataFrame([{
        "Math": Math,
        "Science": Science,
        "English": English,
        "History": History,
        "Total": total,
        "Percentage": percentage
    }])

    pred = model.predict(df)
    final_grade = encoder.inverse_transform(pred)[0]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "grade": final_grade
    })
