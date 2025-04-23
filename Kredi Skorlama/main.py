from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

Instrumentator().instrument(app).expose(app)

model = joblib.load("xgb_model.pkl")

class CreditInput(BaseModel):
    Age : int
    Income : int
    Number_of_Children: int 
    Gender_Male:bool
    Marital_Status_Single: bool
    Home_Ownership_Rented: bool
    Education_encoded: float


@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is active"}

@app.post("/predict")
def predict_credit_score(data: CreditInput):
    input_data = np.array([[data.Age,
                            data.Income,
                            data.Number_of_Children,
                            data.Gender_Male,
                            data.Marital_Status_Single,
                            data.Home_Ownership_Rented,
                            data.Education_encoded]])
    prediction = model.predict(input_data)
    score_map = {0:"Low", 1:"Average", 2:"High"}
    predicted_class = int(prediction[0])

    return {
        "prediction": predicted_class,
         "score_category": score_map[predicted_class]


    }




