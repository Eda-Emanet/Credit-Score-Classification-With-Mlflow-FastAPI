from fastapi import FastAPI
import mlflow.sklearn
from pydantic import BaseModel
import pandas as pd

app = FastAPI()
mlflow.set_tracking_uri("http://localhost:5000")
# model = mlflow.sklearn.load_model('runs:/f8383a2fa3e94dfab02f7790fa237375/iris_model')

# MLflow'tan modeli yükle (models:/ModelName/Stage)
model_uri = 'runs:/264d47a33b9b4b208a48fec20070605a/model'
model = mlflow.sklearn.load_model(model_uri)

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
