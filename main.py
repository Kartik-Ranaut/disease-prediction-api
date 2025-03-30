from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domain for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#define a route at root web address('/')
@app.get('/')
def read():
    return {"message":"Hello, Fastapi"}

class Data(BaseModel):
    name:str
    age:int

@app.post('/post')
def posting(data:Data):
    return{"message":data}




# Load the trained model
model = joblib.load("model.sav")

# Define input data format
class ModelInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/heartdiseasepredict")
def predict(data: ModelInput):
    input_features = np.array([
    data.age, 
    data.sex, 
    data.cp, 
    data.trestbps, 
    data.chol, 
    data.fbs, 
    data.restecg, 
    data.thalach, 
    data.exang, 
    data.oldpeak, 
    data.slope, 
    data.ca, 
    data.thal
    ]).reshape(1, -1)

# Make prediction
    prediction = model.predict(input_features).tolist()

    probability = model.predict_proba(input_features)[0][1]
    return {
            "prediction": prediction,  # Ensure it's an integer
            "probability": round(probability * 100, 2)  # Convert to percentage with 2 decimal places
        }

# Load the trained model
modeld = joblib.load("modeld.sav")

# Define input data format
class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int



@app.post("/diabetespredict")
def predict(data: ModelInput):
    input_features = np.array([
        data.Pregnancies, 
        data.Glucose, 
        data.BloodPressure, 
        data.SkinThickness, 
        data.Insulin, 
        data.BMI, 
        data.DiabetesPedigreeFunction, 
        data.Age
    ]).reshape(1, -1)

# Make prediction
    prediction = modeld.predict(input_features).tolist()
    probability = modeld.predict_proba(input_features)[0][1]
    return {
            "prediction": prediction,  # Ensure it's an integer
            "probability": round(probability * 100, 2)  # Convert to percentage with 2 decimal places
        }

