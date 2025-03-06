from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

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

    return {"prediction": prediction}


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

    return {"prediction": prediction}


