from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import Optional




class PredictionItem(BaseModel):
    PassengerId : Optional[int]#4,
    Pclass : Optional[int]#1,
    Name : Optional[str]#'Futrelle, Mrs. Jacques Heath (Lily May Peel)',
    Sex : Optional[str]#'female',
    Age : Optional[float]#35.0,
    SibSp : Optional[int]#1,
    Parch : Optional[int]#0,
    Ticket : Optional[str]#'13803',#
    Fare : Optional[float]#53.1,
    Cabin : Optional[str]#'C123',
    Embarked : Optional[str]#'S'

model = joblib.load("model.joblib")


app = FastAPI()

@app.post("/prediction")
def prediction_endpoint(item : PredictionItem):
    data = load_data(item)
    data = transform_data(data)
    prediction = get_prediction(data)
    return prediction



def load_data(item : PredictionItem):
    data = pd.DataFrame(pd.Series(item.dict())).T
    return data

def transform_data(data : pd.DataFrame):
    return data

def get_prediction(data : pd.DataFrame):
    prediction = model.predict(data)
    return {"prediction" : int(prediction)}
