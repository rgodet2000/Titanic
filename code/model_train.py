# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 14:45:24 2022

@author: Pierre
"""

import pandas as pd
from model_build import build_pipeline
import joblib


if __name__ == '__main__':
    df = pd.read_csv("train.csv")
    
    y = df['Survived']
    X = df.drop('Survived', axis=1)
    
    model = build_pipeline()
    model.fit(X, y)
    
    joblib.dump(model, "model.joblib")


