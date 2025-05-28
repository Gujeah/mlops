
import pandas as pd
from sklearn.metrics import mean_squared_error
import math
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sys
import os
import mlflow
from mlflow import tracking
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope



tracking.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("pipeline experiment")




url1="https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-01.parquet"
url2="https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2025-02.parquet"



def wrangle(year, month):
    url=f"https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet"
    df=pd.read_parquet(url)
    df["lpep_pickup_datetime"]=pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"]=pd.to_datetime(df["lpep_dropoff_datetime"])
    df["duration"]=df["lpep_dropoff_datetime"]-df["lpep_pickup_datetime"]
    df.duration=df.duration.apply(lambda td:td.total_seconds()/60)
    df=df[((df.duration>=1) & (df.duration<=60))]
    categorical=["PULocationID","DOLocationID"]
    #numerical=["trip_distance"]
    df[categorical]= df[categorical].astype(str)
    df['PU_DO']=df["PULocationID"]+ "_" +df["DOLocationID"]

    return df




df_train=wrangle(year=2025, month=1)
df_val=wrangle(year=2025, month=2)

def create_X(df, dv=None):

    numerical=["trip_distance"]
    categorical=["PU_DO"]

    dicts=df[categorical + numerical].to_dict(orient="records")
    if dv is None:

        dv=DictVectorizer(sparse=True)
        X=dv.fit_transform(dicts)
    else:

        X=dv.transform(dicts)
    return X, dv


X_train, dv=create_X(df_train)
X_val, _=create_X(df_val,dv)



target="duration"
y_train=df_train[target].values

y_val=df_val[target].values




mlflow.xgboost.autolog(disable=True)

def training_model(X_train, y_train, X_val, y_val,dv):


    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        valid = xgb.DMatrix(X_val, label=y_val)
        best_params={
            "reg_alpha": 0.015143662957798473,
            "seed": 42,
            "learning_rate": 0.2635721416413519,
            "objective": "reg:linear",
            "min_child_weight": 11.354716447890874,
            "max_depth": 6,
            "reg_lambda": 0.006716153330161491
        }
        mlflow.log_params(best_params)
        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid) 
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        new_models_dir = "/workspaces/mlops/ml_pipelines/model"
        os.makedirs(new_models_dir, exist_ok=True)
        preprocessor_full_path = os.path.join(new_models_dir, "preprocessor.b")
        with open(preprocessor_full_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(preprocessor_full_path, artifact_path="preprocessor")
        mlflow.log_metric("rmse", rmse)
        mlflow.xgboost.log_model(booster, "models_mlflow")
        print(f"Preprocessor saved to and logged from: {preprocessor_full_path}")
        print("Other metrics and model logged successfully.")

def run(year, month):
    df_train=wrangle(year=year, month=month)
    next_year= year if month <12 else year+1
    next_month=month+1 if month <12 else 1
    df_val=wrangle(next_year, next_month)
    X_train, dv=create_X(df_train)
    X_val, _=create_X(df_val,dv)
    training_model(X_train, y_train, X_val, y_val,dv)

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="training model to predict duration of taxi driver")
    parser.add_argument('--year', type=int, required=True, help='year of the dataset')
    parser.add_argument('--month', type=int, required=True, help='month of the dataset')
    args=parser.parse_args()
    run(year=args.year, month=args.month)
    run(year=2025, month=1)







