import pandas as pd
import numpy as np
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("train_model")

train = pd.read_csv('/home/petr/project/datasets/data_train.csv')

y_train = train['y'].astype('int').values
X_train = train.drop('y', axis=1).values

model = ShapeletTransformClassifier(estimator=RandomForestClassifier(n_estimators=100),
                                  n_shapelet_samples=100,
                                  max_shapelets=100,
                                  batch_size=20)

with mlflow.start_run():
    mlflow.sklearn.log_model(model,
                             artifact_path="stfc",
                             registered_model_name="stfc")
    mlflow.log_artifact(local_path="/home/petr/project/scripts/train_model.py",
                        artifact_path="train_model code")
    mlflow.end_run()

model.fit(X_train, y_train)

with open('/home/petr/project/models/data.pickle', 'wb') as f:
    pickle.dump(model, f)
