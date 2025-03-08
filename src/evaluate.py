import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import os


from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
import mlflow



os.environ['MLFLOW_TRACKING_URI']="https://dagshub.com/Lakehone/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME']="Lakehone"
os.environ['MLFLOW_TRACKING_PASSWORD']="761c242e294f115aac54a7f4acc6b114c878e85a"



## Load all parameters from params.yaml

params = yaml.safe_load(open("params.yaml"))["train"]


def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    mlflow.set_tracking_uri("https://dagshub.com/Lakehone/machinelearningpipeline.mlflow")
    
    ## Load model from disk
    
    model = pickle.load(open(model_path, 'rb'))
    
    prediction = model.predict(X)
    accuracy = accuracy_score(y, prediction)
    
    ## log metrics
    
    mlflow.log_metric("accuracy", accuracy)
    print("Model Accuracy : {accuracy}")
    
    
if __name__=="__name__":
    evaluate(params["data"], params["model"])
    