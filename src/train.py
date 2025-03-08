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

def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv = 5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search


## Load all parameters from params.yaml

params = yaml.safe_load(open("params.yaml"))["train"]

def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]
    
    mlflow.set_tracking_uri("https://dagshub.com/Lakehone/machinelearningpipeline.mlflow")
    
    
    ## start the mlflow eun
    
    with mlflow.start_run():
        # split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        signature = infer_signature(X_train, y_train)
        
        ## Define hyperparameter grid
        
        param_grid = {
            
            'n_estimators': [300, 400, 500],
            'max_depth': [3, 4, 6],
            'min_samples_split' : [3, 5, 6],
            'min_samples_leaf' : [2, 3, 4]
        }
        
        
        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        
        ## Get the best model
        
        best_model = grid_search.best_estimator_
        
        
        ## Predict and evaluate model
        
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) 
        print(f"Accuracy : {accuracy}")
        
        
        ## Log additional metrics
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_samples_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_['min_samples_leaf'])
        
        ## Log the confusion matrix and classification report
        
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        mlflow.log_text(cr, "classification_report.txt")
        
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(best_model, "model", registered_model_name = "Best Model")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
            
        ## directory to save model
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        filename = model_path
        pickle.dump(best_model, open(filename, 'wb'))
        
        print(f"Model saved to {model_path}")
        
        
if __name__=="__main__":
    train(params['data'], params['model'], params['random_state'], params['n_estimators'], params['max_depth'])
            
        