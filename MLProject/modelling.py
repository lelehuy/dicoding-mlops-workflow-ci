import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

def train_model():
    # Setup MLflow tracking URI if present in environment
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Tracking MLflow to DagsHub/Remote server: {tracking_uri}")
    else:
        print("Using local MLflow 'mlruns' directory for tracking.")
    
    # Set/create experiment
    mlflow.set_experiment("Breast_Cancer_Classification")
    
    # Enable Autologging before run starts (Kriteria basic)
    mlflow.sklearn.autolog()
    
    # Load dataset from the preprocessed CSV inside Membangun_model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "namadataset_preprocessing", "breast_cancer_preprocessed.csv")
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        X = df.drop(columns=['target'])
        y = df['target']
        print(f"Loaded dataset from {csv_path}")
    else:
        # Fallback to load from sklearn directly if file is missing
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
        y = pd.Series(cancer.target)
        print("Loaded dataset from sklearn fallback")
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Start MLflow run
    with mlflow.start_run(run_name="Baseline_Random_Forest") as run:
        print("Training baseline Random Forest model with Autolog...")
        
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print("\n--- Model Evaluation ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-Score : {f1:.4f}")
        

        # Save model locally inside the same directory as this script
        model_local_path = os.path.join(current_dir, "model.joblib")
        joblib.dump(model, model_local_path)
        print(f"Model saved locally to {model_local_path}")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    train_model()
