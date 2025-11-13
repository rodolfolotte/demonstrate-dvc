import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import joblib
import sys, os, json

data_path = sys.argv[1]
model_path = sys.argv[2]

df = pd.read_csv(data_path)
X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("demo-mlops")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    metrics = {"accuracy": acc}
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Modelo salvo em {model_path}, accuracy: {acc:.3f}")
