# Demonstração de DVC + CI/CD + MLFlow


## DVC
### Versionamento de dados e pipeline
Inicialização:
```
dvc init
git add .dvc .gitignore
```

Adicionar dados ao controle do DVC:
```
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc
```

Isso cria um arquivo .dvc que referencia o dataset. Você pode enviar o dataset para um remote (Google Drive, S3, etc.):
```
dvc remote add -d myremote gdrive://<folder-id>
dvc push
```

### Pipeline DVC
Arquivo `pipelines/dvc.yaml`:
```
stages:
  preprocess:
    cmd: python src/preprocess.py data/raw/dataset.csv data/processed/clean.csv
    deps:
      - src/preprocess.py
      - data/raw/dataset.csv
    outs:
      - data/processed/clean.csv

  train:
    cmd: python src/train.py data/processed/clean.csv models/model.pkl
    deps:
      - src/train.py
      - data/processed/clean.csv
    outs:
      - models/model.pkl
    metrics:
      - metrics/metrics.json
      - metrics/plots/roc_curve.png

  evaluate:
    cmd: python src/evaluate.py models/model.pkl data/processed/clean.csv
    deps:
      - src/evaluate.py
      - models/model.pkl
```

Executar pipeline:
```
dvc repro
```

## Treino Automático no GitHub Actions
Arquivo `.github/workflows/github-ci.yml`:
```
name: CI/CD - MLOps Demo

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-train:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install dvc[gs] mlflow

      - name: Reproduce DVC pipeline
        run: dvc repro

      - name: Push updated metrics
        run: |
          dvc push
          git add metrics/ dvc.lock
          git commit -m "Update metrics after pipeline run" || echo "No changes"
          git push
```
Isso irá garantir que toda vez que você fizer `push`, o GitHub:
- Instala o ambiente
- Executa o pipeline DVC
- Atualiza métricas automaticamente


## MLFlow - Rastreamento de Experimentos
No script `train.py`, adicione o tracking, e.g.: 
```
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("demo-mlops")

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 50)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

```

Uma vez adicionado, execute: 
```
mlflow ui
```

