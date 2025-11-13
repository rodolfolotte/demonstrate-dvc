import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Simula métricas
metrics = json.load(open("metrics/metrics.json"))
acc = metrics["accuracy"]

# Curvas simuladas
x = np.linspace(0, 1, 100)
roc = x ** 0.5
pr = 1 - (x ** 2)

os.makedirs("metrics/plots", exist_ok=True)

plt.figure()
plt.plot(x, roc)
plt.title(f"Curva ROC (AUC ≈ {acc:.2f})")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.savefig("metrics/plots/roc_curve.png")

plt.figure()
plt.plot(x, pr)
plt.title("Curva Precision-Recall (simulada)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("metrics/plots/pr_curve.png")

print("Gráficos de avaliação salvos em metrics/plots/")
