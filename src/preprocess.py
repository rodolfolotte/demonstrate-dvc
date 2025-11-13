import sys
import pandas as pd

from sklearn.datasets import make_classification

raw_path = sys.argv[1]
out_path = sys.argv[2]

X, y = make_classification(n_samples=300, n_features=5, random_state=42)
df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
df["target"] = y

df.to_csv(out_path, index=False)
print(f"Arquivo processado salvo em {out_path}")
