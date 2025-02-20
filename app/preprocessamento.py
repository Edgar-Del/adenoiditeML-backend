from ml.data.preprocessing import DataPreprocessor
import pandas as pd

# Carregar os dados
df = pd.read_csv("data/raw/dataset.csv")

# Pré-processar os dados
preprocessor = DataPreprocessor()
X, y = preprocessor.fit_transform(df)

# Salvar os dados processados
X.to_csv("data/processed/X.csv", index=False)
y.to_csv("data/processed/y.csv", index=False)
