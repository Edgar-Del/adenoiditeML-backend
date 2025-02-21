from joblib import load
from ml.utils.evaluation import evaluate_model
import pandas as pd

# Carregar o modelo treinado
modelo = load("models/saved/modelo_knn.joblib") 

# Carregar os dados de teste
X_test = pd.read_csv("data/processed/X.csv") 
y_test = pd.read_csv("data/processed/y.csv")

# Avaliar o modelo
evaluate_model(modelo, X_test, y_test)
