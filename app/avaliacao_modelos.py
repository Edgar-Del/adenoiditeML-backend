from joblib import load
from ml.utils.evaluation import evaluate_model
import pandas as pd

# Carregar o modelo treinado
modelo_rf = load("models/saved/modelo_rf.joblib")
modelo_knn = load("models/saved/modelo_knn.joblib")

# Carregar os dados de teste
X_test = pd.read_csv("data/processed/X.csv")  # Supondo que seja o conjunto de teste
y_test = pd.read_csv("data/processed/y.csv")

# Avaliar os modelos
evaluate_model(modelo_rf, X_test, y_test, "results/rf_confusion_matrix.png")
evaluate_model(modelo_knn, X_test, y_test, "results/knn_confusion_matrix.png")
