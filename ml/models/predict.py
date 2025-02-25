import joblib
import pandas as pd
from ml.data.preprocessing import DataPreprocessor

def predict(dados: dict):
    try:
        # Carregar modelos treinados
        modelo_knn = joblib.load("models/saved/modelo_knn.joblib")
        modelo_pca = joblib.load("models/saved/modelo_pca.joblib")
        pre_processor = DataPreprocessor()

        # Transformar entrada
        df_input = pd.DataFrame([dados])
        X_transformed = pre_processor.transform(df_input)
        X_pca = modelo_pca.transform(X_transformed)

        # Fazer predição
        predicao = modelo_knn.predict(X_pca)[0]
        return {"diagnostico": bool(predicao)}
    except Exception as e:
        return {"error": str(e)}
