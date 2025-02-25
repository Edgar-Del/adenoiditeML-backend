from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
from ml.data.preprocessing import DataPreprocessor
from ml.models.train import train_model

router = APIRouter()

# Carregar modelos treinados
modelo_knn = joblib.load("models/saved/modelo_knn.joblib")
modelo_pca = joblib.load("models/saved/modelo_pca.joblib")
pre_processor = DataPreprocessor()

@router.post("/diagnosticar")
def diagnosticar(dados: dict):
    try:
        df_input = pd.DataFrame([dados])
        X_transformed = pre_processor.transform(df_input)
        X_pca = modelo_pca.transform(X_transformed)
        predicao = modelo_knn.predict(X_pca)[0]
        return {"diagnostico": bool(predicao)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/treinar")
def treinar_modelo():
    try:
        train_model("data/processed/X_clustered.csv", "models/saved")
        return {"message": "Modelos treinados com sucesso!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
