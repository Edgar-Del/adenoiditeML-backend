import pandas as pd
import joblib
import numpy as np
from typing import Tuple, Dict

def load_model(model_path: str):
    modelo = joblib.load(f"{model_path}/modelo_rf.joblib")
    pre_processador = joblib.load(f"{model_path}/pre_processador.joblib")
    return modelo, pre_processador

def predict_adenoiditis(input_data: Dict) -> Tuple[bool, int, float]:
    """ Faz a predição para um paciente """
    modelo, pre_processador = load_model("models/saved")

    # Converter o input para um DataFrame antes do pré-processamento
    df_input = pd.DataFrame([input_data])
    print("🔹 Dados recebidos para predição:\n", df_input)

    # Aplicar o pré-processamento correto
    try:
        X = pre_processador.transform(df_input)
        print("🔹 Dados pós-transformação:\n", X)
    except Exception as e:
        print("❌ ERRO no pré-processamento:", e)
        raise e

    # Verificar a compatibilidade das features
    if X.shape[1] != modelo.n_features_in_:
        raise ValueError(f"❌ ERRO: Número de features inconsistente! Esperado: {modelo.n_features_in_}, Recebido: {X.shape[1]}")

    # Fazer a predição
    try:
        prediction = modelo.predict(X)[0]
        probability = modelo.predict_proba(X)[0][1] if hasattr(modelo, "predict_proba") else 0.5
    except Exception as e:
        print("❌ ERRO ao fazer predição:", e)
        raise e

    # Definir o nível de gravidade (baseado na probabilidade)
    cluster = 2 if probability > 0.75 else (1 if probability > 0.5 else 0)

    return bool(prediction), cluster, float(probability)
