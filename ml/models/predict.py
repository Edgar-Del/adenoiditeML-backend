import joblib
import pandas as pd
from typing import Tuple, Dict

def load_model(model_path: str):
    modelo = joblib.load(f"{model_path}/model.joblib")
    pre_processador = joblib.load(f"{model_path}/preprocessor.joblib")
    return modelo, pre_processador

def predict_adenoiditis(input_data: Dict) -> Tuple[int, float]:
    # Carregar o modelo e pre-processador
    modelo, pre_processador = load_model("models/saved")
    
    # Pré-processar entrada
    X = pre_processador.transform(input_data)
    
    # Prever
    prediction = modelo.predict(X)[0]
    probability =  modelo.predict_proba(X)[0][1]
    
    return prediction, probability