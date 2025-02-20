import joblib
import pandas as pd

def load_model():
    try:
        model = joblib.load("models/saved/modelo.joblib")
        preprocessor = joblib.load("models/saved/pre_processador.joblib")
        return model, preprocessor
    except FileNotFoundError:
        raise RuntimeError("Erro: O modelo não foi encontrado! Treine o modelo primeiro.")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar modelo: {e}")

def predict_adenoiditis(input_data: dict):
    model, preprocessor = load_model()
    df_input = pd.DataFrame([input_data])
    X = preprocessor.transform(df_input)
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][1]
    return prediction, confidence
