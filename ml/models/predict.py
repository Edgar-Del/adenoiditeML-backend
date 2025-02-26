import pandas as pd
import joblib
from ml.data.preprocessing import DataPreprocessor

def load_models():
    knn = joblib.load("models/saved/modelo_knn.joblib")
    pca = joblib.load("models/saved/modelo_pca.joblib")
    preprocessor = DataPreprocessor()
    preprocessor.label_encoders['genero'] = joblib.load("models/saved/labelencoder_genero.joblib")
    preprocessor.scaler = joblib.load("models/saved/scaler.joblib")
    return knn, pca, preprocessor

def predict(dados):
    knn, pca, preprocessor = load_models()
    
    df_input = pd.DataFrame([dados])
    
    if 'genero' in df_input.columns:
        df_input['genero'] = preprocessor.label_encoders['genero'].transform([df_input['genero'][0]])[0]
    
    X_transformed = preprocessor.transform(df_input)
    X_pca = pca.transform(X_transformed)
    
    predicao = knn.predict(X_pca)[0]
    return {"diagnostico": bool(predicao)}

if __name__ == "__main__":
    sample_input = {
        "idade_mes": 48,
        "genero": "M",
        "obstrucao_nasal_persistente": 1,
        "secrecao_nasal_purulenta": 0,
        "dor_de_garganta_e_dificuldade_para_engolir": 1,
        "Febre_e_mal_estar_geral": 1,
        "linfodenopatia_cervical": 0,
        "alteracao_na_voz": 1,
        "problemas_auditivos": 0,
        "sintomas_de_infeccao_recorrente": 1,
        "disturbios_de_sono_e_desenvolvimento": 1,
        "tamanho_adenoide": 3
    }
    print(predict(sample_input))
