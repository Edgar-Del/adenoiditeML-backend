from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
from ml.data.preprocessing import DataPreprocessor

router = APIRouter()

# Carregar modelos treinados e pré-processador
try:
    modelo_knn = joblib.load("models/saved/modelo_knn.joblib")
    modelo_pca = joblib.load("models/saved/modelo_pca.joblib")
    pre_processor = DataPreprocessor()

    # Carregar LabelEncoder treinado
    try:
        pre_processor.label_encoders['genero'] = joblib.load("models/saved/labelencoder_genero.joblib")
        print("✅ LabelEncoder carregado para 'genero'.")
    except FileNotFoundError:
        print("❌ ERRO: LabelEncoder não encontrado! Rode o pré-processamento novamente.")
        raise HTTPException(status_code=500, detail="LabelEncoder não encontrado. Rode o pré-processamento primeiro.")

except Exception as e:
    print(f"❌ Erro ao carregar os modelos: {e}")
    raise HTTPException(status_code=500, detail="Erro ao carregar os modelos.")

@router.post("/diagnosticar")
def diagnosticar(dados: dict):
    try:
        # Validar entrada
        required_fields = ["idade_mes", "genero", "obstrucao_nasal_persistente", "secrecao_nasal_purulenta",
                           "dor_de_garganta_e_dificuldade_para_engolir", "Febre_e_mal_estar_geral", "linfodenopatia_cervical",
                           "alteracao_na_voz", "problemas_auditivos", "sintomas_de_infeccao_recorrente",
                           "disturbios_de_sono_e_desenvolvimento", "tamanho_adenoide"]
        missing_fields = [field for field in required_fields if field not in dados]
        if missing_fields:
            raise HTTPException(status_code=400, detail=f"Campos ausentes: {missing_fields}")

        df_input = pd.DataFrame([dados])

        # Remover 'diagnostico_adenoidite' se estiver presente (só deve existir no treinamento)
        if 'diagnostico_adenoidite' in df_input.columns:
            df_input = df_input.drop(columns=['diagnostico_adenoidite'])

        # Garantir que o LabelEncoder está carregado e funcionando corretamente
        if 'genero' in df_input.columns:
            df_input['genero'] = pre_processor.label_encoders['genero'].transform([df_input['genero'][0]])[0]

        # Pré-processamento dos dados
        X_transformed = pre_processor.transform(df_input)
        X_pca = modelo_pca.transform(X_transformed)

        # Fazer predição
        predicao = modelo_knn.predict(X_pca)[0]
        return {"diagnostico": bool(predicao)}

    except HTTPException as http_err:
        raise http_err  # Manter mensagens de erro claras
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Erro: Campo ausente no JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
