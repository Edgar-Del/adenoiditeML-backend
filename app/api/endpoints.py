from fastapi import APIRouter, HTTPException
from app.api.models import PediatricAdenoiditisInput, DiagnosisOutput
from ml.models.predict import predict_adenoiditis
from ml.models.train import train_model
from ml.utils.evaluation import evaluate_model
import joblib
import pandas as pd

router = APIRouter()

@router.post("/diagnosticar", response_model=DiagnosisOutput)
async def diagnose(input_data: PediatricAdenoiditisInput):
    try:
        print("🔹 Entrada recebida:", input_data.dict())  #depuração
        diagnostico, cluster, confianca = predict_adenoiditis(input_data.dict())
        
        recommendations = get_recommendations(cluster, confianca)
        return DiagnosisOutput(
            diagnostico=diagnostico,
            grupo_gravidade=cluster,
            confianca=confianca,
            recomendacoes=recommendations
        )
    except Exception as e:
        print("❌ ERRO:", e)  #depuração básica
        raise HTTPException(status_code=500, detail=str(e))


def get_recommendations(cluster: int, confianca: float) -> str:
    severity_levels = {0: "Leve", 1: "Moderado", 2: "Grave"}
    base_recommendations = {
        0: "Monitorar os sintomas e realizar acompanhamento em 3 meses",
        1: "Agendar acompanhamento em 1 mês e considerar terapia médica",
        2: "Consulta imediata com especialista recomendada"
    }
    return f"Nível de Gravidade: {severity_levels.get(cluster, 'Desconhecido')}\n{base_recommendations.get(cluster, 'Sem recomendações')}"

@router.post("/treinar")
def train():
    try:
        train_model("data/raw/dataset.csv", "models/saved")
        return {"message": "Modelo treinado e salvo com sucesso!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {e}")

@router.get("/avaliar")
def evaluate():
    try:
        modelo = joblib.load("models/saved/modelo.joblib")
        X_test = pd.read_csv("data/processed/X_test.csv")
        y_test = pd.read_csv("data/processed/y_test.csv")
        resultado = evaluate_model(modelo, X_test, y_test)
        return {"message": "Avaliação realizada!", "result": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na avaliação: {e}")
