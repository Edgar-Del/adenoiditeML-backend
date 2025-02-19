from fastapi import APIRouter, HTTPException
from app.api.models import PediatricAdenoiditisInput, DiagnosisOutput
from ml.models.predict import predict_adenoiditis
from typing import List

router = APIRouter()

@router.post("/diagnosticar", response_model=DiagnosisOutput)
async def diagnose(input_data: PediatricAdenoiditisInput):
    try:
        diagnosistico, cluster, confianca = predict_adenoiditis(input_data.dict())
        
        # Gerar recomendações com base na gravidade.
        recommendations = get_recommendations(cluster, confianca)
        
        return DiagnosisOutput(
            diagnosistico=diagnosistico,
            grupo_gravidade=cluster,
            confianca=confianca,
            recomendacoes=recommendations
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_recommendations(cluster: int, confianca: float) -> str:
    severity_levels = {
        0: "Leve",
        1: "Moderado",
        2: "Grave"
    }
    
    base_recommendations = {
        0: "Monitorar os sintomas e realizar acompanhamento em 3 meses",
        1: "Agendar acompanhamento em 1 mês e considerar terapia médica",
        2: "Consulta imediata com especialista recomendada"
    }
    
    return f"Nível de Gravidade: {severity_levels[cluster]}\n{base_recommendations[cluster]}"