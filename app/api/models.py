from pydantic import BaseModel, Field

class PediatricAdenoiditisInput(BaseModel):
    idade: int = Field(..., ge=0, le=5, description="idade no paciente em anos")
    genero: str = Field(..., description="Género do Paciente (M/F)")
    frequencia_ronco: float = Field(..., ge=0, le=10, description="Pontuação da frequência do ronco")
    breathing_difficulty: float = Field(..., ge=0, le=10, description="Pontuação da dificuldade respiratória")
    obstrucao_nasal: float = Field(..., ge=0, le=100, description="Percentual de obstrução nasal.")
    apnea_sono: str = Field(..., description="Presença de apneia do sono (Sim/Não).")
    
    class Config:
        json_schema_extra = {
            "exemplo": {
                "idade": 4,
                "genero": "M",
                "frequencia_ronco": 7.5,
                "dificuldade_respirar": 6.0,
                "obstrucao_nasal": 75.0,
                "apnea_sono": "Sim"
            }
        }

class DiagnosisOutput(BaseModel):
    diagnosistico: bool     # Indica se há um diagnóstico positivo (True/False)
    grupo_gravidade: int    # Cluster de gravidade do sintoma (ex: 0, 1, 2)
    confianca: float        # Nível de confiança da predição (ex: 0.85 para 85%)
    recomendacoes: str      # Recomendações com base no diagnóstico