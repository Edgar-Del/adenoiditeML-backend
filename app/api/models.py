from pydantic import BaseModel, Field
class PediatricAdenoiditisInput(BaseModel):
    idade: int = Field(..., ge=0, le=5, description="Idade do paciente (0 a 5 anos)")
    genero: str = Field(..., description="Gênero do Paciente (M/F)")
    frequencia_ronco: float = Field(..., ge=0, le=10, description="Frequência do ronco")
    dificuldade_respirar: float = Field(..., ge=0, le=10, description="Dificuldade para respirar")
    obstrucao_nasal: float = Field(..., ge=0, le=100, description="Porcentagem de obstrução nasal")
    apnea_sono: str = Field(..., description="Presença de apneia do sono (Sim/Não)")

class DiagnosisOutput(BaseModel):
    diagnostico: bool
    grupo_gravidade: int
    confianca: float
    recomendacoes: str
