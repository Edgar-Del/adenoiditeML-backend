from pydantic import BaseModel

class DiagnosticoInput(BaseModel):
    idade_mes: int
    genero: str
    obstrucao_nasal_persistente: int
    secrecao_nasal_purulenta: int
    dor_de_garganta_e_dificuldade_para_engolir: int
    Febre_e_mal_estar_geral: int
    linfodenopatia_cervical: int
    alteracao_na_voz: int
    problemas_auditivos: int
    sintomas_de_infeccao_recorrente: int
    disturbios_de_sono_e_desenvolvimento: int
    tamanho_adenoide: int

class DiagnosticoOutput(BaseModel):
    diagnostico: bool
