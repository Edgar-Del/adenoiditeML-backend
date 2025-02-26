import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

np.random.seed(42)  # Garantir reprodutibilidade
total_registros = 150

# Criando os dados sintéticos
data = {
    'idade_mes': np.random.randint(6, 145, total_registros),
    'genero': np.random.choice(['M', 'F'], total_registros),
    'obstrucao_nasal_persistente': np.random.binomial(1, 0.7, total_registros),
    'secrecao_nasal_purulenta': np.random.binomial(1, 0.4, total_registros),
    'dor_de_garganta_e_dificuldade_para_engolir': np.random.binomial(1, 0.6, total_registros),
    'Febre_e_mal_estar_geral': np.random.binomial(1, 0.5, total_registros),
    'linfodenopatia_cervical': np.random.binomial(1, 0.3, total_registros),
    'alteracao_na_voz': np.random.binomial(1, 0.4, total_registros),
    'problemas_auditivos': np.random.binomial(1, 0.25, total_registros),
    'sintomas_de_infeccao_recorrente': np.random.binomial(1, 0.5, total_registros),
    'disturbios_de_sono_e_desenvolvimento': np.random.binomial(1, 0.6, total_registros),
    'tamanho_adenoide': np.random.choice([1, 2, 3, 4], total_registros, p=[0.1, 0.3, 0.4, 0.2])
}

# Criar diagnóstico com base em critérios clínicos
prob_diagnostico = (
    0.4 * (data['tamanho_adenoide'] / 4) + 
    0.3 * data['obstrucao_nasal_persistente'] +
    0.2 * data['secrecao_nasal_purulenta'] +
    0.1 * data['Febre_e_mal_estar_geral']
)

# Definir limiar para diagnóstico (~50% positivo e 50% negativo)
limiar = np.percentile(prob_diagnostico, 50)
data['diagnostico_adenoidite'] = (prob_diagnostico > limiar).astype(int)

# Criando DataFrame
df = pd.DataFrame(data)

# Aplicar LabelEncoder ao 'genero'
le_genero = LabelEncoder()
df['genero'] = le_genero.fit_transform(df['genero'])

# Salvar LabelEncoder para uso posterior
joblib.dump(le_genero, "models/saved/labelencoder_genero.joblib")

# Salvar dataset gerado
df.to_csv("data/raw/dataset.csv", index=False)

print("✅ Dataset gerado com sucesso! Shape:", df.shape)
print(df.head())