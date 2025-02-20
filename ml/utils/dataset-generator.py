import pandas as pd
import numpy as np
import random

def gerar_dataset(num_samples=100):
    """
Gerar um dataset sintético para o diagnóstico de adenoidite 

Parâmetros:  
- `num_samples`: Número de registros sintéticos de pacientes a serem gerados  

Retorno: 
- Um DataFrame do pandas com dados sintéticos de pacientes
    """
    # reprodutividade aleatória
    np.random.seed(42)
    
    # gerando os dados
    data = {
        'idade': np.random.randint(0, 6, num_samples),
        'genero': np.random.choice(['M', 'F'], num_samples),
        'frequencia_ronco': np.random.uniform(0, 10, num_samples).round(1),
        'dificuldade_respirar': np.random.uniform(0, 10, num_samples).round(1),
        'obstrucao_nasal': np.random.uniform(0, 100, num_samples).round(1),
        'apnea_sono': np.random.choice(['Sim', 'Não'], num_samples, p=[0.4, 0.6])
    }
    
    # criar o dataframe
    df = pd.DataFrame(data)
    
    # criar uma lógica básica para gerar diagnóstico (impírica)
    df['diagnostico'] = (
        (df['frequencia_ronco'] > 5) & 
        (df['dificuldade_respirar'] > 4) & 
        (df['obstrucao_nasal'] > 50) & 
        (df['apnea_sono'] == 'Sim')
    )
    
    # Gerar agrupamentos de acordo com a gravidade
    df['grupo_gravidade'] = np.where(
        df['diagnostico'],
        np.random.choice([0, 1, 2], df.shape[0], p=[0.3, 0.5, 0.2]),
        0
    )
    
    # Grau de confiança
    df['confianca'] = np.where(
        df['diagnostico'],
        np.random.uniform(0.6, 1.0, df.shape[0]).round(2),
        np.random.uniform(0.1, 0.5, df.shape[0]).round(2)
    )
    
    # Gerar recomendações
    df['recomendacoes'] = df.apply(
        lambda row: 'Encaminhar para avaliação médica especializada' 
        if row['diagnostico'] 
        else 'Monitorar sintomas', 
        axis=1
    )
    
    return df

# Gerar o dataset
dataset_adenoidite = gerar_dataset(100)

# Save to CSV
dataset_adenoidite.to_csv('data/raw/dataset.csv', index=False)

# Mostrar algumas estatísticas básicas do dataset
print(dataset_adenoidite.head())
print("\nEstatísticas do Dataset:")
print(dataset_adenoidite.describe())
print("\nDistribuição de Diagnóstico:")
print(dataset_adenoidite['diagnosistico'].value_counts(normalize=True))
