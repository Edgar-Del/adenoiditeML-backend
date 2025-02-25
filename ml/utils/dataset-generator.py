import pandas as pd
import numpy as np

def gerar_dataset(num_samples=150):
    """
    Gerar um dataset sintético para o diagnóstico de Adenoidite.

    Parâmetros:
    - `num_samples`: Número de registros sintéticos de pacientes a serem gerados.

    Retorno:
    - Um DataFrame do pandas com dados sintéticos de pacientes.
    """
    np.random.seed(42)  # Garantir reprodutibilidade
    
    data = {
        'id_paciente': range(1, num_samples + 1),
        'idade_mes': np.random.randint(6, 145, num_samples),
        'genero': np.random.choice(['M', 'F'], num_samples),
        'obstrucao_nasal_persistente': np.random.binomial(1, 0.7, num_samples),
        'secrecao_nasal_purulenta': np.random.binomial(1, 0.4, num_samples),
        'dor_de_garganta_e_dificuldade_para_engolir': np.random.binomial(1, 0.6, num_samples),
        'Febre_e_mal_estar_geral': np.random.binomial(1, 0.5, num_samples),
        'linfodenopatia_cervical': np.random.binomial(1, 0.3, num_samples),
        'alteracao_na_voz': np.random.binomial(1, 0.4, num_samples),
        'problemas_auditivos': np.random.binomial(1, 0.25, num_samples),
        'sintomas_de_infeccao_recorrente': np.random.binomial(1, 0.5, num_samples),
        'disturbios_de_sono_e_desenvolvimento': np.random.binomial(1, 0.6, num_samples),
        'tamanho_adenoide': np.random.choice([1, 2, 3, 4], num_samples, p=[0.1, 0.3, 0.4, 0.2])
    }

    # Diagnóstico baseado em critérios clínicos
    prob_diagnostico = (
        0.4 * (data['tamanho_adenoide'] / 4) +
        0.3 * data['obstrucao_nasal_persistente'] +
        0.2 * data['secrecao_nasal_purulenta'] +
        0.1 * data['Febre_e_mal_estar_geral']
    )

    # Limiar para balanceamento (~50% casos positivos e negativos)
    limiar = np.percentile(prob_diagnostico, 50)
    data['diagnostico_adenoidite'] = (prob_diagnostico > limiar).astype(int)

    # Criar DataFrame
    df = pd.DataFrame(data)

    # Salvar o dataset gerado
    df.to_csv('data/raw/dataset.csv', index=False)

    print(df.head())
    print("\nEstatísticas do Dataset:")
    print(df.describe())
    print("\nDistribuição de Diagnóstico:")
    print(df['diagnostico_adenoidite'].value_counts(normalize=True))

    return df

# Gerar e salvar o dataset
gerar_dataset(150)
