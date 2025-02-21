import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df_processed = df.copy()

        # Criar nova feature de gravidade dos sintomas
        df_processed['gravidade_sintoma'] = (
            df_processed['frequencia_ronco'] + 
            df_processed['dificuldade_respirar'] + 
            df_processed['obstrucao_nasal']
        ) / 3

        # Processar variáveis categóricas
        categorical_cols = ['genero', 'apnea_sono']
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le

        # Selecionar características (ordem correta)
        clinical_features = [
            'idade', 'genero', 'frequencia_ronco', 'dificuldade_respirar',
            'obstrucao_nasal', 'apnea_sono', 'gravidade_sintoma'
        ]

        X = df_processed[clinical_features]
        y = df_processed['diagnostico'] if 'diagnostico' in df_processed.columns else None

        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=clinical_features)

        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica o mesmo pré-processamento em novos dados."""
        df_processed = df.copy()

        print("🔹 Antes do pré-processamento:\n", df_processed)

        # Criar nova feature de gravidade dos sintomas
        df_processed['gravidade_sintoma'] = (
            df_processed['frequencia_ronco'] + 
            df_processed['dificuldade_respirar'] + 
            df_processed['obstrucao_nasal']
        ) / 3

        # Processar variáveis categóricas
        categorical_cols = ['genero', 'apnea_sono']
        for col in categorical_cols:
            if col in self.label_encoders:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
            else:
                raise ValueError(f"❌ ERRO: LabelEncoder para {col} não encontrado!")

        print("🔹 Depois do pré-processamento:\n", df_processed)

        # Selecionar características (ordem correta)
        clinical_features = [
            'idade', 'genero', 'frequencia_ronco', 'dificuldade_respirar',
            'obstrucao_nasal', 'apnea_sono', 'gravidade_sintoma'
        ]

        # Garantir que as colunas existam no DataFrame
        for col in clinical_features:
            if col not in df_processed:
                raise ValueError(f"❌ ERRO: Coluna {col} está ausente após o pré-processamento!")

        X = df_processed[clinical_features]
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=clinical_features)

        print("🔹 Depois da transformação final:\n", X_scaled)

        return X_scaled
