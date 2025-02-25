import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def fit_transform(self, df: pd.DataFrame):
        df_processed = df.copy()

        # Criar feature de gravidade dos sintomas
        df_processed['gravidade_sintoma'] = (
            df_processed['obstrucao_nasal_persistente'] +
            df_processed['secrecao_nasal_purulenta'] +
            df_processed['Febre_e_mal_estar_geral']
        ) / 3

        # Processar variáveis categóricas
        categorical_cols = ['genero']
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le

        # Selecionar features para modelo
        features = ['idade_mes', 'genero', 'gravidade_sintoma', 'tamanho_adenoide']
        X = df_processed[features]
        y = df_processed['diagnostico_adenoidite']

        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        return X_scaled, y

    def transform(self, df: pd.DataFrame):
        df_processed = df.copy()
        df_processed['gravidade_sintoma'] = (
            df_processed['obstrucao_nasal_persistente'] +
            df_processed['secrecao_nasal_purulenta'] +
            df_processed['Febre_e_mal_estar_geral']
        ) / 3
        
        categorical_cols = ['genero']
        for col in categorical_cols:
            if col in self.label_encoders:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
            else:
                raise ValueError(f"LabelEncoder para {col} não encontrado!")
        
        features = ['idade_mes', 'genero', 'gravidade_sintoma', 'tamanho_adenoide']
        X = df_processed[features]
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        return X_scaled
