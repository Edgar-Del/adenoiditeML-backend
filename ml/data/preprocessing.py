import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Pré-processamento de dados de adenoidite para crianças dos 0 aos 5 anos
        """
        # Cria a cópia do data frame
        df_processed = df.copy()
        
        # Validação da idade (0 aos 5 anos)
        df_processed = df_processed[df_processed['idade'] <= 5]
        
        # "Engenharia" de características baseada em parâmetros clínicos. A gravidade do sintoma é igual à média aritmética dos valores sintomas`
        df_processed['gravidade_sintoma'] = (
            df_processed['frequencia_ronco'] + 
            df_processed['dificuldade_respirar'] + 
            df_processed['obstrucao_nasal']
        ) / 3
        
        # Variáveis categóricas
        categorical_cols = ['genero', 'apnea_sono']
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Características clínicas
        clinical_features = [
            'idade',
            'frequencia_ronco',
            'dificuldade_respirar',
            'obstrucao_nasal',
            'apnea_sono',
            'gravidade_sintoma'
        ]
        
        # Dividir caracteristicas (features) e alvo (target).
        X = df_processed[clinical_features]
        y = df_processed['diagnostico_adenoidite']
        

        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        return X_scaled, y