import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def fit_transform(self, df):
        # Certificar-se de que 'diagnostico_adenoidite' está presente (somente no treinamento)
        if 'diagnostico_adenoidite' not in df.columns:
            raise ValueError("A coluna 'diagnostico_adenoidite' não foi encontrada no dataset de treinamento!")

        # Separar target (y) e features (X)
        y = df['diagnostico_adenoidite']
        X = df.drop(columns=['diagnostico_adenoidite'])

        # Converter 'genero' para numérico usando LabelEncoder
        if 'genero' in X.columns:
            self.label_encoders['genero'] = LabelEncoder()
            X['genero'] = self.label_encoders['genero'].fit_transform(X['genero'])

        # Normalizar os dados
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        # Salvar LabelEncoder e Scaler para uso posterior na API
        joblib.dump(self.label_encoders['genero'], "models/saved/labelencoder_genero.joblib")
        joblib.dump(self.scaler, "models/saved/scaler.joblib")

        print(f"✅ Treinamento: X shape {X_scaled.shape}, y shape {y.shape}")
        return X_scaled, y  # Retorna dois valores corretamente

    def transform(self, df):
        # Garantir que o LabelEncoder foi carregado corretamente
        if 'genero' in df.columns:
            if 'genero' not in self.label_encoders:
                self.label_encoders['genero'] = joblib.load("models/saved/labelencoder_genero.joblib")
            df['genero'] = self.label_encoders['genero'].transform([df['genero'][0]])[0]

        # Aplicar normalização
        self.scaler = joblib.load("models/saved/scaler.joblib")
        df_scaled = pd.DataFrame(self.scaler.transform(df), columns=df.columns)

        print(f"✅ Predição: X shape {df_scaled.shape}")
        return df_scaled
