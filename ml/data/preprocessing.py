import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def fit_transform(self, df):
        if 'diagnostico_adenoidite' not in df.columns:
            raise ValueError("A coluna 'diagnostico_adenoidite' não foi encontrada no dataset de treinamento!")

        y = df['diagnostico_adenoidite']
        X = df.drop(columns=['diagnostico_adenoidite'])

        if 'genero' in X.columns:
            self.label_encoders['genero'] = LabelEncoder()
            X['genero'] = self.label_encoders['genero'].fit_transform(X['genero'])

        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)

        joblib.dump(self.label_encoders, "models/saved/label_encoders.joblib")
        joblib.dump(self.scaler, "models/saved/scaler.joblib")

        print(f"✅ Treinamento: X shape {X_scaled.shape}, y shape {y.shape}")
        return X_scaled, y

    def transform(self, df):
        self.label_encoders = joblib.load("models/saved/label_encoders.joblib")
        self.scaler = joblib.load("models/saved/scaler.joblib")

        if 'genero' in df.columns:
            df['genero'] = self.label_encoders['genero'].transform([df['genero'][0]])[0]

        df_scaled = pd.DataFrame(self.scaler.transform(df), columns=df.columns)

        print(f"✅ Predição: X shape {df_scaled.shape}")
        return df_scaled
