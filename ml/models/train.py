import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.data.preprocessing import DataPreprocessor
from ml.utils.evaluation import evaluate_model

def train_model(data_path: str, model_path: str) -> None:
    # Carregar os dados
    df = pd.read_csv(data_path)
    
    # Preprocessamento de dados
    pre_processador = DataPreprocessor()
    X, y = pre_processador.fit_transform(df)
    
    # Separar os dados (Dados de treino e de teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Treinar o modelo
    modelo = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    modelo.fit(X_train, y_train)
    
    # Avaliar o modelo
    evaluate_model(modelo, X_test, y_test)
    
    # Guardar o modelo e o pré-processador.
    joblib.dump(modelo, f"{model_path}/modelo.joblib")
    joblib.dump(pre_processador, f"{model_path}/pre_processador.joblib")