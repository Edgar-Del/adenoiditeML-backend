import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from ml.data.preprocessing import DataPreprocessor
from ml.utils.evaluation import evaluate_model

def train_model(data_path: str, model_path: str) -> None:
    print("Carregando dataset...")
    df = pd.read_csv(data_path)

    print("Iniciando pré-processamento...")
    pre_processador = DataPreprocessor()
    X, y = pre_processador.fit_transform(df)

    print("Separando dados de treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Treinando o modelo Random Forest...")
    modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_rf.fit(X_train, y_train)

    print("Treinando o modelo KNN...")
    modelo_knn = KNeighborsClassifier(n_neighbors=5)
    modelo_knn.fit(X_train, y_train)

    print("Avaliando os modelos...")
    evaluate_model(modelo_rf, X_test, y_test)
    evaluate_model(modelo_knn, X_test, y_test)

    print("Salvando modelos...")
    joblib.dump(modelo_rf, f"{model_path}/modelo_rf.joblib")
    joblib.dump(modelo_knn, f"{model_path}/modelo_knn.joblib")
    joblib.dump(pre_processador, f"{model_path}/pre_processador.joblib")

    print("Modelos salvos com sucesso!")
