import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def evaluate_model(model_path, data_path):
    print("🔍 Avaliando o modelo...")
    
    # Carregar dataset e modelos
    df = pd.read_csv(data_path)
    knn = joblib.load(f"{model_path}/modelo_knn.joblib")
    pca = joblib.load(f"{model_path}/modelo_pca.joblib")
    preprocessor = joblib.load("models/saved/scaler.joblib")
    
    # Processar os dados
    X = df.drop(columns=['diagnostico_adenoidite'])
    y_true = df['diagnostico_adenoidite']
    X_scaled = preprocessor.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # Fazer predições
    y_pred = knn.predict(X_pca)
    
    # Avaliação
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"🎯 Acurácia do modelo: {accuracy:.2f}%")
    print("📊 Relatório de Classificação:\n", classification_report(y_true, y_pred))
    
    # Matriz de Confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.show()


if __name__ == "__main__":
    evaluate_model("models/saved", "data/raw/dataset.csv")
