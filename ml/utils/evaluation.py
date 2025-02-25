import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def evaluate_model(model_path: str, X_test, y_test, output_path: str):
    # Carregar o modelo treinado
    modelo = joblib.load(model_path)
    
    # Fazer previsões
    y_pred = modelo.predict(X_test)
    
    # Gerar matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negativo', 'Positivo'], yticklabels=['Negativo', 'Positivo'])
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.savefig(output_path)
    plt.close()
    
    # Imprimir relatório de classificação
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Relatório de Classificação:\n", report)
    print(f"Acurácia do modelo: {accuracy * 100:.2f}%")
    return report, accuracy
