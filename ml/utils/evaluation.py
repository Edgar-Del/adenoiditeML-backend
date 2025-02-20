import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test, save_path="results/confusao_matriz.png"):
    # Criar o diretório se não existir
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Prever
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Acurácia: {accuracy:.2f}")
    print("\nRelatório de Classificação:\n", report)

    # Gerar matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.ylabel("Rótulo Verdadeiro")
    plt.xlabel("Rótulo Previsto")

    # Salvar a matriz de confusão
    plt.savefig(save_path)
    plt.close()
