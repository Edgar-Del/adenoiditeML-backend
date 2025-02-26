# 🏥 Diagnóstico de Adenoidite com Machine Learning

## Visão Geral do Projecto
Este projecto desenvolve uma solução de Aplicação de Machine Learning no Diagnóstico de Hipertrofia das Adenoides em Crianças de 0 a 5 Anos, um Estudo nos Hospitais Pediátricos de Benguela e Lubango, Angola. A solução inclui um backend em **FastAPI**, um modelo de **KNN + PCA** e uma API REST para predição.

## Componentes Principais
- API de Machine Learning baseada em Python utilizando FastAPI.
- Frontend desenvolvido em Next.js para interação com o usuário.


## 🚀 1️⃣ Configuração Inicial

Antes de iniciar os testes, configure seu ambiente:

### 📌 Passos iniciais:
```bash
# Ativar o ambiente virtual
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Instalar as dependências
pip install -r requirements.txt
```
---

## 📂 2️⃣ Gerar o Dataset

```bash
python dataset_generator.py
```
✅ **Saída esperada:**
```
✅ Dataset gerado com sucesso! Shape: (150, 13)
```
📌 O arquivo `dataset.csv` será salvo em `data/raw/`.

---

## 🔄 3️⃣ Executar o Pré-processamento

```bash
python -m app.preprocessamento
```
✅ **Saída esperada:**
```
✅ Treinamento: X shape (150, 12), y shape (150,)
```
📌 `LabelEncoder` e `Scaler` são salvos na pasta `models/saved/`.

---

## 📊 4️⃣ Treinar o Modelo

```bash
python -m ml.models.train data/raw/dataset.csv models/saved
```
✅ **Saída esperada:**
```
✅ Modelos salvos com sucesso!
🎯 Acurácia do modelo: XX.XX%
```
📌 Os modelos `modelo_knn.joblib` e `modelo_pca.joblib` serão salvos em `models/saved/`.

---

## 🩺 5️⃣ Testar a Predição

```bash
python -m ml.models.predict
```
✅ **Saída esperada:**
```json
{"diagnostico": true}  # ou false
```

---

## 🖥️ 6️⃣ Testar a API via FastAPI

1️⃣ **Iniciar a API:**
```bash
uvicorn app.main:app --reload
```
✅ **Saída esperada:**
```
INFO: Uvicorn running on http://127.0.0.1:8000
```

2️⃣ **Testar a API com `curl`:**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/diagnosticar' \
  -H 'Content-Type: application/json' \
  -d '{
    "idade_mes": 48,
    "genero": "M",
    "obstrucao_nasal_persistente": 1,
    "secrecao_nasal_purulenta": 0,
    "dor_de_garganta_e_dificuldade_para_engolir": 1,
    "Febre_e_mal_estar_geral": 1,
    "linfodenopatia_cervical": 0,
    "alteracao_na_voz": 1,
    "problemas_auditivos": 0,
    "sintomas_de_infeccao_recorrente": 1,
    "disturbios_de_sono_e_desenvolvimento": 1,
    "tamanho_adenoide": 3
  }'
```
✅ **Saída esperada:**
```json
{"diagnostico": true}  # ou false
```

---

## 📊 7️⃣ Avaliar o Modelo

```bash
python -m ml.utils.evaluation
```
✅ **Saída esperada:**
```
🎯 Acurácia do modelo: XX.XX%
📊 Relatório de Classificação:
...
✅ Matriz de Confusão gerada!
```

---

## API Endpoints

- POST `/api/v1/diagnosticar`: ENVIAR DADOS PARA TESTAR

## Docker


```bash
docker build -t adenoiditis-ml .
docker run -p 8000:8000 adenoiditis-ml
```

## Esturuta do projecto

- `app/`: App FastAPI
- `ml/`: Código do Agente ML
- `tests/`: Testes Unitários
- `data/`: Ficheiros de dados
- `models/`: Modelos Salvos
```
## Contribuições
1. Faça o fork do repositório
2. Crie sua branch de funcionalidade
3. Faça commit das suas alterações
4. Envie (push) para a branch
5. Crie um novo Pull Request

## Licença
GNU General Public License (GPL-3.0)
