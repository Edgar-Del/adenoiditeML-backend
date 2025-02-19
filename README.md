# Diagnóstico de Adenoidite com Machine Learning

## Visão Geral do Projecto
Este projecto desenvolve uma solução de ML para diagnosticar adenoidite em crianças de 0 a 5 anos, adaptada para o Hospital Pediátrico do Lubango.

## Componentes Principais
- API de Machine Learning baseada em Python utilizando FastAPI.
- Modelos de Machine Learning:
  - Classificação com K-Nearest Neighbors (KNN).
  - Agrupamento com K-Means Clustering.
- Frontend desenvolvido em Next.js para interação com o usuário.

## Instruções de Configuração

### Configuração do Backend
1. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Treinar o modelo:
```bash
python -m ml.models.train
```

4. Rodar a API:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- POST `/api/v1/diagnosticar`: ENVIAR DADOS PARA TESTAR
- GET `/api/v1/health`: RECEBER O FEEDBACK "MÉDICO" DO SISTEMA

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

### Configuração do Frontend
1. Instale as dependências:
```bash
npm install
npm run dev
```
## Contribuições
1. Faça o fork do repositório
2. Crie sua branch de funcionalidade
3. Faça commit das suas alterações
4. Envie (push) para a branch
5. Crie um novo Pull Request

## Licença
GNU General Public License (GPL-3.0)
