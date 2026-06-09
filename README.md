# ⚽ Copa do Mundo 2026 - Pipeline de Machine Learning & Dashboard

Este repositório contém uma solução completa de Machine Learning para prever os resultados e placares dos jogos da **Copa do Mundo 2026**. O projeto conta com uma esteira automatizada de dados (ingestão, preparo, treino e previsão) e uma interface interativa hospedada diretamente no GitHub Pages.

---

## 🚀 Como Funciona a Arquitetura?

O projeto foi construído seguindo princípios modernos de **Serverless MLOps** (sem custo de servidores):

1. **Ingestão & Preparo (Python)**: Os dados históricos de partidas desde 1872 e os rankings FIFA são baixados, cruzados e usados para calcular o **rating ELO histórico** de cada seleção, além de médias móveis de gols e peso de forma recente (com decaimento exponencial).
2. **Modelagem de ML**: 
   - **Classificador (Random Forest)**: Prevê a probabilidade do resultado seco (Vitória / Empate / Derrota).
   - **Regressão (Poisson Regressor)**: Prevê a quantidade de gols marcados por equipe no confronto.
3. **Frontend Desacoplado**: O script de predição exporta apenas arquivos JSON contendo os parâmetros dos modelos e os jogos pré-calculados. O arquivo [index.html](index.html) consome estes JSONs e simula as partidas em tempo real utilizando JavaScript puro no navegador do usuário.
4. **Automação (GitHub Actions)**: Todo dia às 12:00 BRT, um robô baixa os resultados mais novos de futebol pelo mundo, retreina os modelos com a nova inteligência, salva as predições atualizadas e faz o deploy automático no **GitHub Pages**.

---

## 📁 Estrutura do Repositório

```text
├── .github/workflows/
│   └── ml_pipeline.yml       # Automação do GitHub Actions (Retreino Diário)
├── config/
│   └── params.yaml           # Parâmetros gerais de modelagem e caminhos
├── data/                     # Dados do projeto (Ignorados no Git, exceto processados essenciais)
│   ├── raw/                  # Resultados brutos, rankings FIFA e fixtures
│   └── processed/            # Features finais estruturadas em formato Parquet
├── models/                   # Binários dos modelos treinados (.joblib)
├── reports/                  # Saídas oficiais consumidas pelo Dashboard
│   ├── model_export.json     # Coeficientes do modelo de Poisson e estatísticas das seleções
│   └── previsoes_copa2026.json # Previsões pré-calculadas de todos os 104 jogos da Copa
├── src/                      # Código fonte do Pipeline
│   ├── 01_ingestao.py        # Ingestão de dados históricos e tabela da Copa 2026
│   ├── 02_preparo.py         # Engenharia de features, cálculo de ELO e forma recente
│   ├── 03_treinamento.py     # Validação temporal e treinamento dos modelos de ML
│   └── 04_previsao.py        # Inferência dos jogos e exportação dos resultados para JSON
├── index.html                # Painel interativo / Dashboard (HTML5, CSS3, JS)
├── requirements.txt          # Dependências do Python
└── README.md                 # Este arquivo de documentação
```

---

## 🛠️ Como Executar Localmente

### 1. Requisitos
Certifique-se de ter o **Python 3.10+** instalado em sua máquina.

### 2. Configurar o Ambiente
Crie um ambiente virtual e instale as dependências:
```bash
# Criar venv
python3 -m venv venv

# Ativar venv
source venv/bin/activate  # Linux/macOS
# ou: venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 3. Rodar a Esteira de Machine Learning
Você pode rodar os passos sequencialmente para atualizar a inteligência do modelo:
```bash
python src/01_ingestao.py
python src/02_preparo.py
python src/03_treinamento.py
python src/04_previsao.py
```

### 4. Abrir a Dashboard no Navegador
Como a dashboard consome arquivos locais utilizando requisições assíncronas (`fetch`), os navegadores bloqueiam o acesso direto por questões de segurança (CORS) caso você dê duplo clique no arquivo `index.html`. 

Para abrir o projeto localmente, suba um servidor web super leve na raiz do repositório:
```bash
python3 -m http.server 8000
```
Agora abra seu navegador e acesse:
👉 **`http://localhost:8000`**

---

## 📈 Métricas de Validação dos Modelos

Os modelos foram avaliados em um split de validação temporal estrito (jogos anteriores a 2022 para treino, e jogos pós-2022 para teste):

- **Random Forest Classifier**:
  - **Acurácia (Accuracy)**: **59.26%** (Superior à média humana de ~50% em palpites de torneios de futebol).
  - **Log Loss**: **0.9021** (Métrica de calibração de probabilidade).
- **Poisson Regressor (Gols)**:
  - **Erro Médio Absoluto (MAE)**: **0.9333 gols** por partida de cada seleção.

---

## 🌍 Publicação & Deploy Grátis (GitHub Pages)

Este projeto foi desenhado para ser publicado em qualquer hospedagem estática. Para habilitar o deploy gratuito no GitHub Pages:

1. Acesse seu repositório no GitHub.
2. Vá em **Settings** > **Pages**.
3. Em *Build and deployment*, selecione a branch `main` e a pasta `/ (root)`.
4. Salve e aguarde 1 minuto. O site estará disponível em:
   `https://<seu-usuario>.github.io/ml_copa/`
