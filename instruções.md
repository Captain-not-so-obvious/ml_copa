# 🏆 Pipeline de ML: Previsor de Resultados da Copa do Mundo 2026

Este documento define a arquitetura, as ferramentas e o passo a passo para a construção de um sistema de Machine Learning capaz de prever o **vencedor (W/D/L)** e o **placar esperado** de jogos da Copa do Mundo 2026.

> **Filosofia do modelo:** resultados recentes têm peso maior via decaimento exponencial, mas o histórico completo das equipes é preservado. Nenhuma API ou dado externo pago é utilizado.

---

## 🛠️ Stack Tecnológico

| Função                  | Biblioteca/Ferramenta                        |
|-------------------------|----------------------------------------------|
| Linguagem               | Python 3.11+                                 |
| Manipulação de Dados    | `pandas`, `numpy`                            |
| Machine Learning        | `scikit-learn`, `statsmodels`                |
| Serialização de Modelos | `joblib`                                     |
| Ingestão de Dados       | `kagglehub`, `requests`                      |
| Análise Exploratória    | `matplotlib`, `seaborn`                      |
| Ambiente                | `venv` local (Linux)                         |
| Versionamento           | Git / GitHub                                 |

---

## 📊 Fontes de Dados (100% Gratuitas)

| Fonte | O que fornece | Como acessar |
|---|---|---|
| **Kaggle** – `martj42/international-football-results` | Histórico de resultados internacionais (1872–atual) | `kagglehub` |
| **Kaggle** – `tadhgfitzgerald/fifa-international-soccer-mens-ranking-1993now` | Ranking FIFA histórico mensal por seleção | `kagglehub` |
| **OpenFootball** – `worldcup.json` (GitHub) | Fixtures e grupos da Copa 2026 | `requests` (JSON público) |

> **Por que não APIs em tempo real?** A Copa 2026 tem jogos com datas e fixtures já conhecidas. O OpenFootball é open-source, sem chave, sem limite de requisições e suficiente para todas as partidas do torneio.

---

## 🧠 Arquitetura de Modelos (Dois Modelos Complementares)

O sistema usa **dois modelos independentes** treinados sobre os mesmos dados:

```
                        ┌─────────────────────────┐
                        │   Features do Jogo      │
                        │  (Elo, forma, ranking…)  │
                        └────────────┬────────────┘
                                     │
               ┌─────────────────────┴──────────────────────┐
               ▼                                            ▼
  ┌────────────────────────┐                ┌───────────────────────────┐
  │  Modelo 1: Classifier  │                │  Modelo 2: Poisson Regr.  │
  │  RandomForest / XGBoost│                │  statsmodels GLM Poisson  │
  └────────────┬───────────┘                └──────────────┬────────────┘
               │                                           │
               ▼                                           ▼
     Resultado: W / D / L                    Gols esperados: μ_A e μ_B
                                                           │
                                                           ▼
                                              Distribuição de Poisson
                                          P(Score) = P(gols_A=x) * P(gols_B=y)
                                          → Placar mais provável + ranking placares
```

**Por que Poisson?** A distribuição de Poisson é o modelo estatístico padrão da literatura acadêmica para modelagem de gols em futebol. Dado um μ (média esperada de gols), ela gera a probabilidade de cada placar possível (0-0, 1-0, 2-1, etc.).

---

## 📂 Estrutura do Projeto

```text
ML_Copa/
│
├── data/                        # [IGNORADO NO GIT]
│   ├── raw/                     # Arquivos originais brutos
│   │   ├── results.csv          # Histórico de partidas (Kaggle)
│   │   └── fifa_ranking.csv     # Ranking FIFA histórico (Kaggle)
│   └── processed/               # Dados prontos para treino
│       └── features.parquet     # Dataset com todas as features calculadas
│
├── models/                      # [IGNORADO NO GIT]
│   ├── classifier.joblib        # Modelo de classificação (W/D/L)
│   └── poisson.joblib           # Modelo de regressão de gols
│
├── src/
│   ├── 01_ingestao.py           # Download automatizado de todas as fontes
│   ├── 02_preparo.py            # Limpeza + Feature Engineering completo
│   ├── 03_treinamento.py        # Treino dos dois modelos + avaliação
│   └── 04_previsao.py           # Inferência para jogos da Copa 2026
│
├── notebooks/
│   └── eda.ipynb                # Exploração e visualização dos dados
│
├── config/
│   └── params.yaml              # Hiperparâmetros e configurações centralizadas
│
├── reports/                     # Métricas, gráficos de avaliação gerados
│
├── .gitignore
└── requirements.txt
```

---

## 🚀 Passo a Passo da Implementação

### Passo 0: Configuração do Ambiente

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn statsmodels joblib kagglehub requests matplotlib seaborn pyyaml pyarrow
pip freeze > requirements.txt
```

---

### Passo 1: Ingestão de Dados (`src/01_ingestao.py`)

**Objetivo:** Obter todas as fontes de dados de forma automatizada, sem intervenção manual.

**Ações:**
1. Usar `kagglehub` para baixar `results.csv` (histórico de partidas internacionais).
2. Usar `kagglehub` para baixar `fifa_ranking.csv` (ranking FIFA mensal por seleção).
3. Usar `requests` para buscar os fixtures da Copa 2026 via `openfootball/worldcup.json`.
4. Salvar todos os arquivos brutos em `data/raw/`.

> **Nota:** O dataset do Kaggle `martj42/international-football-results` é atualizado pela comunidade e contém partidas até o presente. É nossa principal fonte histórica.

---

### Passo 2: Preparo e Feature Engineering (`src/02_preparo.py`)

**Objetivo:** Transformar os dados brutos em features que o modelo consiga aprender.

**Ações:**

**2.1 Limpeza:**
- Remover partidas com dados ausentes críticos (placar, seleções).
- Normalizar nomes de seleções (garantir consistência entre os datasets).
- Filtrar apenas partidas competitivas (Copa do Mundo, Eliminatórias, Copa das Nações) com peso maior, mantendo amistosos com peso reduzido.

**2.2 Decaimento Temporal (feature de peso):**
Calcular o peso `w` de cada partida com base na data:

```
w = 0.5 ^ (dias_desde_a_partida / 365)
```
- Partidas de hoje → peso ≈ 1.0
- Partidas de 1 ano atrás → peso ≈ 0.5
- Partidas de 4 anos atrás → peso ≈ 0.06
- O histórico completo é preservado, mas com influência decrescente.

**2.3 Features por Seleção (janelas deslizantes calculadas por equipe):**

| Feature | Descrição |
|---|---|
| `avg_gols_marcados_5j` | Média de gols marcados nos últimos 5 jogos |
| `avg_gols_sofridos_5j` | Média de gols sofridos nos últimos 5 jogos |
| `win_rate_1ano` | % de vitórias no último ano (jogos ponderados) |
| `form_score` | Pontuação de forma: W=3, D=1, L=0, média dos últimos 5 |
| `ranking_fifa` | Posição no ranking FIFA no mês do jogo |
| `elo_rating` | Rating Elo calculado internamente a partir do histórico |

**2.4 Features por Partida:**

| Feature | Descrição |
|---|---|
| `elo_diff` | Diferença de Elo entre time A e time B |
| `ranking_diff` | Diferença de posição no ranking FIFA |
| `h2h_win_rate` | % de vitórias do time A no histórico de confrontos diretos |
| `neutral_venue` | 1 = campo neutro (Copa do Mundo é sempre neutro) |
| `tournament_weight` | Copa do Mundo=1.0, Eliminatórias=0.75, Amistoso=0.25 |

**2.5 Target (variáveis que o modelo vai aprender):**
- `resultado`: 1=Vitória A, 0=Empate, -1=Vitória B → **para o Classifier**
- `gols_a` e `gols_b`: número inteiro de gols → **para o Poisson**

**2.6 Salvar:** Exportar o dataset final como `data/processed/features.parquet` (formato binário, ~5x mais rápido e menor que CSV).

---

### Passo 3: Treinamento e Avaliação (`src/03_treinamento.py`)

**Objetivo:** Treinar os dois modelos com divisão temporal correta e avaliar a qualidade.

> ⚠️ **Regra crítica: NUNCA usar `train_test_split` aleatório em séries temporais.**
> Isso causaria *data leakage* (o modelo "veria o futuro" durante o treino).
> **A divisão deve ser por data:** treino = antes de 2022, teste = 2022–2025.

**Ações:**

**3.1 Split Temporal:**
```
Treino: partidas até 31/12/2021
Teste:  partidas de 01/01/2022 até hoje
```

**3.2 Treinar Modelo 1 – Classifier (W/D/L):**
- Algoritmo: `RandomForestClassifier` com `sample_weight` (decaimento temporal)
- Avaliar com: `accuracy`, `log_loss`, `classification_report` (precision/recall por classe)

**3.3 Treinar Modelo 2 – Regressão de Poisson:**
- Algoritmo: `statsmodels GLM` com família Poisson
- Um modelo para `gols_a` e um para `gols_b` (ou modelo conjunto)
- Avaliar com: `MAE` (Mean Absolute Error) em gols, `RMSE`

**3.4 Salvar:**
- `models/classifier.joblib`
- `models/poisson.joblib`
- `reports/metricas.txt` com os resultados da avaliação

---

### Passo 4: Previsão – Copa 2026 (`src/04_previsao.py`)

**Objetivo:** Gerar previsões de vencedor e placar para qualquer partida da Copa 2026.

**Ações:**

1. Carregar `classifier.joblib` e `poisson.joblib`.
2. Buscar fixtures da Copa 2026 via OpenFootball (já baixado na ingestão).
3. Para cada jogo, montar o vetor de features (Elo, ranking, forma dos dois times).
4. Rodar `classifier.predict_proba()` → Probabilidades: P(W), P(D), P(L).
5. Rodar o modelo Poisson → `μ_A` e `μ_B` → simular distribuição de placares.
6. Gerar **ranking de placares mais prováveis** (ex: 2-1: 14%, 1-0: 12%, 1-1: 11%…).
7. Exibir resultado consolidado por jogo.

**Exemplo de output esperado:**
```
🇧🇷 Brasil vs Argentina 🇦🇷
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Vitória Brasil:  52.3%
Empate:          22.1%
Vitória Argentina: 25.6%

Placar mais provável: 1-0 (13.2%)
Gols esperados: Brasil 1.4 | Argentina 0.9
```

---

## 🛡️ Regras Essenciais do `.gitignore`

```gitignore
venv/
__pycache__/
*.pyc
data/
models/
reports/
.env
*.parquet
```

---

## 📈 Métricas de Sucesso

O modelo será considerado **bom o suficiente** para uso quando:

| Métrica | Meta mínima |
|---|---|
| Accuracy W/D/L (test set) | ≥ 52% (baseline humano: ~50%) |
| Log Loss | ≤ 1.0 |
| MAE em gols (por time) | ≤ 0.8 gols |
| Calibração de probabilidades | Curva de calibração próxima da diagonal |

> **Contexto:** Futebol tem alta variância — um modelo com 55% de accuracy já supera a maioria dos modelos publicados em literatura acadêmica. O objetivo não é perfeição, é consistência e calibração.

---

## 🔮 Evolução Futura (Após implementação)

1. **GitHub Actions:** Workflow cron para retreinar o modelo mensalmente com novos resultados.
2. **Frontend:** Dashboard em React/Django para visualizar previsões graficamente.
3. **Simulação de Monte Carlo:** Rodar 10.000 simulações do torneio inteiro para calcular P(campeão) por seleção.
4. **Ensemble:** Combinar Classifier + Poisson em um modelo ensemble para maior robustez.