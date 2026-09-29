# Churn Prediction — Cartão de Crédito (Banco)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema que **identifica clientes com alta probabilidade de churn**, estima o **valor de cada cliente (CLV)**,
calcula uma **recomendação de retenção** com ROI esperado e roteia a ação para time humano ou fluxo automatizado.

Projeto de portfólio (nível intermediário) em ciência de dados + engenharia/implantação:
modelo em **lote agendado**, banco **Supabase** (Postgres), orquestração com **n8n** e **Docker**.

---

## 1. Problema de negócio

Perder um cliente de cartão de crédito custa caro: custo de aquisição, receita futura (anuidade, juros,
interchange) e efeito em reputação. A pergunta central:

> Dado o comportamento recente do cliente, qual a probabilidade de churn e qual ação de retenção
> **maximiza o lucro esperado** (e não só o recall)?

O sistema não diz apenas "quem está em risco", mas **por quê** (SHAP) e **quanto vale agir** (CLV × custo de
campanha), evitando gastar com clientes de baixo valor ou baixo risco.

## 2. Dados

- **BankChurners — Credit Card Customers (Kaggle, CC0):** ~10.127 clientes, churn ~16%.
- Variáveis: demográficas, categoria do cartão, tempo de relacionamento, produtos, meses de inatividade,
  contatos com o SAC, limite, saldo rotativo, valor/quantidade de transações (12m) e variação Q4vQ1.
- As 2 colunas `Naive_Bayes_*` (vazamento do target) são **descartadas**.

## 3. Como foi feito (pipeline)

| Etapa | Artefato |
|---|---|
| EDA orientada a negócio | `notebooks/01_eda.ipynb` |
| Engenharia de atributos (`src/features.py`, fonte única treino+servidor) | `notebooks/02_feature_engineering.ipynb` |
| Modelagem: Logística (baseline) vs XGBoost + CV estratificada + desbalanceamento | `notebooks/03_modeling.ipynb` |
| Explicabilidade SHAP (global + local) | `notebooks/04_shap.ipynb` |
| CLV, matriz custo/benefício, threshold ótimo, segmentação | `notebooks/05_business_layer.ipynb` |
| Treino end-to-end → artefatos | `src/train.py` |

### Resultados (holdout de 20%)

| Modelo | PR-AUC | Recall@0.5 | F1 | Recall@0.3 |
|---|---|---|---|---|
| Regressão Logística | 0.77 | 0.87 | 0.68 | 0.93 |
| **XGBoost (final)** | **0.969** | **0.90** | **0.90** | **0.94** |

Corte operacional recomendado de **0.51** (maximiza lucro esperado da campanha); nesse corte a holdout
acerta 90% dos churners com 90% de precisão. O XGBoost domina a logística principalmente na região de
precisão relevante para retenção (PR-AUC é a métrica de foco, não acurácia).

### Camada de negócio — decisão e segmentação

Premissas financeiras (configuráveis em `src/business.py`, documentadas no relatório):

- Margem 4% sobre transações; juros 12% sobre saldo rotativo; anuidade por categoria;
- CLV = lucro anual × fator de anuidade (5 anos, 10% de desconto);
- Custo campanha: **$80** (contato personalizado) / **$5** (automatizada); lift de retenção 15% / 5%.

**Política por-cliente** (`lift·p·CLV > custo`) rende mais que qualquer corte global
(EV ≈ **$20,7k** vs **$19,0k** na holdout — o oposto de "reter todo mundo").

| Segmento | Critério | Ação |
|---|---|---|
| `retencao_prioritaria` (476) | alto risco × alto valor | time humano, oferta personalizada |
| `acao_automatizada` (1.166) | alto risco × baixo valor | oferta automática (e-mail/app) |
| `monitorar_upsell` (4.588) | baixo risco × alto valor | não gastar; buscar expansão |
| `monitorar` (3.897) | baixo risco × baixo valor | sem ação |

## 4. Arquitetura de implantação

**Supabase fica em nuvem** (Postgres gerenciado + API REST). **Docker roda só** o serviço de scoring (FastAPI)
e o **n8n**.

```
Supabase (clientes) → scoring-service (modelo + CLV + segmento) → Supabase (churn_scores)
                                                    ↑  n8n agenda e roteia (Slack/e-mail)  ↓
```

### Estrutura

```
churn-project/
├── docker-compose.yml          # churn-scoring (API :8000) + n8n (:5679)
├── .env.example                # copiar para .env
├── requirements.txt            # ambiente de treino/análise
├── data/raw/, data/processed/  # dataset + saídas de scoring (inclui o laudo SHAP do lote)
├── notebooks/                  # 01_eda..05_business_layer (executados)
├── src/features.py             # engenharia de atributos (fonte única)
├── src/business.py             # CLV, threshold, segmentação, ROI
├── src/explain.py             # SHAP por cliente (fonte única treino+serviço)
├── src/train.py                # treino end-to-end
├── tests/                      # pytest: features, negócio, serviço, SHAP e dashboard
├── artifacts/                  # churn_model.pkl + metadados (não versionado)
├── scoring-service/            # FastAPI + Dockerfile + requirements
├── supabase/schema.sql         # tabelas clientes e churn_scores (+ view priorizada)
├── supabase/load_clientes.py   # carrega o CSV no Supabase via REST
├── n8n/churn_workflow.json     # fluxo agendado (importável)
└── dashboard/app.py            # Streamlit (priorização + explicabilidade)
```

## 5. Como reproduzir

### 5.1 Ambiente de treino

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows | source .venv/bin/activate (Linux/Mac)
pip install -r requirements.txt
kaggle datasets download -d sakshigoyal7/credit-card-customers -p data/raw --unzip
python src/train.py               # gera artifacts/* e data/processed/churn_scores_demo.csv
python scripts/build_notebooks.py # gera/executa notebooks 01..05
python -m pytest                  # 67 testes (ver 5.6)
```

### 5.2 Supabase

1. Crie um projeto (plano Free) em [supabase.com](https://supabase.com);
2. Rode `supabase/schema.sql` no **SQL Editor** (tabelas `clientes` e `churn_scores`, view `v_priorizados`, RLS);
3. Copie `.env.example` → `.env` e preencha `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`;
4. Carregue o dataset na tabela `clientes`:
   ```bash
   pip install python-dotenv
   python supabase/load_clientes.py --csv data/raw/BankChurners.csv
   ```
   (alternativa: no Table Editor do Supabase, import CSV e ajuste os nomes de colunas para o snake_case do schema).

### 5.3 Serviço de scoring + n8n (Docker)

**Pré-requisito:** os artefatos de `artifacts/` não são versionados e o `Dockerfile` os copia para dentro
da imagem — rode `python src/train.py` **antes** do primeiro build, senão o `docker compose up --build` falha.

```bash
cp .env.example .env   # preencha as variáveis
docker compose up -d --build
```

- `GET  http://localhost:8000/health` → serviço de scoring;
- `POST http://localhost:8000/run-scoring` → roda o lote (chave `X-API-Key` = `RUN_API_KEY`);
- `http://localhost:5679` → n8n (login básico do `.env`; a porta interna 5678 é publicada em 5679).

Sem Docker, o mesmo serviço roda local — precisa do `src/` no `PYTHONPATH` (o Dockerfile faz essa cópia):

```bash
set PYTHONPATH=src && python -m uvicorn app:app --app-dir scoring-service --port 8000
```

**Modo demo (sem Supabase):** defina `CHURN_SOURCE=local` no `.env`; o serviço lê
`data/raw/BankChurners.csv` (montado em `/data`) e grava `data/processed/churn_scores_local.csv`.

Além dos scores, **cada rodada do lote produz o laudo SHAP por cliente** em
`data/processed/shap_by_client.json` (em ~5s para 10k clientes). O mesmo arquivo
é usado pela tabela `shap` do container porque o compose monta `./data:/data`, então
o laudo exibido no dashboard corresponde sempre ao modelo que gerou o score — e não
ao último treino. Se o cálculo do SHAP falhar, o lote ainda conclui: ele só não
reescreve o laudo.

`DATA_DIR` é opcional: sem ele, o serviço usa a pasta `data/` do projeto (e `/data`
dentro do container, que é o volume montado).

### 5.4 Importar o fluxo no n8n

```bash
docker compose exec n8n n8n import:workflow --input=/data/churn_workflow.json
```

O fluxo (`n8n/churn_workflow.json`) faz: **Schedule Trigger** (domingo 22h) → **HTTP Request**
(POST `/run-scoring` com `X-API-Key`) → **IF** (existem prioritários?) → ramo `true` notifica
Slack/e-mail da equipe; ramo `false` não faz nada. Ajuste credenciais (Slack/SMTP) e destinatários no UI do n8n.

### 5.5 Dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

- Sem variáveis Supabase → usa `data/processed/churn_scores_demo.csv` (demo);
- Com `SUPABASE_URL` + `SUPABASE_ANON_KEY` → lê a view `v_priorizados` (última rodada de cada cliente).
- Abas: visão geral (KPIs, quadrantes risco×valor, segmentos), lista priorizada (com filtros + download CSV)
  e explicabilidade por cliente (motivos SHAP).
- Os filtros da barra lateral (segmento, score mínimo e busca por ID) valem para **todas** as abas,
  e a aba de explicabilidade informa a origem do laudo (lote ou treino).

### 5.6 Testes

```bash
python -m pytest          # 67 testes, ~60s (3 exigem as chaves do Supabase)
```

| Arquivo | O que trava |
|---|---|
| `tests/test_features.py` | contrato de colunas do treino, descarte das colunas de leak, cobertura de todos os rótulos categóricos (renda/escolaridade), imputação, fórmulas das features derivadas, divisão por zero, e **lote parcial não derruba o scoring** |
| `tests/test_business.py` | fórmula do CLV e do fator de anuidade, regra `lift·p·CLV > custo` (estritamente maior), invariante de que a decisão por-cliente domina qualquer corte global, os 4 quadrantes, ROI e consistência dos parâmetros salvos |
| `tests/test_scoring_service.py` | mapeamento snake_case→camelCase do Supabase, cobertura de todas as colunas do `schema.sql`, contrato de saída do `score_frame`, **igualdade com os scores do treino**, e a API HTTP (`/health`, `/run-scoring`, `X-API-Key`) em modo local |
| `tests/test_explain.py` | formato do laudo SHAP consumido pelo dashboard, ordem por impacto, determinismo, gravação do laudo datado da rodada e **falha do SHAP não derruba o lote** |
| `tests/test_dashboard.py` | execução real do app via `AppTest`: KPIs, filtros da sidebar afetando todas as abas, filtro sem resultado, laudo da última rodada, **paginação da view** (o PostgREST corta em 1000 linhas por requisição, e sem desempate em `score_churn` a paginação repete e perde clientes) e o caminho Supabase (pulado se as chaves não estiverem no ambiente) |

Os testes que dependem de `artifacts/` ou do dataset são pulados automaticamente se eles não existirem
(`pytest.skip`), então a suíte roda em qualquer estado do repositório.

## 6. Decisões de projeto (por quê)

- **Métricas:** PR-AUC/Recall/F1 — classe rara (16%) e decisão financeira tornam acurácia enganosa.
- **Desbalanceamento:** `scale_pos_weight` no XGBoost e `class_weight="balanced"` na logística
  (equivale, em efeito, ao SMOTE/undersampling sem inflar tempo de treino).
- **Threshold:** otimizado por **lucro esperado**, não fixado em 0.5; e a melhor política é **por-cliente**.
- **Ingenieria única (`src/features.py`):** mesma função no treino e no servidor elimina *train/serve skew*.
- **Explicabilidade única (`src/explain.py`):** o mesmo código de SHAP roda no treino e no lote, então o laudo por
  cliente nunca fica defasado em relação ao modelo que gerou o score.
- **Supabase cloud (e não self-hosted):** overhead de rodar Postgres+Auth+Realtime+Studio não agrega valor ao
  portfólio; Docker fica só para o que precisa de isolamento reproduzível (scoring e n8n).
- **n8n como orquestrador:** agendamento, roteamento condicional e notificação sem código; demonstra a camada
  de MLOps/automação sem framework pesado.

## 7. Futuro / melhorias (nível avançado)

- Análise de sobrevivência (Kaplan-Meier/Cox) para estimar **quando** o cliente sai;
- Calibração das probabilidades (financeiro) e auditoria de **fairness** por gênero/idade/renda;
- Simulação de campanha com ROI monetizado e métricas de *challenger*;
- Contextualização Brasil (concorrência de bancos digitais, portabilidade, cartão sem anuidade) com
  valores em R$.

## 8. Licença

Código deste repositório sob licença **MIT** — ver [`LICENSE`](LICENSE).

O dataset **BankChurners — Credit Card Customers** é distribuído separadamente sob **CC0-1.0** pelo Kaggle
e não é abrangido pela licença MIT. Ele não é versionado aqui (ver `.gitignore`); é baixado pelo comando
`kaggle datasets download` da seção 5.1.

---

*Relatório executivo com as premissas e números detalhados: [`docs/relatorio_executivo.md`](docs/relatorio_executivo.md).*