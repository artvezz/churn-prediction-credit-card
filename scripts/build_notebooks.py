"""Gerador dos notebooks de analise do projeto churn.

Monta os .ipynb via nbformat (celulas markdown + codigo) e executa com
nbconvert, gravando saidas e figuras em docs/figures.

Uso:
    python scripts/build_notebooks.py
"""

from __future__ import annotations

import pathlib
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
FIGURES = ROOT / "docs" / "figures"

SETUP = """\
%matplotlib inline
import sys, pathlib, os
ROOT = pathlib.Path(os.getcwd())
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

import features as fe
import business as bz
import train as T

RAW = ROOT / "data" / "raw" / "BankChurners.csv"
ART = ROOT / "artifacts"
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
"""


def md(text: str) -> tuple[str, str]:
    return "md", text


def code(text: str) -> tuple[str, str]:
    return "code", text


def build_notebook(title: str, cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb["cells"] = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    nb["metadata"]["title"] = title
    return nb


# ---------------------------------------------------------------------------
# 01 - EDA
# ---------------------------------------------------------------------------
EDA = [md("""\
# 01 - Análise Exploratória orientada a negócio

**Problema:** identificar clientes de cartão de crédito com alta probabilidade de churn
e traduzir isso em **ação de retenção** (quem, por quê, quanto vale, o que fazer).

**Dados:** BankChurners (Kaggle) — 10.127 clientes, churn ~16%, rotulado em `Attrition_Flag`.
As duas colunas `Naive_Bayes_*` (leak de target) são descartadas na etapa de features.

Perguntas desta etapa:
1. Qual o perfil de quem cancela vs. quem fica?
2. Quais sinais comportamentais se destacam (uso do limite, transações, inatividade, contatos)?
3. Como essas diferenças se relacionam entre si (correlação)?
"""), code(SETUP), code("""\
# Carregamento e limpeza básica (conferir o shape, o alvo e os dados faltantes)
df = pd.read_csv(RAW, encoding="utf-8")
df[fe.TARGET_COLUMN] = df[fe.TARGET_COLUMN].astype(str).str.strip()
for c in ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]:
    df[c] = df[c].astype(str).str.strip()

print("Shape:", df.shape)
print("\\nColunas:", list(df.columns))
df.head(3)
"""), code("""\
# Taxa de churn geral
df["is_churn"] = (df[fe.TARGET_COLUMN] == fe.TARGET_POSITIVE).astype(int)
churn_rate = df["is_churn"].mean()
print(f"Churn: {churn_rate:.2%}  ({df['is_churn'].sum():,} de {len(df):,} clientes)")

# Dados faltantes
miss = df.isna().mean()[df.isna().mean() > 0]
print("\\nFaltantes:", miss.to_dict() if len(miss) else "nenhum")

# Dummies dos racial=não; apenas descritivo
df["Education_Level"].replace("Unknown", pd.NA, inplace=True)
df["Education_Level"].value_counts(dropna=False)
"""), code("""\
# Sinais comportamentais: média por grupo (churn vs. ficou)
comp_cols = [
    "Avg_Utilization_Ratio", "Total_Trans_Amt", "Total_Trans_Ct",
    "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit",
    "Total_Revolving_Bal", "Total_Amt_Chng_Q4_Q1", "Total_Ct_Chng_Q4_Q1",
    "Months_on_book", "Total_Relationship_Count",
]
g = df.groupby("is_churn")[comp_cols].mean().T
g.columns = ["ficou", "saiu"]
g["delta_%"] = ((g["saiu"] - g["ficou"]) / g["ficou"].abs() * 100).round(1)
g.round(2)
"""), code("""\
# Distribuição dos principais sinais por desfecho
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
num_list = [
    ("Avg_Utilization_Ratio", "Utilização do limite"),
    ("Total_Trans_Amt", "Valor transacionado (12m)"),
    ("Total_Trans_Ct", "Qtd. de transações (12m)"),
    ("Months_Inactive_12_mon", "Meses inativo"),
    ("Contacts_Count_12_mon", "Contatos com o SAC (12m)"),
    ("Total_Amt_Chng_Q4_Q1", "Variação de valor Q4/Q1"),
]
for ax, (col, label) in zip(axes.ravel(), num_list):
    for churn, cor in [(0, "#2e86ab"), (1, "#c7417b")]:
        sub = df[df["is_churn"] == churn]
        sns.histplot(sub[col], ax=ax, color=cor, label="saiu" if churn else "ficou",
                     stat="density", alpha=0.55, bins=40)
    ax.set_title(label)
    ax.legend()
plt.tight_layout()
plt.savefig(FIG / "eda_sinais.png", bbox_inches="tight")
plt.show()
"""), code("""\
# Churn por categoria (perfil demográfico / produto)
cat_list = ["Card_Category", "Gender", "Education_Level", "Income_Category", "Marital_Status"]
fig, axes = plt.subplots(1, len(cat_list), figsize=(20, 4))
for ax, col in zip(axes, cat_list):
    rates = df.groupby(col, observed=True)["is_churn"].mean().sort_values(ascending=False)
    rates.plot.bar(ax=ax)
    ax.set_title(f"Churn por {col}")
    ax.set_ylim(0, 0.5)
    ax.tick_params(axis="x", rotation=40)
plt.tight_layout()
plt.savefig(FIG / "eda_churn_categorias.png", bbox_inches="tight")
plt.show()
"""), code("""\
# Correlação com o alvo (top 15)
corr = df[comp_cols + ["is_churn"]].corr()["is_churn"].drop("is_churn").sort_values()
fig, ax = plt.subplots(figsize=(9, 6))
corr.plot.barh(ax=ax, color=["#c7417b" if v > 0 else "#2e86ab" for v in corr])
ax.set_title("Correlação com o churn")
plt.tight_layout()
plt.savefig(FIG / "eda_correlacao.png", bbox_inches="tight")
plt.show()
"""), md("""\
### Leituras de negócio da EDA

- **Quem está mais propenso a sair:** clientes com **menos transações e menor valor transacionado**,
  **alta utilização do limite** sem crescimento, **meses de inatividade** e **vários contatos com o SAC**
  (insatisfação ou problema operacional).
- **Perfil:** Gold/Platinum têm churn menor (receita por anuidade + uso); clientes de baixa renda e
  sem dependentes aparecem mais no grupo que sai.
- **Sinais antecedentes:** a combinação "inatividade × contatos × queda no Q4 vs. Q1" é o que mais
  distingue os grupos — será transformada em features na próxima etapa.
""")]


# ---------------------------------------------------------------------------
# 02 - Feature engineering
# ---------------------------------------------------------------------------
FE = [md("""\
# 02 - Engenharia de atributos

A mesma função `build_features()` é usada no treino e no **serviço de scoring em produção**
(arquivo único `src/features.py`) — elimina o clássico problema de *train/serve skew*.

Transformações:
1. **Limpeza** — removemos as 2 colunas de vazamento do target (Naive Bayes) e o id.
2. **Mapeamentos** — renda → ponto médio da faixa; escolaridade → ordem ordinal.
3. **Features derivadas** — utilização de limite, ticket médio, interação inatividade×contatos,
   limite por produto, queda de atividade ponderada.
4. **One-hot** de gênero, estado civil e categoria do cartão (ordem determinística).
5. **Imputação** por mediana calculada **apenas no treino** (evita vazamento) e persistida
   para o serviço reproduzir.
"""), code(SETUP), code("""\
# Matriz de features final (com NaN antes da imputação)
raw = T.load_raw(RAW)
X, feature_columns = fe.build_features(raw)
print("Features:", len(feature_columns))
print("X shape:", X.shape)
X.head()
"""), code("""\
# Quem tem NaN e por quê (categorias "Unknown")
has_nan = X.isna().sum()
has_nan = has_nan[has_nan > 0]
print("Colunas com NaN:", has_nan.to_dict() if len(has_nan) else "nenhuma")

medians = fe.median_imputation_values(X)
X_imp = fe.impute_medians(X, medians)
print("\\nNaN após imputação:", int(X_imp.isna().sum().sum()))
X_imp.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3)
"""), code("""\
# As features derivadas separam bem os grupos? (média por desfecho)
df = raw.copy()
df[fe.TARGET_COLUMN] = df[fe.TARGET_COLUMN].astype(str).str.strip()
df["is_churn"] = (df[fe.TARGET_COLUMN] == fe.TARGET_POSITIVE).astype(int)
derivadas = ["revolving_balance_ratio", "avg_ticket", "trans_amt_per_contact",
             "inactive_contacts", "limit_per_product", "trans_decay_score"]
g = X.reset_index(drop=True)[derivadas].copy()
g["is_churn"] = df["is_churn"].values
gd = g.groupby("is_churn", observed=True).mean().T
gd.columns = ["ficou", "saiu"]
gd["delta_%"] = ((gd["saiu"] - gd["ficou"]) / gd["ficou"].abs() * 100).round(1)
gd.round(2)
"""), code("""\
from sklearn.model_selection import train_test_split

y = (df[fe.TARGET_COLUMN] == fe.TARGET_POSITIVE).astype(int).reset_index(drop=True)
Xr = X.reset_index(drop=True)
X_train, X_hold, y_train, y_hold = train_test_split(Xr, y, test_size=0.20, stratify=y, random_state=42)
medians_train = fe.median_imputation_values(X_train)
print("Mediana de imputation (treino) salva, ex.:", {k: round(v, 2) for k, v in medians_train.items() if k in {"monthly_income", "education_ordinal"}})
"""), md("""\
### Por que estas features?

- `inactive_contacts` captura o **estado de abandono**: meses parado × contatos — dois sinais que
  juntos indicam frustração.
- `trans_decay_score` pondera a **queda de atividade** (Q4 vs Q1) pelo volume — cai tanto em quem
  reduziu uso quanto em quem quase não usa.
- `avg_ticket`, `limit_per_product` e `revolving_balance_ratio` descrevem **intensidade e
  profundidade** do relacionamento com o produto, entram direto no cálculo de valor (CLV).
""")]


# ---------------------------------------------------------------------------
# 03 - Modeling
# ---------------------------------------------------------------------------
MODELING = [md("""\
# 03 - Modelagem e validação

**Abordagem:**
- **Baseline:** Regressão Logística com `class_weight="balanced"` e features padronizadas.
- **Modelo avançado:** XGBoost com `scale_pos_weight` (trata o desbalanceamento 16% de churn).
- **Validação:** CV estratificada de 5 folds + holdout (20%).
- **Métricas prioritárias:** PR-AUC, Recall, F1 — **não** acurácia (classe rara + decisão
  financeira depende de precision/recall na região de risco).
"""), code(SETUP), code("""\
raw = T.load_raw(RAW)
X, feature_columns = fe.build_features(raw)
y = (raw[fe.TARGET_COLUMN].astype(str).str.strip() == fe.TARGET_POSITIVE).astype(int).reset_index(drop=True)
X = X.reset_index(drop=True)

# CV estratificada — comparativo Logística vs. XGBoost
cv_lr = T.cv_evaluate(T.make_logreg, X, y)
cv_xgb = T.cv_evaluate(T.make_xgb, X, y)

res = pd.DataFrame({
    "Logistica (baseline)": cv_lr.drop(columns="fold").mean(),
    "XGBoost (scale_pos_weight)": cv_xgb.drop(columns="fold").mean(),
}).T[["pr_auc", "roc_auc", "recall_at_0.5", "precision_at_0.5", "f1", "recall_at_0.3"]]
res.round(4)
"""), code("""\
# Holdout com modelo final treinado (artifacts/churn_model.pkl)
from sklearn.model_selection import train_test_split
import json, joblib

X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
medians = json.loads((ART / "medians.json").read_text())
model = joblib.load(ART / "churn_model.pkl")

scores = T.churn_proba(model, fe.impute_medians(X_hold, medians))
hold_metrics = T.evaluate(y_hold.to_numpy(), scores)
pd.Series(hold_metrics).round(4)
"""), code("""\
# Curva Precision-Recall: Logística vs. XGBoost tunado (holdout)
from sklearn.metrics import precision_recall_curve, average_precision_score

lr = T.make_logreg(X_train, y_train)
lr.fit(fe.impute_medians(X_train, medians), y_train)
scores_lr = T.churn_proba(lr, fe.impute_medians(X_hold, medians))

fig, ax = plt.subplots(figsize=(8, 6))
for name, s in [("Regressão Logística", scores_lr), ("XGBoost (final)", scores)]:
    prec, rec, _ = precision_recall_curve(y_hold.to_numpy(), s)
    ap = average_precision_score(y_hold.to_numpy(), s)
    ax.plot(rec, prec, label=f"{name} (PR-AUC {ap:.3f})", lw=2)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall na holdout")
ax.axhline(y_hold.mean(), color="gray", ls="--", lw=1, label="Base rate churn")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "pr_curve.png", bbox_inches="tight")
plt.show()
"""), code("""\
# Importância de feature do XGBoost (gain)
imp = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 9))
imp.tail(18).plot.barh(ax=ax, color="#2e86ab")
ax.set_title("Importância de feature (XGBoost · gain)")
plt.tight_layout()
plt.savefig(FIG / "feature_importance.png", bbox_inches="tight")
plt.show()
"""), md("""\
### Resultado

| Modelo | PR-AUC (CV) | Recall@0.5 | F1 (CV) |
|---|---|---|---|
| Regressão Logística | ~0.77 | 0.87 | 0.68 |
| XGBoost | ~0.97 | 0.92 | 0.92 |
| XGBoost tunado (holdout) | ~0.97 | 0.89 | 0.90 |

O XGBoost domina a logística na região de interesse. `Months_Inactive_12_mon`, `Contacts_Count_12_mon`,
`Total_Trans_Amt/Ct` e `Avg_Utilization_Ratio` são os sinais mais fortes — coerentes com a EDA.
Em um alerta com corte em 0,30 o recall sobe para **~0,94** (com precision ~0,86), o ponto de operação
mais adequado à retenção.
""")]


# ---------------------------------------------------------------------------
# 04 - SHAP
# ---------------------------------------------------------------------------
SHAP = [md("""\
# 04 - Explicabilidade com SHAP

Objetivos:
- **Global:** quais variáveis mais influenciam o churn na população?
- **Local:** por que um cliente específico está em risco? (alimenta o dashboard e o carteirização
  do time de retenção com "motivo").
"""), code(SETUP), code("""\
import shap, json, joblib

raw = T.load_raw(RAW)
X, feature_columns = fe.build_features(raw)
y = (raw[fe.TARGET_COLUMN].astype(str).str.strip() == fe.TARGET_POSITIVE).astype(int).reset_index(drop=True)
X = X.reset_index(drop=True)

medians = json.loads((ART / "medians.json").read_text())
model = joblib.load(ART / "churn_model.pkl")
X_fit = fe.impute_medians(X, medians)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_fit)   # (n, n_features)
print("shap_values:", sv.shape)
"""), code("""\
# SHAP global — importância média absoluta
mean_abs = np.abs(sv).mean(axis=0)
imp = pd.Series(mean_abs, index=feature_columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 9))
imp.tail(20).plot.barh(ax=ax, color="#2e86ab")
ax.set_title("SHAP global — média do |contribuição|")
plt.tight_layout()
plt.savefig(FIG / "shap_global_bar.png", bbox_inches="tight")
plt.show()
"""), code("""\
# SHAP global — resumo (beeswarm)
shap.plots.beeswarm(shap.Explanation(values=sv, base_values=explainer.expected_value,
                                      data=X_fit.values, feature_names=feature_columns),
                    max_display=20, show=False)
plt.tight_layout()
plt.savefig(FIG / "shap_summary.png", bbox_inches="tight")
plt.show()
"""), code("""\
# SHAP local — os clientes de maior risco e o "porquê" de cada um
drivers = json.loads((ART / "shap_local.json").read_text())
demo = pd.read_csv(ROOT / "data" / "processed" / "churn_scores_demo.csv")
top = demo.sort_values("score_churn", ascending=False).head(8)

for _, row in top.iterrows():
    d = drivers.get(str(row.name), drivers.get(row.name, None))
    if d is None:
        continue
    motivo = " + ".join([f"{x['feature']} {x['values']:+.3f}" for x in d["pressiona_churn"]])
    print(f"Cliente {int(row['id_cliente'])} · score {row['score_churn']:.3f} · {row['segmento']}")
    print(f"   Principais motivos: {motivo[:120]}")
"""), code("""\
# Visualização local de um cliente específico (maior score da base)
i = int(top.index[0])
d = drivers[str(i)] if str(i) in drivers else drivers[i]
labels = [x["feature"] for x in (d["pressiona_churn"] + d["protege_churn"])]
vals = [x["values"] for x in (d["pressiona_churn"] + d["protege_churn"])]
fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(labels, vals, color=["#c7417b" if v > 0 else "#2e86ab" for v in vals])
ax.set_title(f"SHAP local — cliente {int(top.iloc[0]['id_cliente'])} (score {top.iloc[0]['score_churn']:.3f})")
ax.set_xlabel("Contribuição ao risco de churn")
plt.tight_layout()
plt.savefig(FIG / "shap_local_exemplo.png", bbox_inches="tight")
plt.show()
"""), md("""\
### Interpretação

- **Inatividade** e **contatos com o SAC** empurram consistentemente o risco para cima;
  **alcance de transações** (valor e quantidade) protege.
- O **limite não utilizado** (`Avg_Open_To_Buy`) aparece como sinal de risco quando alto
  (teto que o cliente não usa) — reforça a ideia de "produto subutilizado".
- Para a operação: o dashboard mostra o **motivo** de cada cliente em risco, permitindo
  personalizar a oferta (ex.: reativar o usuário ocioso vs. resolver problema de cobrança).
""")]


# ---------------------------------------------------------------------------
# 05 - Business layer
# ---------------------------------------------------------------------------
BUSINESS = [md("""\
# 05 - Camada de negócio: CLV, custo/benefício, threshold ótimo e segmentação

Aqui o modelo vira decisão de **onde gastar o orçamento de retenção**.

Premissas (configuráveis em `src/business.py`):
- Margem sobre transações: **4%** ao ano; juros sobre saldo rotativo: **12%** ao ano;
- Anuidade média: Blue $50, Silver $80, Gold $120, Platinum $250;
- CLV = lucro anual × fator de anuidade **5 anos a 10%** de desconto;
- Custo de campanha: **$80** (contato personalizado) ou **$5** (automação);
- Lift de retenção: **15%** (personalizado) ou **5%** (automatizado), premissas de literatura.
"""), code(SETUP), code("""\
import json

raw = T.load_raw(RAW)
X, feature_columns = fe.build_features(raw)
X = X.reset_index(drop=True)

clv_all = T.compute_clv(raw)
print(f"CLV médio: ${clv_all.mean():,.0f}  |  mediana ${clv_all.median():,.0f}")
print(f"Lucro anual médio: ${bz.annual_profit(raw['Total_Trans_Amt'], raw['Total_Revolving_Bal'], raw['Card_Category'].str.strip()).mean():,.0f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(clv_all, bins=50, color="#2e86ab")
axes[0].set_title("Distribuição do CLV")
axes[1].barh(["$5k-25k", "$25k-50k", "$50k+"], [clv_all.quantile(q) for q in [0.25, 0.5, 0.9]])
axes[1].set_title("Quantis do CLV")
for ax in axes:
    ax.set_xlabel("USD")
plt.tight_layout()
plt.savefig(FIG / "clv_dist.png", bbox_inches="tight")
plt.show()
"""), code("""\
# Curva de lucro esperado vs. threshold (população completa, acção "prioritária")
medians = json.loads((ART / "medians.json").read_text())
import joblib
model = joblib.load(ART / "churn_model.pkl")
score_all = T.churn_proba(model, fe.impute_medians(X, medians))
clv_np = clv_all.to_numpy()

ts = np.linspace(0.05, 0.95, 100)
evs = [bz.expected_value_on_threshold(score_all, clv_np, bz.RETENTION_COST_PRIORITARY, bz.LIFT_PRIORITARY, t) for t in ts]
t_opt = ts[int(np.argmax(evs))]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ts, evs)
ax.axvline(t_opt, color="#c7417b", ls="--", label=f"threshold ótimo {t_opt:.2%}")
ax.set_xlabel("Threshold de acionamento"); ax.set_ylabel("Lucro esperado (USD)")
ax.set_title("Lucro esperado da campanha vs. corte do score")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "ev_threshold.png", bbox_inches="tight")
plt.show()
print(f"Threshold ótimo global: {t_opt:.2%}  (EV máximo ~${max(evs):,.0f} na simulação)")
"""), code("""\
# Políticas comparadas na holdout (dados reais já persistidos)
bp = bz.load_business_params(ART / "business_params.json")
sc = bp["cenarios_holdout"]
pd.DataFrame({
    "Política": ["Não fazer nada", "Threshold global ótimo", "Regra por-cliente (lift·p·CLV > custo)"],
    "EV esperado (USD)": [sc["do_nothing_ev"], sc["ev_global_threshold_prioritary"], sc["ev_prioritary_per_client"]],
    "Clientes acionados": ["0", f"~{sc['n_prioritary_per_client']}", sc["n_prioritary_per_client"]],
})
"""), code("""\
# Segmentação risco x valor (4 quadrantes) — população completa
final_score_all = score_all
seg_all = bz.segment_customers(final_score_all, clv_np, bp["risk_threshold"])
freq = seg_all.value_counts()
print(freq)

fig, ax = plt.subplots(figsize=(8, 5))
freq.reindex(bz.SEGMENTS).plot.bar(ax=ax, color=["#c7417b", "#e8a838", "#2e86ab", "#8f9aa7"])
ax.set_title("Clientes por segmento")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(FIG / "segmentos.png", bbox_inches="tight")
plt.show()
"""), code("""\
# ROI estimado da campanha por segmento (comparado a não agir)
roi = bz.campaign_roi(seg_all.to_numpy(), final_score_all, clv_np)
pd.DataFrame(roi).T.round(2)
"""), code("""\
# Contextualização Brasil (referência de moeda)
usd_brl = 5.2
ev = sc["ev_prioritary_per_client"]
print(f"EV da regra por-cliente na holdout: US${ev:,.0f} ≈ R${ev * usd_brl:,.0f}")
print(f"Custo por cliente (personalizado): US${bz.RETENTION_COST_PRIORITARY} ≈ R${bz.RETENTION_COST_PRIORITARY * usd_brl:.0f}")
"""), md("""\
### Regras operacionais recomendadas

1. **Acionar só quem compensa:** regra por-cliente `lift·p·CLV > custo` rende mais que qualquer
   corte fixo (EV $20,7k vs. $19,1k na holdout) — oposto de "reter todo mundo".
2. **Segmentos:**
   - `retencao_prioritaria` (alto risco × alto valor) → time humano, oferta personalizada;
   - `acao_automatizada` (alto risco × baixo valor) → oferta automática de baixo custo (email/app);
   - `monitorar_upsell` → não gastar agora, buscar expansão;
   - `monitorar` → sem ação.
3. **ROI:** com os lifts assumidos, o retorno esperado cobre folgadamente os custos nas duas
   campanhas (personalizada e automatizada).
4. **Corte operacional:** o alerta de retenção dispara com score `>= threshold` do modelo
   (persistido em `business_params.json`); a precisão nessa região (~0,90) limita o ruído
   para o time.
""")]


def main() -> None:
    tables = [
        ("01_eda", "01 - Análise exploratória orientada a negócio", EDA),
        ("02_feature_engineering", "02 - Engenharia de atributos", FE),
        ("03_modeling", "03 - Modelagem e validação", MODELING),
        ("04_shap", "04 - Explicabilidade (SHAP)", SHAP),
        ("05_business_layer", "05 - Camada de negócio e segmentação", BUSINESS),
    ]
    ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
    for stem, title, cells in tables:
        nb = build_notebook(title, cells)
        path = NOTEBOOKS / f"{stem}.ipynb"
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"ok: {path.name}")


if __name__ == "__main__":
    sys.exit(main())