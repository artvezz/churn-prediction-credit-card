"""Engenharia de atributos canonica do projeto de churn.

Fonte unica de verdade: o mesmo modulo e usado nos notebooks de treino e no
servico de scoring (Docker). Qualquer mudanca aqui deve ser refletida em ambos
os ambientes (o Dockerfile copia este arquivo para dentro da imagem).

Convencoes:
- Entrada: DataFrame com as colunas brutas do BankChurners (credit card customers).
- Saida: DataFrame com as colunas de features finais (numericas + dummies), na
  ordem deterministica registrada em `FEATURE_COLUMNS`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

# Colunas brutas que vazam o target (geradas por um modelo naive bayes no
# proprio arquivo) e o identificador do cliente. Tem de ser descartadas sempre.
LEAK_COLUMNS = [
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

ID_COLUMN = "CLIENTNUM"
TARGET_COLUMN = "Attrition_Flag"
TARGET_POSITIVE = "Attrited Customer"

CATEGORICAL_COLUMNS = ["Gender", "Marital_Status", "Card_Category"]

# Renda: midpoint de cada faixa (em USD/ano). "Unknown" vira NaN (imputado).
# O dataset (e a tabela `clientes`) usa a rotulacao "Less than $40K"; a chave
# "<$40K" e mantida para tolerar a variacao de rotulagem entre fontes.
INCOME_MIDPOINT = {
    "Less than $40K": 20_000.0,
    "<$40K": 20_000.0,
    "$40K - $60K": 50_000.0,
    "$60K - $80K": 70_000.0,
    "$80K - $120K": 100_000.0,
    "$120K +": 120_000.0,
    "Unknown": np.nan,
}

# Escolaridade: ordem ordinal.
EDUCATION_ORDINAL = {
    "Uneducated": 0,
    "High School": 1,
    "College": 2,
    "Graduate": 3,
    "Post-Graduate": 4,
    "Doctorate": 5,
    "Unknown": np.nan,
}

# Anuidade media por categoria de cartao (USD/ano) - usada na camada de negocio.
CARD_ANNUAL_FEE = {"Blue": 50.0, "Silver": 80.0, "Gold": 120.0, "Platinum": 250.0}

# Colunas numericas brutas usadas diretamente como features.
RAW_NUMERIC = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

# Features mapeadas (renda -> midpoint, escolaridade -> ordinal).
MAPPED = ["monthly_income", "education_ordinal"]

# Features derivadas (engenharia de atributos).
ENGINEERED = [
    "revolving_balance_ratio",     # saldo rotativo / limite
    "avg_ticket",                  # valor medio por transacao
    "trans_amt_per_contact",       # transacao anual por contato com SAC
    "inactive_contacts",           # interacao: inatividade x contatos
    "limit_per_product",           # limite por produto contratado
    "trans_decay_score",           # queda de atividade ponderada pelo volume
]


def _clean_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Remove espacos das categorias textuais (o dataset vem com trailing spaces)."""
    out = df.copy()
    for col in CATEGORICAL_COLUMNS + [
        TARGET_COLUMN,
        "Education_Level",
        "Marital_Status",
        "Income_Category",
    ]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    return out


def _add_mapped_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["monthly_income"] = out["Income_Category"].map(INCOME_MIDPOINT)
    out["education_ordinal"] = out["Education_Level"].map(EDUCATION_ORDINAL)
    return out


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features derivadas de combinacoes das colunas brutas. Todas com o
    denominador protegido contra divisao por zero."""
    out = df.copy()
    limit = out["Credit_Limit"]
    trans_ct = out["Total_Trans_Ct"]
    contacts = out["Contacts_Count_12_mon"]
    inactive = out["Months_Inactive_12_mon"]
    n_products = out["Total_Relationship_Count"]

    out["revolving_balance_ratio"] = np.where(limit > 0, out["Total_Revolving_Bal"] / limit, 0.0)
    out["avg_ticket"] = np.where(trans_ct > 0, out["Total_Trans_Amt"] / trans_ct, 0.0)
    out["trans_amt_per_contact"] = np.where(
        contacts + 1 > 0, out["Total_Trans_Amt"] / (contacts + 1), out["Total_Trans_Amt"]
    )
    out["inactive_contacts"] = inactive * (contacts + 1)
    out["limit_per_product"] = np.where(n_products + 1 > 0, limit / (n_products + 1), limit)
    out["trans_decay_score"] = out["Total_Amt_Chng_Q4_Q1"] * out["Total_Trans_Amt"]
    return out


def _dummies_with_order(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """One-hot encoding com ordem deterministica das categorias."""
    dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
    dummies = dummies.reindex(sorted(dummies.columns), axis=1)
    return dummies


def build_features(df: pd.DataFrame, drop_columns: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Transforma o DataFrame bruto em matriz de features pronta para o modelo.

    NOTA: a matriz pode conter NaNs (categorias "Unknown" mapeadas para nada).
    A imputacao acontece DEPOIS da divisao treino/teste (medianas calculadas
    apenas no treino), via `impute_medians`.

    Returns
    -------
    X : DataFrame com as features finais ordenadas (possivelmente com NaN).
    feature_columns : lista ordenada dos nomes das colunas (persistida para
        garantir a mesma ordem entre treino e servico de scoring).
    """
    df = df.drop(columns=[c for c in LEAK_COLUMNS if c in df.columns])
    df = df.drop(columns=[c for c in [TARGET_COLUMN, ID_COLUMN] if drop_columns and c in df.columns])

    df = _clean_categorical(df)
    df = _add_mapped_features(df)
    df = _add_engineered_features(df)

    # Ordem deterministica: numericas (alfabeticas) + dummies (alfabeticas).
    numeric_blocks = [RAW_NUMERIC, MAPPED, ENGINEERED]
    numeric_cols = list(dict.fromkeys(c for block in numeric_blocks for c in block if c in df.columns))
    numeric_cols.sort()

    dummy_frames = [_dummies_with_order(df, col) for col in CATEGORICAL_COLUMNS if col in df.columns]
    dummies = pd.concat(dummy_frames, axis=1) if dummy_frames else pd.DataFrame(index=df.index)

    X = pd.concat([df[numeric_cols], dummies], axis=1)
    X = X.astype(float)
    return X, list(X.columns)


def median_imputation_values(X: pd.DataFrame) -> dict[str, float]:
    """Medianas de imputacao calculadas sobre um subconjunto (normalmente o treino)."""
    return {c: float(X[c].median()) for c in X.columns}


def impute_medians(X: pd.DataFrame, values: dict[str, float]) -> pd.DataFrame:
    """Aplica as medianas salvas no treino. Barrasu colunas ausentes."""
    missing = [c for c in values if c not in X.columns]
    if missing:
        raise ValueError(f"Colunas ausentes para imputacao: {missing}")
    X = X.copy()
    for c, v in values.items():
        if X[c].isna().any():
            X[c] = X[c].fillna(v)
    return X


def save_feature_columns(feature_columns: list[str], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)


def load_feature_columns(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def reconcile_columns(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Garante que X tenha exatamente as colunas de feature_columns, na mesma ordem.

    Usado no servico de scoring (valida contra o que foi treinado) para evitar
    sobrevivencia silenciosa (silent column mismatch) com categorias novas.
    """
    missing = [c for c in feature_columns if c not in X.columns]
    extra = [c for c in X.columns if c not in feature_columns]
    if missing:
        raise ValueError(f"Colunas ausentes vs treino: {missing}")
    if extra:
        raise ValueError(f"Colunas inesperadas vs treino: {extra}")
    return X[feature_columns]