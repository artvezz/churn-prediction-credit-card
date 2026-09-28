"""Testes da engenharia de atributos (fonte unica treino <-> servidor)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import features as fe
from conftest import sem_treino


# --- Contrato de colunas (train/serve skew) ---

@sem_treino
def test_build_features_bate_com_colunas_do_treino(raw, feature_columns):
    X, cols = fe.build_features(raw)
    assert cols == feature_columns
    assert list(X.columns) == feature_columns
    assert X.shape[0] == raw.shape[0]


@sem_treino
def test_colunas_de_leak_do_target_sao_descartadas(raw):
    assert all(c in raw.columns for c in fe.LEAK_COLUMNS)
    X, cols = fe.build_features(raw)
    assert not [c for c in cols if c.startswith("Naive_Bayes")]
    assert fe.TARGET_COLUMN not in cols
    assert fe.ID_COLUMN not in cols


@sem_treino
def test_reconcile_columns_rejeita_coluna_faltante_e_extra(raw, feature_columns):
    X, _ = fe.build_features(raw)
    with pytest.raises(ValueError, match="Colunas ausentes"):
        fe.reconcile_columns(X, feature_columns + ["coluna_que_nao_existe"])
    with pytest.raises(ValueError, match="inesperadas"):
        extra = X.copy()
        extra["coluna_que_nao_existe"] = 0.0
        fe.reconcile_columns(extra, feature_columns)


def test_reconcile_columns_preserva_a_ordem_do_treino():
    X = pd.DataFrame({"b": [1.0], "a": [2.0], "c": [3.0]})
    out = fe.reconcile_columns(X, ["a", "b", "c"])
    assert list(out.columns) == ["a", "b", "c"]


# --- Rotulos categoricos: nenhum cliente pode ficar sem feature por rotulo errado ---

def test_income_midpoint_cobre_todos_os_rotulos_do_dataset(raw):
    """Regressao: o dataset usa "Less than $40K"; a chave antiga "<$40K" nunca
    casava e 35% da base ficava com renda imputada pela mediana."""
    income = raw["Income_Category"].astype(str).str.strip()
    faltando = sorted({v for v in income.unique() if v != "Unknown" and pd.isna(fe.INCOME_MIDPOINT.get(v))})
    assert not faltando, f"rotulos de renda sem midpoint em INCOME_MIDPOINT: {faltando}"


def test_income_midpoint_aceita_as_duas_rotulagens():
    assert fe.INCOME_MIDPOINT["Less than $40K"] == fe.INCOME_MIDPOINT["<$40K"] == 20_000.0
    assert np.isnan(fe.INCOME_MIDPOINT["Unknown"])


def test_education_ordinal_cobre_todos_os_rotulos_do_dataset(raw):
    edu = raw["Education_Level"].astype(str).str.strip()
    faltando = sorted({v for v in edu.unique() if v != "Unknown" and pd.isna(fe.EDUCATION_ORDINAL.get(v))})
    assert not faltando, f"rotulos de escolaridade sem ordinal: {faltando}"


@sem_treino
def test_renda_so_fica_nan_para_unknown(raw, medians):
    X, _ = fe.build_features(raw)
    income = raw["Income_Category"].astype(str).str.strip()
    n_nan = int(X["monthly_income"].isna().sum())
    assert n_nan == int((income == "Unknown").sum())
    assert medians["monthly_income"] > 0


# --- Imputacao ---

@sem_treino
def test_imputacao_nao_deixa_nan(raw, feature_columns, medians):
    X, _ = fe.build_features(raw)
    X = fe.reconcile_columns(X, feature_columns)
    X = fe.impute_medians(X, medians)
    assert int(X.isna().sum().sum()) == 0


def test_impute_medians_falha_se_faltar_coluna_de_treino():
    X = pd.DataFrame({"a": [1.0, np.nan]})
    with pytest.raises(ValueError, match="Colunas ausentes"):
        fe.impute_medians(X, {"a": 0.0, "b": 0.0})


# --- Features derivadas ---

def _linha_base() -> dict:
    return {
        "Customer_Age": 45,
        "Dependent_count": 2,
        "Months_on_book": 40,
        "Total_Relationship_Count": 4,
        "Months_Inactive_12_mon": 3,
        "Contacts_Count_12_mon": 2,
        "Credit_Limit": 2000.0,
        "Total_Revolving_Bal": 1000.0,
        "Avg_Open_To_Buy": 1000.0,
        "Total_Amt_Chng_Q4_Q1": 0.5,
        "Total_Trans_Amt": 5000.0,
        "Total_Trans_Ct": 100,
        "Total_Ct_Chng_Q4_Q1": 0.2,
        "Avg_Utilization_Ratio": 0.5,
        "Income_Category": "$60K - $80K",
        "Education_Level": "Graduate",
        "Gender": "F",
        "Marital_Status": "Single",
        "Card_Category": "Gold",
    }


def test_features_derivadas_seguem_as_formulas():
    row = _linha_base()
    X, _ = fe.build_features(pd.DataFrame([row]))
    r = X.iloc[0]
    assert r["revolving_balance_ratio"] == pytest.approx(1000.0 / 2000.0)
    assert r["avg_ticket"] == pytest.approx(50.0)
    assert r["trans_amt_per_contact"] == pytest.approx(5000.0 / 3)
    assert r["inactive_contacts"] == pytest.approx(3 * 3)
    assert r["limit_per_product"] == pytest.approx(2000.0 / 5)
    assert r["trans_decay_score"] == pytest.approx(0.5 * 5000.0)
    assert r["monthly_income"] == pytest.approx(70_000.0)
    assert r["education_ordinal"] == 3.0


def test_espacos_nas_categorias_sao_removidos():
    row = _linha_base() | {"Card_Category": " Gold ", "Gender": " M "}
    X, _ = fe.build_features(pd.DataFrame([row]))
    assert X.columns.get_loc("Card_Category_Gold") >= 0
    assert "Card_Category_ Gold" not in X.columns
    assert "Gender_M" in X.columns


def test_divisao_por_zero_nao_gera_nan_nem_inf():
    row = _linha_base() | {
        "Credit_Limit": 0.0,
        "Total_Revolving_Bal": 0.0,
        "Total_Trans_Ct": 0,
        "Total_Trans_Amt": 0.0,
        "Contacts_Count_12_mon": 0,
    }
    X, _ = fe.build_features(pd.DataFrame([row]))
    r = X.iloc[0]
    for col in ("revolving_balance_ratio", "avg_ticket", "trans_amt_per_contact", "limit_per_product"):
        assert np.isfinite(r[col]), f"{col} nao finito: {r[col]}"
    assert r["revolving_balance_ratio"] == 0.0
    assert r["avg_ticket"] == 0.0


def test_engenharia_e_deterministica():
    df = pd.DataFrame([_linha_base(), _linha_base() | {"Customer_Age": 50}])
    a, cols_a = fe.build_features(df)
    b, cols_b = fe.build_features(df)
    assert cols_a == cols_b
    pd.testing.assert_frame_equal(a, b)


def test_dummies_tem_ordem_deterministica():
    df = pd.DataFrame([_linha_base() | {"Card_Category": c} for c in ["Silver", "Blue", "Gold"]])
    X, _ = fe.build_features(df)
    cards = [c for c in X.columns if c.startswith("Card_Category_")]
    assert cards == sorted(cards)


def test_lote_parcial_ainda_produz_todas_as_dummies_do_treino(raw, feature_columns):
    """Regressao: sem o universo fixo de categorias, um lote sem cartao Platinum
    (ou sem cliente de algum genero) derrubava o scoring com "Colunas ausentes"."""
    sem_platinum = raw[raw["Card_Category"].str.strip() != "Platinum"].head(50)
    assert "Platinum" not in set(sem_platinum["Card_Category"].str.strip())

    X, cols = fe.build_features(sem_platinum)
    assert cols == feature_columns
    X = fe.reconcile_columns(X, feature_columns)
    assert X["Card_Category_Platinum"].sum() == 0


def test_categoria_inedita_e_rejeitada_pelo_reconcile(raw, feature_columns):
    """Uma categoria que o modelo nunca viu precisa falhar, nao ser ignorada."""
    df = pd.DataFrame([_linha_base() | {"Card_Category": "Corporate"}])
    X, _ = fe.build_features(df)
    assert "Card_Category_Corporate" in X.columns
    with pytest.raises(ValueError, match="inesperadas"):
        fe.reconcile_columns(X, feature_columns)
