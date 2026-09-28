"""Testes da explicabilidade local (SHAP) e do laudo gerado no lote."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import explain
from conftest import ARTIFACTS, sem_treino


def _modelo():
    import joblib

    return joblib.load(ARTIFACTS / "churn_model.pkl")


def _x_pronto(raw, colunas, medianas, n: int = 30):
    import features as fe

    X, _ = fe.build_features(raw.head(n))
    return fe.impute_medians(fe.reconcile_columns(X, colunas), medianas)


@sem_treino
def test_laudo_tem_o_formato_esperado_pelo_dashboard(raw, feature_columns, medians):
    X = _x_pronto(raw, feature_columns, medians, 40)
    drivers = explain.shap_drivers(_modelo(), X, raw.head(40)["CLIENTNUM"])

    assert len(drivers) == 40
    for chave, d in drivers.items():
        assert chave.isdigit(), f"chave de cliente invalida: {chave!r}"
        assert set(d) == {"pressiona_churn", "protege_churn"}
        for grupo in d.values():
            assert 1 <= len(grupo) <= 3
            for item in grupo:
                assert set(item) == {"feature", "values"}
                assert item["feature"] in feature_columns
                assert isinstance(item["values"], float)


@sem_treino
def test_pressiona_tem_contribuicao_positiva_e_protege_negativa(raw, feature_columns, medians):
    X = _x_pronto(raw, feature_columns, medians, 20)
    drivers = explain.shap_drivers(_modelo(), X, raw.head(20)["CLIENTNUM"])

    for d in drivers.values():
        assert all(x["values"] >= 0 for x in d["pressiona_churn"])
        assert all(x["values"] < 0 for x in d["protege_churn"])


@sem_treino
def test_laudo_ordena_por_impacto_absoluto(raw, feature_columns, medians):
    X = _x_pronto(raw, feature_columns, medians, 20)
    drivers = explain.shap_drivers(_modelo(), X, raw.head(20)["CLIENTNUM"])

    for d in drivers.values():
        for grupo in ("pressiona_churn", "protege_churn"):
            impactos = [abs(x["values"]) for x in d[grupo]]
            assert impactos == sorted(impactos, reverse=True)


@sem_treino
def test_laudo_e_deterministico(raw, feature_columns, medians):
    X = _x_pronto(raw, feature_columns, medians, 20)
    ids = raw.head(20)["CLIENTNUM"]

    assert explain.shap_drivers(_modelo(), X, ids) == explain.shap_drivers(_modelo(), X, ids)


def test_shap_dependencia_esta_disponivel():
    pytest.importorskip("shap", reason="explicabilidade exige shap")


@sem_treino
def test_lote_grava_o_laudo_com_metadados(tmp_path, raw, monkeypatch):
    """O lote precisa produzir um laudo novo (datado da rodada) para o dashboard."""
    import app as modulo

    X = _x_pronto(raw, modulo.bundle.feature_columns, modulo.bundle.medians, 30)
    monkeypatch.setattr(modulo, "DATA_DIR", tmp_path)

    n = modulo.explain_frame(X, raw.head(30)["CLIENTNUM"], "2026-09-28 10:00:00")

    destino = tmp_path / "processed" / "shap_by_client.json"
    assert destino.exists()
    assert n == 30

    dados = json.loads(destino.read_text(encoding="utf-8"))
    assert dados["_meta"] == {"data_execucao": "2026-09-28 10:00:00", "n_clientes": 30}
    dados.pop("_meta")
    assert len(dados) == 30
    assert all(k.isdigit() for k in dados)


@sem_treino
def test_falha_no_shap_nao_derruba_o_lote(tmp_path, raw, monkeypatch):
    import app as modulo

    monkeypatch.setattr(modulo, "DATA_DIR", tmp_path)

    def _explode(*_args, **_kwargs):
        raise RuntimeError("sem memoria")

    monkeypatch.setattr(modulo.explain, "shap_drivers", _explode)
    assert modulo.explain_frame(pd.DataFrame(), raw.head(5)["CLIENTNUM"], "x") == 0
    assert not (tmp_path / "processed" / "shap_by_client.json").exists()
