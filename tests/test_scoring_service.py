"""Testes do servico de scoring: mapeamento do Supabase, contrato de saida e API.

O ponto central e garantir que o caminho de serving reproduz o treino (fonte unica
de features) e que a leitura do Supabase entrega as colunas que o modelo espera.
"""

from __future__ import annotations

import importlib
import os
import re

import numpy as np
import pandas as pd
import pytest

from conftest import ARTIFACTS, DEMO_SCORES, ROOT, sem_treino

pytestmark = sem_treino


@pytest.fixture(scope="module")
def svc():
    import app as modulo

    return modulo


# --- Mapeamento snake_case (Supabase) -> camelCase (dataset) ---

def test_to_raw_columns_renomeia_de_fato(svc):
    """Regressao: o mapa era invertido, o rename virava no-op e `build_features`
    quebrava com KeyError em `Income_Category` no modo Supabase."""
    entrada = pd.DataFrame(
        [
            {
                "id_cliente": 768805383,
                "income_category": "Less than $40K",
                "card_category": "Gold",
                "attrition_flag": "Existing Customer",
                "data_carga": "2026-01-01",
            }
        ]
    )
    saida = svc._to_raw_columns(entrada)

    assert "Income_Category" in saida.columns
    assert "CLIENTNUM" in saida.columns
    assert "Attrition_Flag" in saida.columns
    assert "income_category" not in saida.columns


def test_to_raw_columns_descarta_coluna_sem_equivalente(svc):
    saida = svc._to_raw_columns(pd.DataFrame([{"id_cliente": 1, "data_carga": "2026-01-01"}]))
    assert list(saida.columns) == ["CLIENTNUM"]


def test_to_raw_columns_preserva_numeros_e_valores(svc):
    entrada = pd.DataFrame([{"id_cliente": 7, "total_trans_amt": 1234.56, "credit_limit": 5000.0}])
    saida = svc._to_raw_columns(entrada)
    assert float(saida["Total_Trans_Amt"].iloc[0]) == 1234.56
    assert float(saida["Credit_Limit"].iloc[0]) == 5000.0
    assert int(saida["CLIENTNUM"].iloc[0]) == 7


def test_mapa_cobre_todas_as_colunas_do_schema(svc):
    """Trava de contrato: se alguem add coluna no schema.sql sem mapear aqui,
    o teste falha em vez de a coluna silenciosamente sumir no serving."""
    sql = (ROOT / "supabase" / "schema.sql").read_text(encoding="utf-8")
    bloco = re.search(r"create table if not exists public\.clientes \((.*?)\n\);", sql, re.S)
    assert bloco, "nao encontrei a tabela public.clientes no schema.sql"

    colunas = re.findall(r"^\s{4}(\w+)\s+(?:bigint|integer|text|numeric|timestamptz)", bloco.group(1), re.M)
    assert "id_cliente" in colunas

    sem_mapa = [c for c in colunas if c not in svc.CLIENT_SNAKE_TO_RAW and c != "data_carga"]
    assert not sem_mapa, f"colunas do schema sem mapeamento no servico: {sem_mapa}"


def test_mapa_nao_tem_colunas_duplicadas(svc):
    assert len(set(svc.CLIENT_SNAKE_TO_RAW.values())) == len(svc.CLIENT_SNAKE_TO_RAW)


# --- Contrato de saida do score_frame ---

def test_score_frame_contrato_de_colunas(svc, raw):
    saida = svc.score_frame(raw.head(50))
    assert list(saida.columns) == [
        "id_cliente",
        "score_churn",
        "clv_usd",
        "segmento",
        "acao",
    ]
    assert len(saida) == 50
    assert saida["score_churn"].between(0.0, 1.0).all()
    assert (saida["clv_usd"] > 0).all()
    assert set(saida["segmento"].unique()).issubset(set(svc.business.SEGMENTS))
    assert not saida.isna().any().any()


def test_score_frame_ids_preservados(svc, raw):
    saida = svc.score_frame(raw.head(20))
    assert list(saida["id_cliente"]) == list(raw["CLIENTNUM"].head(20))


def test_score_frame_deterministico(svc, raw):
    a = svc.score_frame(raw.head(50))
    b = svc.score_frame(raw.head(50))
    pd.testing.assert_frame_equal(a, b)


def test_score_frame_bate_com_a_linha_do_demo_do_treino(svc, raw):
    """Funcao unica de features: o batch de serving tem que reproduzir os scores
    que o treino salvou em data/processed/churn_scores_demo.csv."""
    if not DEMO_SCORES.exists():
        pytest.skip("demo de scoring ausente: rode `python src/train.py`")

    demo = pd.read_csv(DEMO_SCORES, encoding="utf-8")
    esperado = demo.sort_values("id_cliente").reset_index(drop=True)
    obtido = svc.score_frame(raw).sort_values("id_cliente").reset_index(drop=True)

    assert len(obtido) == len(esperado)
    np.testing.assert_allclose(obtido["score_churn"], esperado["score_churn"], atol=1e-4)
    np.testing.assert_allclose(obtido["clv_usd"], esperado["clv_usd"], atol=1e-2)
    assert list(obtido["segmento"]) == list(esperado["segmento"])


def test_threshold_do_servico_vem_do_treino(svc, business_params):
    assert svc.bundle.risk_threshold == pytest.approx(business_params["risk_threshold"])
    assert len(svc.bundle.feature_columns) == 32
    assert svc.bundle.medians


# --- API HTTP (modo local, sem Supabase) ---

@pytest.fixture
def api_client(tmp_path, raw):
    csv = tmp_path / "clientes.csv"
    raw.head(200).to_csv(csv, index=False)

    os.environ["CHURN_SOURCE"] = "local"
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["CLIENTES_CSV"] = str(csv)
    os.environ["RUN_API_KEY"] = "segredo-de-teste"
    os.environ.pop("SUPABASE_URL", None)
    os.environ.pop("SUPABASE_SERVICE_KEY", None)

    modulo = importlib.import_module("app")
    importlib.reload(modulo)

    from fastapi.testclient import TestClient

    with TestClient(modulo.app) as client:
        yield client, modulo, tmp_path

    os.environ.pop("CHURN_SOURCE", None)
    os.environ.pop("DATA_DIR", None)
    os.environ.pop("CLIENTES_CSV", None)
    os.environ.pop("RUN_API_KEY", None)
    importlib.reload(modulo)


def test_health_nao_exige_chave(api_client):
    client, _, _ = api_client
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["fonte"] == "local"
    assert r.json()["modelo_carregado"] is True


def test_run_scoring_exige_x_api_key(api_client):
    client, _, _ = api_client
    assert client.post("/run-scoring").status_code == 401
    assert client.post("/run-scoring", headers={"X-API-Key": "errada"}).status_code == 401


def test_run_scoring_gera_lote_e_resumo(api_client):
    client, modulo, tmp_path = api_client
    r = client.post("/run-scoring", headers={"X-API-Key": "segredo-de-teste"})

    assert r.status_code == 200
    corpo = r.json()
    assert corpo["ok"] is True
    assert corpo["total_clientes"] == 200
    assert set(corpo["n_por_segmento"]) == set(modulo.business.SEGMENTS)
    assert 0.0 <= corpo["score_medio"] <= 1.0
    assert "roi_campanha" in corpo

    destino = tmp_path / "processed" / "churn_scores_local.csv"
    assert destino.exists()
    salvo = pd.read_csv(destino, encoding="utf-8")
    assert len(salvo) == 200
    assert {"id_cliente", "score_churn", "clv_usd", "segmento", "acao", "data_execucao"} == set(salvo.columns)


def test_run_scoring_rejeita_csv_inexistente(api_client):
    client, modulo, tmp_path = api_client
    modulo.CLIENTES_CSV = str(tmp_path / "nao-existe.csv")
    try:
        r = client.post("/run-scoring", headers={"X-API-Key": "segredo-de-teste"})
        assert r.status_code == 404
    finally:
        modulo.CLIENTES_CSV = str(tmp_path / "clientes.csv")


def test_bundle_carrega_os_artefatos_do_treino(svc):
    assert (ARTIFACTS / "churn_model.pkl").exists()
    assert svc.bundle.model is not None
