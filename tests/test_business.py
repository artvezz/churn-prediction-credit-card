"""Testes da camada de negocio: CLV, regra de acao e segmentacao."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import business as bz


# --- CLV ---

def test_annual_profit_soma_margem_juros_e_anuidade():
    t = pd.Series([10_000.0])
    r = pd.Series([1_000.0])
    c = pd.Series(["Gold"])
    lucro = bz.annual_profit(t, r, c)
    esperado = 0.04 * 10_000.0 + 0.12 * 1_000.0 + 120.0
    assert float(lucro.iloc[0]) == pytest.approx(esperado)


def test_categoria_de_cartao_desconhecida_usa_anuidade_padrao():
    lucro = bz.annual_profit(pd.Series([0.0]), pd.Series([0.0]), pd.Series(["inexistente"]))
    assert float(lucro.iloc[0]) == pytest.approx(50.0)


def test_fator_de_anuidade_5_anos_a_10pct():
    assert bz.annuity_factor(0.10, 5) == pytest.approx(3.790786769, rel=1e-6)


def test_clv_escala_com_a_margem_sobre_transacoes():
    """CLV = (margem*transacoes + juros*saldo + anuidade) x fator de anuidade;
    dobrar as transacoes dobra so a parte da margem (a anuidade e fixa)."""
    c = pd.Series(["Blue", "Blue"])
    r = pd.Series([0.0, 0.0])
    t = pd.Series([5_000.0, 10_000.0])
    clv = bz.clv(t, r, c)
    fator = bz.annuity_factor(bz.DISCOUNT_RATE, bz.CLV_HORIZON_YEARS)

    assert float(clv.iloc[1] - clv.iloc[0]) == pytest.approx(bz.TRANSACTION_MARGIN * 5_000.0 * fator)
    assert (clv > 0).all()


def test_clv_cresce_com_saldo_rotativo():
    c = pd.Series(["Blue", "Blue"])
    t = pd.Series([0.0, 0.0])
    r = pd.Series([1_000.0, 2_000.0])
    assert float(bz.clv(t, r, c).iloc[1]) > float(bz.clv(t, r, c).iloc[0])


# --- Regra de acao (lift * p * CLV > custo) ---

def test_per_client_decision_exige_ev_estritamente_positivo():
    # lift*p*CLV == custo exatamente -> nao aciona (regra estritamente maior).
    custo, lift = 80.0, 0.15
    clv = 1_000.0
    score = np.array([custo / (lift * clv), (custo / (lift * clv)) * 1.01])
    acionar = bz.per_client_decision(score, np.array([clv, clv]), custo, lift)
    assert list(acionar) == [False, True]


def test_regra_por_cliente_dominina_o_melhor_threshold_global():
    """Invariante do relatorio: escolher cliente a cliente maximiza o EV total,
    logo nao existe threshold global que a supere."""
    rng = np.random.default_rng(42)
    score = rng.beta(2, 8, size=4000)
    clv = rng.lognormal(6.0, 0.6, size=4000)
    custo, lift = 80.0, 0.15

    ev_global, _ = bz.expected_value_matrix(score, clv, custo, lift)
    ev_global = bz.optimal_threshold(score, clv, custo, lift)[1]
    mask = bz.per_client_decision(score, clv, custo, lift)
    ev_cliente = float((lift * score[mask] * clv[mask] - custo).sum())

    assert ev_cliente >= ev_global
    assert ev_cliente > 0


def test_nao_fazer_nada_tem_ev_zero():
    ev, custo = bz.expected_value_matrix(np.array([]), np.array([]), 80.0, 0.15)
    assert ev == 0.0 and custo == 0.0


def test_expected_value_on_threshold_ignora_abaixo_do_corte():
    score = np.array([0.9, 0.1])
    clv = np.array([1_000.0, 1_000.0])
    ev = bz.expected_value_on_threshold(score, clv, 80.0, 0.15, 0.5)
    assert ev == pytest.approx(0.15 * 0.9 * 1_000.0 - 80.0)


def test_optimal_threshold_aciona_menos_clientes_que_acionar_todos():
    rng = np.random.default_rng(7)
    score = rng.uniform(0, 1, 2000)
    clv = np.full(2000, 5_000.0)  # acima de custo/lift = 533,33 -> ha clientes acionaveis
    threshold, ev = bz.optimal_threshold(score, clv, 80.0, 0.15)
    assert 0.0 <= threshold <= 1.0
    assert ev > 0
    assert 0 < int((score >= threshold).sum()) < 2000


def test_ninguem_e_acionado_quando_o_clv_nao_cobre_o_custo():
    score = np.linspace(0, 1, 100)
    clv = np.full(100, 100.0)  # 0,15 * 1,0 * 100 = 15 < 80 -> EV sempre negativo
    threshold, ev = bz.optimal_threshold(score, clv, 80.0, 0.15)
    assert ev <= 0
    assert not bz.per_client_decision(score, clv, 80.0, 0.15).any()


# --- Segmentacao risco x valor ---

def test_segment_customers_preenche_os_quatro_quadrantes():
    clv = np.array([1_000.0, 1_000.0, 100.0, 100.0])
    score = np.array([0.9, 0.1, 0.9, 0.1])
    seg = bz.segment_customers(score, clv, risk_threshold=0.5, clv_quantile=0.5)
    assert list(seg) == [
        "retencao_prioritaria",
        "monitorar_upsell",
        "acao_automatizada",
        "monitorar",
    ]


def test_segment_customers_usa_apenas_os_segmentos_declarados():
    rng = np.random.default_rng(3)
    seg = bz.segment_customers(rng.uniform(0, 1, 500), rng.lognormal(6, 1, 500), 0.5, 0.5)
    assert set(seg.unique()).issubset(set(bz.SEGMENTS))
    assert len(seg) == 500


def test_todo_segmento_tem_acao_e_custo_declarados():
    assert set(bz.ACTION_LABEL) == set(bz.SEGMENTS)
    assert set(bz.SEGMENT_COST) == set(bz.SEGMENTS)
    assert set(bz.SEGMENT_LIFT) == set(bz.SEGMENTS)


# --- ROI da campanha ---

def test_campaign_roi_usa_custo_e_lift_do_segmento():
    seg = np.array(["retencao_prioritaria", "retencao_prioritaria", "monitorar"])
    score = np.array([0.9, 0.8, 0.1])
    clv = np.array([1_000.0, 1_000.0, 1_000.0])
    roi = bz.campaign_roi(seg, score, clv)

    assert roi["retencao_prioritaria"]["n_clientes"] == 2
    assert roi["retencao_prioritaria"]["custo_total"] == pytest.approx(160.0)
    esperado = 0.15 * (0.9 + 0.8) * 1_000.0
    assert roi["retencao_prioritaria"]["retorno_esperado"] == pytest.approx(esperado, rel=1e-3)
    assert roi["retencao_prioritaria"]["roi"] == pytest.approx((esperado - 160.0) / 160.0, rel=1e-3)


def test_campaign_roi_trata_segmento_vazio_sem_erro():
    roi = bz.campaign_roi(np.array(["monitorar"]), np.array([0.1]), np.array([500.0]))
    assert roi["retencao_prioritaria"] == {
        "n_clientes": 0,
        "custo_total": 0.0,
        "retorno_esperado": 0.0,
        "roi": 0.0,
    }


# --- Persistencia dos parametros ---

def test_save_and_load_business_params_roundtrip(tmp_path, business_params):
    path = tmp_path / "params.json"
    bz.save_business_params(business_params, str(path))
    assert bz.load_business_params(str(path))["risk_threshold"] == business_params["risk_threshold"]


def test_parametros_de_negocio_estao_dentro_do_esperado(business_params):
    assert 0.0 < business_params["risk_threshold"] < 1.0
    assert business_params["fee_by_card"] == bz.CARD_ANNUAL_FEE
    assert business_params["clv_horizon_years"] == bz.CLV_HORIZON_YEARS
    assert business_params["discount_rate"] == bz.DISCOUNT_RATE
    for chave in ("trans_margin", "revolving_apr", "retention_cost_prioritary", "lift_prioritary"):
        assert business_params[chave] > 0
