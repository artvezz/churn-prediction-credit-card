"""Camada de negocio: CLV, matriz custo/beneficio, threshold otimo e segmentacao.

Toda constante financeira aqui (margem, anuidade, custo de campanha, taxa de
desconto) e uma presuncao explicita e configuravel. As premissas estao
documentadas no README e no relatorio executivo.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from features import CARD_ANNUAL_FEE

# --- Premissas financeiras (USD) ---

# Margem anual sobre transacoes (interchange + juros nao-rotativos) - conservadora.
TRANSACTION_MARGIN = 0.04

# Juros medios anuais sobre o saldo rotativo (APR efetivo de interesse).
REVOLVING_APR = 0.12

# Custo medio da campanha de retencao por cliente.
RETENTION_COST_PRIORITARY = 80.0   # contato personalizado (time humano)
RETENTION_COST_AUTOMATED = 5.0     # oferta automatizada (e-mail)

# Lift medio de retencao do tipo de acao (fracao que efetivamente fica).
LIFT_PRIORITARY = 0.15  # contato/offerta construida pelo time
LIFT_AUTOMATED = 0.05   # oferta automatizada generica

# Horizonte e taxa de desconto do CLV.
CLV_HORIZON_YEARS = 5
DISCOUNT_RATE = 0.10

SEGMENTS = ["retencao_prioritaria", "acao_automatizada", "monitorar_upsell", "monitorar"]
ACTION_LABEL = {
    "retencao_prioritaria": "Retencao prioritaria (contato personalizado)",
    "acao_automatizada": "Acao automatizada de baixo custo",
    "monitorar_upsell": "Monitorar e explorar upsell",
    "monitorar": "Monitorar",
}

# Custo e lift por segmento (usados na estimativa de ROI).
SEGMENT_COST = {
    "retencao_prioritaria": RETENTION_COST_PRIORITARY,
    "acao_automatizada": RETENTION_COST_AUTOMATED,
    "monitorar_upsell": 0.0,
    "monitorar": 0.0,
}
SEGMENT_LIFT = {
    "retencao_prioritaria": LIFT_PRIORITARY,
    "acao_automatizada": LIFT_AUTOMATED,
    "monitorar_upsell": 0.0,
    "monitorar": 0.0,
}


# --- CLV ---

def annual_profit(trans_amt: pd.Series, revolving_bal: pd.Series, card_category: pd.Series) -> pd.Series:
    """Lucro bruto anual estimado por cliente (USD).

    = margem sobre transacoes + juros sobre saldo rotativo + anuidade do cartao.
    """
    fee = card_category.map(CARD_ANNUAL_FEE).fillna(50.0)
    return TRANSACTION_MARGIN * trans_amt + REVOLVING_APR * revolving_bal + fee


def annuity_factor(rate: float, years: int) -> float:
    return (1 - (1 + rate) ** (-years)) / rate


def clv(trans_amt: pd.Series, revolving_bal: pd.Series, card_category: pd.Series) -> pd.Series:
    """Valor do cliente ao longo do tempo (USD) = lucro anual x fator de anuidade."""
    return annual_profit(trans_amt, revolving_bal, card_category) * annuity_factor(DISCOUNT_RATE, CLV_HORIZON_YEARS)


# --- Decisao de acao e threshold otimo ---

def expected_value_matrix(
    score: np.ndarray, clv_value: np.ndarray, cost: float, lift: float
) -> tuple[float, float]:
    """Retorna (EV_descontado por cliente, custo bruto) para UMA regra constante.

    Regra: acionar todos os clientes da populacao com o tipo de acao definido.
    EV por cliente = probabilidade de ser retido pela acao x CLV - custo.
    """
    ev = lift * score * clv_value - cost
    return float(ev.sum()), float(cost * len(score))


def expected_value_on_threshold(
    score: np.ndarray, clv_value: np.ndarray, cost: float, lift: float, threshold: float
) -> float:
    """EV de acionar apenas clientes com score >= threshold."""
    mask = score >= threshold
    return float((lift * score[mask] * clv_value[mask] - cost).sum())


def optimal_threshold(
    score: np.ndarray, clv_value: np.ndarray, cost: float, lift: float
) -> tuple[float, float]:
    """Varre thresholds e devolve (threshold, EV) que maximiza o lucro esperado.

    Nota: com CLV heterogeneo, a regra por-cliente (acionar quando
    lift*p_i*CLV_i > custo) domina qualquer threshold global. A comparacao entre
    as duas e exposada no notebook de negocio.
    """
    candidates = np.quantile(score, np.linspace(0.05, 0.95, 100))
    evs = [expected_value_on_threshold(score, clv_value, cost, lift, t) for t in candidates]
    best = int(np.argmax(evs))
    return float(candidates[best]), float(evs[best])


def per_client_decision(
    score: np.ndarray, clv_value: np.ndarray, cost: float, lift: float
) -> np.ndarray:
    """Decisao otima por cliente: acionar se lift*p*CLV > custo."""
    return lift * score * clv_value > cost


# --- Segmentacao risco x valor (4 quadrantes) ---

def segment_customers(
    score: np.ndarray, clv_value: np.ndarray, risk_threshold: float, clv_quantile: float = 0.50
) -> pd.Series:
    """Classifica clientes nos 4 quadrantes risco x valor.

    - Alto risco: score >= risk_threshold.
    - Alto valor: CLV >= quantil `clv_quantile` da populacao.

    retencoes = alto risco x alto valor
    acao automatizada = alto risco x baixo valor
    monitorar_upsell = baixo risco x alto valor
    monitorar = baixo risco x baixo valor
    """
    high_risk = score >= risk_threshold
    high_value = clv_value >= np.quantile(clv_value, clv_quantile)

    segment = np.where(high_risk & high_value, "retencao_prioritaria", "monitorar")
    segment = np.where(high_risk & ~high_value, "acao_automatizada", segment)
    segment = np.where(~high_risk & high_value, "monitorar_upsell", segment)
    return pd.Series(segment, dtype="category")


def campaign_roi(segment: np.ndarray, score: np.ndarray, clv_value: np.ndarray) -> dict:
    """ROI estimado da campanha por segmento (comparado a nao fazer nada).

    Retorno esperado = soma(score * CLV) dos retidos pela acao.
    """
    seg = pd.Series(segment)
    out = {}
    for name in ["retencao_prioritaria", "acao_automatizada"]:
        mask = (seg == name).to_numpy()
        n = int(mask.sum())
        if n == 0:
            out[name] = {"n_clientes": 0, "custo_total": 0.0, "retorno_esperado": 0.0, "roi": 0.0}
            continue
        cost_total = SEGMENT_COST[name] * n
        retorno = float((SEGMENT_LIFT[name] * score[mask] * clv_value[mask]).sum())
        out[name] = {
            "n_clientes": n,
            "custo_total": round(cost_total, 2),
            "retorno_esperado": round(retorno, 2),
            "roi": round((retorno - cost_total) / cost_total, 4) if cost_total else 0.0,
        }
    return out


def save_business_params(params: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, default=float)


def load_business_params(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)