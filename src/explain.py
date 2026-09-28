"""Explicabilidade local (SHAP) — fonte unica treino <-> servico.

O mesmo modulo e usado no `src/train.py` (para gerar os artefatos dos notebooks) e
no `scoring-service` (para produzir o laudo de cada cliente em cada rodada do lote).
Assim o "por que este cliente esta em risco" sempre corresponde ao modelo que gerou
o score, em vez de ficar congelado no ultimo treino.

Contrato de saida (consumido pelo dashboard):
    {
      "<id_cliente>": {
          "pressiona_churn": [{"feature": ..., "values": ...}, ...],
          "protege_churn":  [{"feature": ..., "values": ...}, ...],
      },
      ...
    }
"""

from __future__ import annotations

import pandas as pd


def shap_drivers(
    model, X: pd.DataFrame, ids: pd.Series | list, top: int = 3
) -> dict[str, dict]:
    """Top contribuicoes SHAP por cliente, indexado por `id_cliente`.

    Usa TreeExplainer sobre a matriz ja imputada. Para cada cliente, ordena as
    features por impacto absoluto e devolve ate `top` que empurram o churn
    (contribuicao positiva) e ate `top` que protegem (contribuicao negativa), cada
    grupo ja ordenado do maior para o menor impacto.
    """
    import shap

    values = shap.TreeExplainer(model).shap_values(X)
    cols = list(X.columns)
    drivers: dict[str, dict] = {}

    for i, client_id in enumerate(ids):
        order = abs(values[i]).argsort()[::-1]
        pressiona, protege = [], []
        for j in order:
            name = cols[j]
            value = float(values[i][j])
            if value >= 0:
                if len(pressiona) < top:
                    pressiona.append({"feature": name, "values": round(value, 4)})
            elif len(protege) < top:
                protege.append({"feature": name, "values": round(value, 4)})
            if len(pressiona) >= top and len(protege) >= top:
                break
        drivers[str(int(client_id))] = {
            "pressiona_churn": sorted(pressiona, key=lambda x: -abs(x["values"])),
            "protege_churn": sorted(protege, key=lambda x: -abs(x["values"])),
        }

    return drivers
