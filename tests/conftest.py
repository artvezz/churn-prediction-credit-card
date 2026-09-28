"""Configuração compartilhada dos testes.

Deixa `src/` e `scoring-service/` importáveis (mesmo mecanismo do PYTHONPATH do
container) e expõe fixtures para os artefatos do treino.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scoring-service"))

ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
RAW_CSV = DATA / "raw" / "BankChurners.csv"
DEMO_SCORES = DATA / "processed" / "churn_scores_demo.csv"

# Sem os artefatos do treino so da para rodar os testes de negocio/feature engineering.
sem_treino = pytest.mark.skipif(
    not (ARTIFACTS / "churn_model.pkl").exists(),
    reason="artefatos de treino ausentes: rode `python src/train.py`",
)


@pytest.fixture(scope="session")
def raw() -> "pd.DataFrame":
    pd = pytest.importorskip("pandas")
    if not RAW_CSV.exists():
        pytest.skip("dataset ausente: baixe o BankChurners.csv em data/raw/")
    return pd.read_csv(RAW_CSV, encoding="utf-8")


@pytest.fixture(scope="session")
def feature_columns() -> list[str]:
    return json.loads((ARTIFACTS / "feature_columns.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def medians() -> dict[str, float]:
    return {
        k: float(v)
        for k, v in json.loads((ARTIFACTS / "medians.json").read_text(encoding="utf-8")).items()
    }


@pytest.fixture(scope="session")
def business_params() -> dict:
    return json.loads((ARTIFACTS / "business_params.json").read_text(encoding="utf-8"))
