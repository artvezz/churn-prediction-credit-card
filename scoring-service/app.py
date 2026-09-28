"""Servico de scoring de churn (FastAPI, batch).

Fluxo acionado pelo n8n (agendado):
    1. le os dados dos clientes (Supabase `clientes` ou CSV em modo demo);
    2. aplica a mesma engenharia de atributos do treino (features.py);
    3. preve probabilidade de churn com o modelo treinado;
    4. calcula CLV e segmenta em risco x valor (business.py);
    5. grava em `churn_scores` (Supabase) ou `churn_scores_local.csv` (demo);
    6. retorna um resumo para o n8n rotear as acoes.

Modos (env CHURN_SOURCE):
    supabase  -> le e grava no Supabase (default)
    local     -> le data/raw/BankChurners.csv e grava data/processed/... (demo)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel

import business
import explain
import features as fe

MODEL_DIR = Path(__file__).resolve().parent / "model"
# Sem DATA_DIR definido, usa a pasta `data/` do projeto: no container resolve
# para /data (WORKDIR /app + volume montado) e localmente para <projeto>/data.
DATA_DIR = Path(os.environ.get("DATA_DIR") or (Path(__file__).resolve().parent.parent / "data"))

RUN_API_KEY = os.environ.get("RUN_API_KEY", "")
CHURN_SOURCE = os.environ.get("CHURN_SOURCE", "supabase")
CLIENTES_CSV = os.environ.get("CLIENTES_CSV", str(DATA_DIR / "raw" / "BankChurners.csv"))


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ModelBundle:
    """Empacota modelo + metadados carregados uma unica vez no startup."""

    def __init__(self) -> None:
        self.model = joblib.load(MODEL_DIR / "churn_model.pkl")
        self.feature_columns = json.loads((MODEL_DIR / "feature_columns.json").read_text())
        self.medians = {k: float(v) for k, v in json.loads((MODEL_DIR / "medians.json").read_text()).items()}
        bp = load_json(MODEL_DIR / "business_params.json")
        self.risk_threshold = float(bp["risk_threshold"])
        self.clv_quantile = float(bp.get("clv_quantile", 0.50))
        print(f"[startup] modelo carregado · features={len(self.feature_columns)} · threshold={self.risk_threshold}")


bundle = ModelBundle()
app = FastAPI(title="Churn Scoring Service", version="1.0.0")


class RunResponse(BaseModel):
    ok: bool
    rodada: str
    total_clientes: int
    n_por_segmento: dict[str, int]
    score_medio: float
    roi_campanha: dict
    shap_clientes: int = 0
    detalhe: str


def _check_key(x_api_key: str | None) -> None:
    if RUN_API_KEY and x_api_key != RUN_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="chave de execucao invalida")


@app.get("/health")
def health():
    return {"status": "ok", "modelo_carregado": True, "fonte": CHURN_SOURCE}


def _raw_from_supabase() -> pd.DataFrame:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_KEY"]
    rows: list[dict] = []
    offset = 0
    limit = 1000
    while True:
        req = urllib.request.Request(
            f"{url}/rest/v1/clientes?select=*&order=id_cliente.asc&limit={limit}&offset={offset}",
            headers={"apikey": key, "Authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                batch = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Supabase indisponivel: {e}")
        rows.extend(batch)
        if len(batch) < limit:
            break
        offset += limit

    if not rows:
        raise HTTPException(status_code=404, detail="Nenhum cliente encontrado em clientes")
    return pd.DataFrame(rows)


def _to_raw_columns(df_snake: pd.DataFrame) -> pd.DataFrame:
    """Mapeia snake_case do Supabase de volta para as colunas do dataset (camelCase).

    `CLIENT_SNAKE_TO_RAW` ja aponta de snake (Supabase) para camel (dataset), entao
    o rename e direto. Colunas sem equivalente no dataset (ex.: `data_carga`) sao
    descartadas para que `features.build_features` receba o contrato exato do treino.
    """
    out = df_snake.rename(columns=CLIENT_SNAKE_TO_RAW)
    keep = set(CLIENT_SNAKE_TO_RAW.values())
    return out.drop(columns=[c for c in out.columns if c not in keep])


CLIENT_SNAKE_TO_RAW = {
    "id_cliente": "CLIENTNUM",
    "attrition_flag": "Attrition_Flag",
    "customer_age": "Customer_Age",
    "gender": "Gender",
    "dependent_count": "Dependent_count",
    "education_level": "Education_Level",
    "marital_status": "Marital_Status",
    "income_category": "Income_Category",
    "card_category": "Card_Category",
    "months_on_book": "Months_on_book",
    "total_relationship_count": "Total_Relationship_Count",
    "months_inactive_12_mon": "Months_Inactive_12_mon",
    "contacts_count_12_mon": "Contacts_Count_12_mon",
    "credit_limit": "Credit_Limit",
    "total_revolving_bal": "Total_Revolving_Bal",
    "avg_open_to_buy": "Avg_Open_To_Buy",
    "total_amt_chng_q4_q1": "Total_Amt_Chng_Q4_Q1",
    "total_trans_amt": "Total_Trans_Amt",
    "total_trans_ct": "Total_Trans_Ct",
    "total_ct_chng_q4_q1": "Total_Ct_Chng_Q4_Q1",
    "avg_utilization_ratio": "Avg_Utilization_Ratio",
}


def _client_ids(raw: pd.DataFrame) -> pd.Series:
    return raw["CLIENTNUM"] if "CLIENTNUM" in raw.columns else raw["id_cliente"]


def predict(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Engenharia de features -> predict -> CLV -> segmentacao.

    Returns
    -------
    out   : DataFrame com score, CLV, segmento e acao por cliente
    X_fit : matriz de features imputada (mesma que o modelo viu)
    ids   : identificador de cada cliente, alinhado a `out`
    """
    ids = _client_ids(raw)

    X, _ = fe.build_features(raw)
    X = fe.reconcile_columns(X, bundle.feature_columns)
    X = fe.impute_medians(X, bundle.medians)

    proba = bundle.model.predict_proba(X)
    score = proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.ravel()

    clv_value = business.clv(
        raw["Total_Trans_Amt"].astype(float),
        raw["Total_Revolving_Bal"].astype(float),
        raw["Card_Category"].astype(str).str.strip(),
    )
    clv_value = clv_value.to_numpy()

    segmento = business.segment_customers(score, clv_value, bundle.risk_threshold, bundle.clv_quantile)

    out = pd.DataFrame(
        {
            "id_cliente": ids.values,
            "score_churn": np.round(score, 4),
            "clv_usd": np.round(clv_value, 2),
            "segmento": segmento.to_numpy(),
            "acao": [business.ACTION_LABEL[s] for s in segmento.to_numpy()],
        }
    )
    return out, X, ids


def score_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Mesma coisa que `predict`, devolvendo apenas a tabela de scores."""
    out, _, _ = predict(raw)
    return out


def explain_frame(X: pd.DataFrame, ids: pd.Series, rodada: str) -> int:
    """Produz o laudo SHAP da rodada e grava em `data/processed/shap_by_client.json`.

    O dashboard le esse arquivo, entao o "por que este cliente esta em risco"
    corresponde sempre ao modelo que gerou o score — em vez de ficar congelado
    no ultimo treino. Falha aqui nao derruba o lote: e uma camada de explica-
    bilidade, nao o score em si.
    """
    try:
        drivers = explain.shap_drivers(bundle.model, X, ids)
    except Exception as exc:  # pragma: no cover - defensivo
        print(f"[aviso] SHAP nao calculado: {exc}")
        return 0

    drivers["_meta"] = {"data_execucao": rodada, "n_clientes": int(len(ids))}
    dest = DATA_DIR / "processed" / "shap_by_client.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(drivers, f)
    print(f"[shap] laudo de {len(ids)} clientes salvo em {dest}")
    return len(ids)


def _upsert_supabase(out: pd.DataFrame) -> None:
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ["SUPABASE_SERVICE_KEY"]
    payload = out.to_dict(orient="records")
    req = urllib.request.Request(
        f"{url}/rest/v1/churn_scores",
        data=json.dumps(payload).encode(),
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status >= 300:
                raise HTTPException(status_code=resp.status, detail="falha ao gravar churn_scores")
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"falha ao gravar churn_scores: {e}")


@app.post("/run-scoring", response_model=RunResponse)
def run_scoring(x_api_key: str | None = Header(default=None)):
    _check_key(x_api_key)

    if CHURN_SOURCE == "supabase":
        raw = _raw_from_supabase()
        raw = _to_raw_columns(raw)
        source_desc = "supabase:clientes"
    else:
        path = Path(CLIENTES_CSV)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"CSV de clientes nao encontrado: {path}")
        raw = pd.read_csv(path, encoding="utf-8")
        source_desc = f"csv:{path.name}"

    out, X_fit, ids = predict(raw)
    rodada = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    out["data_execucao"] = rodada

    n_shap = explain_frame(X_fit, ids, rodada)

    if CHURN_SOURCE == "supabase":
        _upsert_supabase(out)
    else:
        dest = DATA_DIR / "processed" / "churn_scores_local.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest, index=False)

    n_by_seg = {s: int((out["segmento"] == s).sum()) for s in business.SEGMENTS}
    roi = business.campaign_roi(out["segmento"].to_numpy(), out["score_churn"].to_numpy(), out["clv_usd"].to_numpy())

    return RunResponse(
        ok=True,
        rodada=rodada,
        total_clientes=len(out),
        n_por_segmento=n_by_seg,
        score_medio=round(float(out["score_churn"].mean()), 4),
        roi_campanha=roi,
        shap_clientes=n_shap,
        detalhe=f"Fonte: {source_desc} · {len(out)} clientes pontuados",
    )
