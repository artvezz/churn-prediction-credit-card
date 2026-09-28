"""Pipeline end-to-end de treino: features -> CV -> modelos -> artefatos.

Uso:
    python src/train.py

Gera em `artifacts/`:
    churn_model.pkl        modelo final (XGBoost) pronto para o servico de scoring
    feature_columns.json   ordem das colunas do modelo
    medians.json           medianas de imputacao (treino)
    business_params.json   premissoes financeiras + threshold otimo + quantil de valor
    business_results.json  comparativo de cenarios (do-nothing vs. acoes)
    models_summary.json    metricas da CV comparando os modelos
    shap_local.json        top contribuicoes SHAP por cliente (dashboard)
    demo_churn_scores.csv  demo de run de scoring para dashboard/modo local
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

import business
import features as fe

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw" / "BankChurners.csv"
ARTIFACTS = ROOT / "artifacts"
PROCESSED = ROOT / "data" / "processed"

FLAG_POSITIVE = fe.TARGET_POSITIVE


# ------------------------------ Carregamento ------------------------------

def load_raw(path: pathlib.Path = DATA_RAW) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df[fe.TARGET_COLUMN] = df[fe.TARGET_COLUMN].astype(str).str.strip()
    return df


# ------------------------------ Metricas ------------------------------

def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Metricas prioritarias: PR-AUC, recall, f1. Acurcia fica fora do foco."""
    pred = (proba >= 0.5).astype(int)
    pred_low = (proba >= 0.30).astype(int)
    return {
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "precision_at_0.5": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall_at_0.5": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "recall_at_0.3": round(recall_score(y_true, pred_low, zero_division=0), 4),
        "precision_at_0.3": round(precision_score(y_true, pred_low, zero_division=0), 4),
    }


# ------------------------------ Modelos ------------------------------

def make_logreg(X_train: pd.DataFrame, y_train: pd.Series | None = None):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Features em escalas muito diferentes (limite em milhares vs. razoes 0-1);
    # padronizacao e essencial para a regressao logistica convergir.
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42),
            ),
        ]
    )


def make_xgb(X_train: pd.DataFrame, y_train: pd.Series, **overrides) -> XGBClassifier:
    pos = int(y_train.sum())
    neg = int((1 - y_train).sum())
    params = dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=neg / max(pos, 1),
        n_estimators=800,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    params.update(overrides)
    return XGBClassifier(**params)


def cv_evaluate(model_factory, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
    """Validacao cruzada estratificada. Retorna DataFrame fold x metricas."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        Xtr_fit = fe.impute_medians(Xtr, fe.median_imputation_values(Xtr))
        Xva_fit = fe.impute_medians(Xva, fe.median_imputation_values(Xtr))
        model = model_factory(Xtr, ytr)
        model.fit(Xtr_fit, ytr)
        proba = churn_proba(model, Xva_fit)
        rows.append({"fold": fold, **evaluate(yva.to_numpy(), proba)})
    return pd.DataFrame(rows)


def churn_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Probabilidade da classe positiva (churn = 'Attrited Customer')."""
    p = model.predict_proba(X)
    return p[:, 1] if p.shape[1] == 2 else p[:, 0]


# ------------------------------ Treino final ------------------------------

def tune_xgboost(Xtr: pd.DataFrame, ytr: pd.Series, n_iter: int = 20) -> XGBClassifier:
    """Busca aleatoria leve de hyperparams com CV interna no treino."""
    from sklearn.model_selection import RandomizedSearchCV

    pos = int(ytr.sum())
    neg = int((1 - ytr).sum())
    base = make_xgb(Xtr, ytr)
    search = RandomizedSearchCV(
        base,
        param_distributions={
            "n_estimators": [300, 500, 800],
            "learning_rate": [0.03, 0.08, 0.15],
            "max_depth": [3, 4, 6],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.7, 0.9],
            "min_child_weight": [1, 3],
            "reg_lambda": [0.5, 1.0, 3.0],
            "scale_pos_weight": [neg / max(pos, 1)],
        },
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="average_precision",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(Xtr, ytr)
    return search.best_estimator_


def fit_final_model(X, y) -> XGBClassifier:
    """Ajusta o melhor XGBoost no treino completo."""
    X_fit = fe.impute_medians(X, fe.median_imputation_values(X))
    best = tune_xgboost(X_fit, y, n_iter=20)
    best.fit(X_fit, y)
    return best


# ------------------------------ Artefatos e demo ------------------------------

def compute_clv(raw: pd.DataFrame) -> pd.Series:
    cat = raw["Card_Category"].astype(str).str.strip()
    return business.clv(
        raw["Total_Trans_Amt"], raw["Total_Revolving_Bal"], cat
    )


def build_demo_scores(
    raw: pd.DataFrame,
    X: pd.DataFrame,
    medians: dict,
    model,
    clv_value: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    X_fit = fe.impute_medians(X, medians)
    score = churn_proba(model, X_fit)
    segment = business.segment_customers(score, clv_value.to_numpy(), threshold).to_numpy()

    out = pd.DataFrame(
        {
            "id_cliente": raw[fe.ID_COLUMN].values,
            "score_churn": np.round(score, 4),
            "clv_usd": np.round(clv_value.values, 2),
            "segmento": segment,
            "acao": [business.ACTION_LABEL[s] for s in segment],
        }
    )
    out["data_execucao"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    return out


def shap_drivers(model, X: pd.DataFrame, medians: dict, raw: pd.DataFrame | None = None) -> None:
    """Calcula SHAP (TreeExplainer) e salva as top contribuicoes por cliente.

    A logica esta em `explain.py` (fonte unica com o servico de scoring); aqui
    apenas materializamos os artefatos dos notebooks:
      - shap_local.json: drivers indexados por posicao (ordem dos dados);
      - shap_by_client.json: drivers indexados por id_cliente (para o dashboard).
    """
    import explain

    X_fit = fe.impute_medians(X, medians)
    by_position = explain.shap_drivers(model, X_fit, ids=list(range(len(X_fit))))

    with open(ARTIFACTS / "shap_local.json", "w", encoding="utf-8") as f:
        json.dump(by_position, f)

    if raw is not None:
        by_client = {
            str(int(raw.iloc[i][fe.ID_COLUMN])): by_position[str(i)] for i in range(len(by_position))
        }
        with open(ARTIFACTS / "shap_by_client.json", "w", encoding="utf-8") as f:
            json.dump(by_client, f)
    print("SHAP local salvo em artifacts/shap_local.json e artifacts/shap_by_client.json")


def main() -> None:
    t0 = time.time()
    ARTIFACTS.mkdir(exist_ok=True)
    PROCESSED.mkdir(exist_ok=True)

    raw = load_raw()
    print(f"Dados brutos: {raw.shape}")

    # Features (com NaN; imputacao via medianas do treino).
    X, feature_columns = fe.build_features(raw)
    y = (raw[fe.TARGET_COLUMN] == FLAG_POSITIVE).astype(int).reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Split estratificado para avaliacao final.
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    medians = fe.median_imputation_values(X_train)
    X_train_fit = fe.impute_medians(X_train, medians)
    X_hold_fit = fe.impute_medians(X_hold, medians)

    # 1) Baseline: regressao logistica
    print("\n-- CV: Regressao Logistica (class_weight=balanced) --")
    cv_lr = cv_evaluate(make_logreg, X, y)
    print(cv_lr.drop(columns="fold").mean().to_string())

    # 2) XGBoost com scale_pos_weight (sem ajuste fino)
    print("\n-- CV: XGBoost (scale_pos_weight) --")
    cv_xgb = cv_evaluate(make_xgb, X, y)
    print(cv_xgb.drop(columns="fold").mean().to_string())

    # 3) XGBoost tunado
    print("\n-- Tunando XGBoost (busca aleatoria, CV interna) --")
    final_model = fit_final_model(X_train_fit, y_train)
    print(f"   Best params: {final_model.get_params()}")

    print("\n-- Holdout (20%) com modelo final --")
    hold_metrics = evaluate(y_hold.to_numpy(), churn_proba(final_model, X_hold_fit))
    print(pd.Series(hold_metrics).to_string())

    # Camada de negocio: CLV e decisao de acao
    clv_all = compute_clv(raw)

    # Otimiza threshold global usando holdout (custos da acao prioritaria).
    hold_score = churn_proba(final_model, X_hold_fit)
    hold_clv = clv_all.iloc[X_hold.index].to_numpy()
    threshold_opt, ev_opt = business.optimal_threshold(
        hold_score, hold_clv, business.RETENTION_COST_PRIORITARY, business.LIFT_PRIORITARY
    )
    print(f"\nThreshold otimo (EV maximo): {threshold_opt:.3f} | EV esperado: {ev_opt:.0f} USD na holdout")

    # Persiste comparativo de metricas (CV e holdout) para o dashboard/relatorio.
    summary = {
        "regressao_logistica_cv": cv_lr.drop(columns="fold").mean().round(4).to_dict(),
        "xgboost_cv": cv_xgb.drop(columns="fold").mean().round(4).to_dict(),
        "xgboost_tunado_holdout": hold_metrics,
        "best_params": {k: v for k, v in final_model.get_params().items() if k in [
            "n_estimators", "learning_rate", "max_depth", "subsample",
            "colsample_bytree", "min_child_weight", "reg_lambda", "scale_pos_weight",
        ]},
    }
    with open(ARTIFACTS / "models_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    business_params = {
        "trans_margin": business.TRANSACTION_MARGIN,
        "revolving_apr": business.REVOLVING_APR,
        "fee_by_card": business.CARD_ANNUAL_FEE,
        "clv_horizon_years": business.CLV_HORIZON_YEARS,
        "discount_rate": business.DISCOUNT_RATE,
        "retention_cost_prioritary": business.RETENTION_COST_PRIORITARY,
        "retention_cost_automated": business.RETENTION_COST_AUTOMATED,
        "lift_prioritary": business.LIFT_PRIORITARY,
        "lift_automated": business.LIFT_AUTOMATED,
        "risk_threshold": round(threshold_opt, 4),
        "clv_quantile": 0.50,
        "metricas_holdout": hold_metrics,
    }

    # Comparativo de cenarios na holdout:
    scenarios = business_scenarios(hold_score, hold_clv)
    business_params["cenarios_holdout"] = scenarios
    business.save_business_params(scenarios, ARTIFACTS / "business_results.json")

    # Segmentacao completa (para demo/dashboard)
    segment_all = business.segment_customers(
        churn_proba(final_model, fe.impute_medians(X, medians)),
        clv_all.to_numpy(),
        threshold_opt,
    )
    business_params["n_segmento"] = {
        s: int((segment_all == s).sum()) for s in business.SEGMENTS
    }

    # Artefatos
    joblib.dump(final_model, ARTIFACTS / "churn_model.pkl")
    fe.save_feature_columns(feature_columns, ARTIFACTS / "feature_columns.json")
    with open(ARTIFACTS / "medians.json", "w", encoding="utf-8") as f:
        json.dump(medians, f, indent=2)
    business.save_business_params(business_params, ARTIFACTS / "business_params.json")

    # Demo de run de scoring (para dashboard / modo local do servico)
    demo = build_demo_scores(raw, X, medians, final_model, clv_all, threshold_opt)
    demo.to_csv(PROCESSED / "churn_scores_demo.csv", index=False)

    # SHAP local
    shap_drivers(final_model, X, medians, raw)

    print(f"\nArtefatos em {ARTIFACTS}")
    print(f"Tempo total: {time.time() - t0:.1f}s")


def business_scenarios(score: np.ndarray, clv_value: np.ndarray) -> dict:
    """Compara politicas de acao na holdout e reporta EV e custo.

    - do nothing: nao aciona ninguem (baseline de hoje).
    - threshold global otimo: aciona quem tem score >= t* (custo/lift prioritaria).
    - regra por-cliente: aciona apenas quando lift*p*CLV > custo (otima global).
    """
    threshold_opt, ev_thr = business.optimal_threshold(
        score, clv_value, business.RETENTION_COST_PRIORITARY, business.LIFT_PRIORITARY
    )
    decide = business.per_client_decision(
        score, clv_value, business.RETENTION_COST_PRIORITARY, business.LIFT_PRIORITARY
    )
    ev_per_client = float(
        (business.LIFT_PRIORITARY * score * clv_value - business.RETENTION_COST_PRIORITARY)[
            decide
        ].sum()
    )
    cost_per_client = business.RETENTION_COST_PRIORITARY * int(decide.sum())
    return {
        "do_nothing_ev": 0.0,
        "ev_global_threshold_prioritary": round(ev_thr, 2),
        "threshold_global": round(threshold_opt, 4),
        "ev_prioritary_per_client": round(ev_per_client, 2),
        "custo_prioritary_per_client": round(cost_per_client, 2),
        "n_prioritary_per_client": int(decide.sum()),
    }


if __name__ == "__main__":
    main()