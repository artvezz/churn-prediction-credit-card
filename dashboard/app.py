"""Dashboard de priorizacao de retencao de churn (Streamlit).

Le os resultados de scoring de:
  1. Supabase (view v_priorizados) se SUPABASE_URL + SUPABASE_ANON_KEY estiverem no .env;
  2. senao, fallback local: data/processed/churn_scores_demo.csv (run de exemplo).

Uso:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SEG_CORES = {
    "retencao_prioritaria": "#c7417b",
    "acao_automatizada": "#e8a838",
    "monitorar_upsell": "#2e86ab",
    "monitorar": "#8f9aa7",
}
SEG_LABEL = {
    "retencao_prioritaria": "Retencao prioritaria (contato pessoal)",
    "acao_automatizada": "Acao automatizada (baixo custo)",
    "monitorar_upsell": "Monitorar + upsell",
    "monitorar": "Monitorar",
}
ORDEM = ["retencao_prioritaria", "acao_automatizada", "monitorar_upsell", "monitorar"]


@st.cache_data(ttl=300, show_spinner=False)
def load_scores() -> tuple[pd.DataFrame, str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_ANON_KEY", "")
    if url and key:
        try:
            r = requests.get(
                f"{url}/rest/v1/v_priorizados?select=*&order=score_churn.desc",
                headers={"apikey": key, "Authorization": f"Bearer {key}"},
                timeout=8,
            )
            if r.status_code == 200 and r.json():
                df = pd.DataFrame(r.json())
                df["data_execucao"] = pd.to_datetime(df["data_execucao"])
                return df, "Supabase"
        except requests.RequestException:
            pass
    demo = ROOT / "data" / "processed" / "churn_scores_demo.csv"
    if not demo.exists():
        return pd.DataFrame(), "Demo local (sem dados)"
    df = pd.read_csv(demo).rename(columns={"CLIENTNUM": "id_cliente"})
    df["data_execucao"] = pd.to_datetime(df["data_execucao"])
    return df, f"Demo local ({demo.name})"


@st.cache_data
def load_drivers() -> dict[str, Any]:
    path = ROOT / "artifacts" / "shap_by_client.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def kpi_cols(n: int) -> tuple[Any, ...]:
    return st.columns(n)


st.set_page_config(page_title="Priorizacao de Retencao - Churn", page_icon="📊", layout="wide")

df, fonte = load_scores()
drivers = load_drivers()

st.title("Priorização de Retenção — Churn de Cartão de Crédito")
st.caption(f"Fonte dos dados: **{fonte}** · execução mais recente: {df['data_execucao'].max()}")

bp = {}
path_bp = ROOT / "artifacts" / "business_params.json"
if path_bp.exists():
    bp = json.loads(path_bp.read_text(encoding="utf-8"))

with st.sidebar:
    st.header("Filtros")
    seg_sel = st.multiselect("Segmento", ORDEM, default=ORDEM)
    min_score = st.slider("Score mínimo (prob. de churn)", 0.0, 1.0, 0.0, 0.01)
    if st.button("Recarregar dados"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption(f"Threshold do modelo: {bp.get('risk_threshold', '-'):.3f}" if isinstance(bp.get('risk_threshold'), float) else "Threshold do modelo: -")

df_f = df[(df["segmento"].isin(seg_sel)) & (df["score_churn"] >= min_score)]

high_risk = df["score_churn"] >= bp.get("risk_threshold", 0.5)
clv_em_risco = float(df.loc[high_risk, "clv_usd"].sum())
resp_total = int(high_risk.sum())

tab1, tab2, tab3 = st.tabs(["Visão geral", "Clientes priorizados", "Explicabilidade"])

with tab1:
    c1, c2, c3, c4, c5 = kpi_cols(5)
    c1.metric("Clientes", f"{len(df):,}")
    c2.metric("Em risco (alta prob.)", f"{resp_total:,}", help="score >= threshold do modelo")
    c3.metric("CLV estimado em risco", f"US$ {clv_em_risco:,.0f}")
    c4.metric("Valor médio (CLV)", f"US$ {df['clv_usd'].mean():,.0f}")
    c5.metric("Score médio", f"{df['score_churn'].mean():.3f}")

    st.subheader("Distribuição por segmento (risco × valor)")
    counts = df["segmento"].value_counts()
    fig1 = px.bar(
        counts.reindex([s for s in ORDEM if s in counts.index]),
        orientation="v",
        color_discrete_map=SEG_CORES,
        labels={"value": "Clientes", "index": "Segmento"},
    )
    fig1.update_traces(marker_color=[SEG_CORES[s] for s in counts.reindex([s for s in ORDEM if s in counts.index]).index])
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Risco × Valor (quadrantes)")
    fig2 = px.scatter(
        df,
        x="clv_usd",
        y="score_churn",
        color="segmento",
        hover_data=["id_cliente"],
        color_discrete_map=SEG_CORES,
        opacity=0.75,
        labels={"clv_usd": "CLV (USD)", "score_churn": "Probabilidade de churn"},
    )
    if isinstance(bp.get("risk_threshold"), float):
        fig2.add_hline(y=bp["risk_threshold"], line_dash="dash", line_color="gray",
                       annotation_text="threshold", annotation_position="top right")
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("Lista de priorização (ordem: risco e valor)")
    cols_show = ["id_cliente", "score_churn", "clv_usd", "segmento", "acao", "data_execucao"]
    tab = df_f.sort_values(["segmento", "score_churn"], ascending=[True, False] if len(df_f) else False)[cols_show]
    tab["segmento"] = tab["segmento"].map(SEG_LABEL).fillna(tab["segmento"])
    tab.columns = ["ID", "Score churn", "CLV (USD)", "Segmento", "Ação recomendada", "Execução"]
    st.dataframe(tab, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇ Baixar CSV dos priorizados",
        df_f.to_csv(index=False).encode("utf-8"),
        file_name="churn_priorizados.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Por que cada cliente está em risco? (SHAP local)")
    st.caption("Motivos que empurram o risco e motivos que protegem — para personalizar a oferta de retenção.")
    id_opts = df_f.sort_values("score_churn", ascending=False)["id_cliente"].astype(str).tolist()[:50]
    escolha = st.selectbox("Cliente", id_opts, format_func=lambda x: f"{x} · score {df_f[df_f['id_cliente'].astype(str) == x]['score_churn'].iloc[0]:.3f}")
    d = drivers.get(escolha)
    if d:
        motivos = [x["feature"] for x in d["pressiona_churn"]] + [x["feature"] for x in d["protege_churn"]]
        vals = [x["values"] for x in d["pressiona_churn"]] + [x["values"] for x in d["protege_churn"]]
        cores = ["#c7417b"] * len(d["pressiona_churn"]) + ["#2e86ab"] * len(d["protege_churn"])
        fig3 = px.bar(
            pd.DataFrame({"feature": motivos, "values": vals, "cor": cores}),
            x="values", y="feature", orientation="h", color="cor",
            color_discrete_map={"#c7417b": "#c7417b", "#2e86ab": "#2e86ab"},
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.caption(f"Vermelho: empurra o churn | Azul: protege do churn — cliente **{escolha}**")
    else:
        st.info("Explicabilidade disponível no modo demo/local (mapeamento SHAP por cliente).")