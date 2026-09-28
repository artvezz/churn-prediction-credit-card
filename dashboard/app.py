"""Dashboard de priorizacao de retencao de churn (Streamlit).

Le os resultados de scoring de:
  1. Supabase (view v_priorizados) se SUPABASE_URL + SUPABASE_ANON_KEY estiverem no .env;
  2. senao, fallback local: data/processed/churn_scores_demo.csv (run de exemplo).

Uso:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import hashlib
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


def _fonte_dados() -> str:
    """Identidade da fonte de dados (host + hash da chave), usada como chave de cache.

    Sem isso, `cache_data` devolveria o resultado da configuracao anterior por ate
    5 minutos apos editar o `.env` — o dashboard continuaria mostrando a demo.
    A chave em si nunca e exposta: entra so o hash.
    """
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_ANON_KEY", "")
    if not (url and key):
        return "demo"
    return f"{url}|{hashlib.sha256(key.encode()).hexdigest()[:12]}"


PAGINA = 1000
MAX_PAGINAS = 50  # teto de seguranca: 50k clientes


@st.cache_data(ttl=300, show_spinner=False)
def load_scores(fonte: str) -> tuple[pd.DataFrame, str]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_ANON_KEY", "")
    if fonte != "demo" and url and key:
        cabecalho = {"apikey": key, "Authorization": f"Bearer {key}"}
        # Pagina de verdade: o PostgREST corta em 1000 linhas por requisicao, e sem
        # isso o dashboard mostraria so os 1000 clientes de maior score.
        linhas: list[dict] = []
        try:
            for pagina in range(MAX_PAGINAS):
                r = requests.get(
                    f"{url}/rest/v1/v_priorizados",
                    headers=cabecalho,
                    params={"select": "*", "order": "score_churn.desc", "limit": PAGINA, "offset": pagina * PAGINA},
                    timeout=8,
                )
                if r.status_code != 200:
                    break
                bloco = r.json()
                linhas += bloco
                if len(bloco) < PAGINA:
                    break
            if linhas:
                df = pd.DataFrame(linhas)
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


LAUDO_ROTULOS = (
    (ROOT / "data" / "processed" / "shap_by_client.json", "lote"),
    (ROOT / "artifacts" / "shap_by_client.json", "treino"),
)


def _arquivo_laudo() -> Path | None:
    for caminho, _ in LAUDO_ROTULOS:
        if caminho.exists():
            return caminho
    return None


@st.cache_data(ttl=300, show_spinner=False)
def load_drivers(caminho: str, mtime: float) -> tuple[dict[str, Any], str]:
    """Laudo SHAP por cliente.

    Prioridade: arquivo da ultima rodada (`data/processed/shap_by_client.json`,
    escrito pelo servico de scoring) e, na ausencia dele, o gerado no treino
    (`artifacts/shap_by_client.json`). O `mtime` entra na chave do cache para que
    um laudo novo gerado pelo lote apareca imediatamente.
    """
    if not caminho:
        return {}, "indisponivel"
    rotulo = next((r for c, r in LAUDO_ROTULOS if str(c) == caminho), "lote")
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    meta = dados.pop("_meta", None)
    quando = meta.get("data_execucao") if isinstance(meta, dict) else None
    return dados, rotulo + (f" ({quando})" if quando else "")


def kpi_cols(n: int) -> tuple[Any, ...]:
    return st.columns(n)


st.set_page_config(page_title="Priorizacao de Retencao - Churn", page_icon="📊", layout="wide")

df, fonte = load_scores(_fonte_dados())
_laudo = _arquivo_laudo()
drivers, origem_drivers = load_drivers(
    str(_laudo) if _laudo else "", _laudo.stat().st_mtime if _laudo else 0.0
)

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
filtros_ativos = len(df_f) != len(df) or set(seg_sel) != set(ORDEM) or min_score > 0.0

high_risk = df_f["score_churn"] >= bp.get("risk_threshold", 0.5)
clv_em_risco = float(df_f.loc[high_risk, "clv_usd"].sum()) if len(df_f) else 0.0
resp_total = int(high_risk.sum())
media_clv = float(df_f["clv_usd"].mean()) if len(df_f) else 0.0
media_score = float(df_f["score_churn"].mean()) if len(df_f) else 0.0

tab1, tab2, tab3 = st.tabs(["Visão geral", "Clientes priorizados", "Explicabilidade"])

with tab1:
    if filtros_ativos:
        st.info(f"Filtros da barra lateral aplicados: **{len(df_f):,}** de {len(df):,} clientes.")
    if not len(df_f):
        st.warning("Nenhum cliente atende aos filtros. Ajuste o score mínimo ou os segmentos.")
    c1, c2, c3, c4, c5 = kpi_cols(5)
    c1.metric("Clientes", f"{len(df_f):,}")
    c2.metric("Em risco (alta prob.)", f"{resp_total:,}", help="score >= threshold do modelo")
    c3.metric("CLV estimado em risco", f"US$ {clv_em_risco:,.0f}")
    c4.metric("Valor médio (CLV)", f"US$ {media_clv:,.0f}")
    c5.metric("Score médio", f"{media_score:.3f}")

    st.subheader("Distribuição por segmento (risco × valor)")
    counts = df_f["segmento"].value_counts()
    fig1 = px.bar(
        counts.reindex([s for s in ORDEM if s in counts.index]),
        orientation="v",
        color_discrete_map=SEG_CORES,
        labels={"value": "Clientes", "index": "Segmento"},
    )
    fig1.update_traces(marker_color=[SEG_CORES[s] for s in counts.reindex([s for s in ORDEM if s in counts.index]).index])
    st.plotly_chart(fig1, width="stretch")

    st.subheader("Risco × Valor (quadrantes)")
    fig2 = px.scatter(
        df_f,
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
    st.plotly_chart(fig2, width="stretch")

with tab2:
    st.subheader("Lista de priorização (ordem: risco e valor)")
    cols_show = ["id_cliente", "score_churn", "clv_usd", "segmento", "acao", "data_execucao"]
    tab = df_f.sort_values(["segmento", "score_churn"], ascending=[True, False] if len(df_f) else False)[cols_show]
    tab["segmento"] = tab["segmento"].map(SEG_LABEL).fillna(tab["segmento"])
    tab.columns = ["ID", "Score churn", "CLV (USD)", "Segmento", "Ação recomendada", "Execução"]
    st.dataframe(tab, width="stretch", hide_index=True)

    st.download_button(
        "⬇ Baixar CSV dos priorizados",
        df_f.to_csv(index=False).encode("utf-8"),
        file_name="churn_priorizados.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Por que cada cliente está em risco? (SHAP local)")
    st.caption("Motivos que empurram o risco e motivos que protegem — para personalizar a oferta de retenção.")
    st.caption(f"Origem do laudo: **{origem_drivers}")
    id_opts = df_f.sort_values("score_churn", ascending=False)["id_cliente"].astype(str).tolist()[:50]
    if not id_opts:
        st.warning("Nenhum cliente na seleção atual. Ajuste os filtros da barra lateral.")
    else:
        escolha = st.selectbox("Cliente", id_opts, format_func=lambda x: f"{x} · score {df_f[df_f['id_cliente'].astype(str) == x]['score_churn'].iloc[0]:.3f}")
        d = drivers.get(escolha)
        if d:
            motivos = [x["feature"] for x in d["pressiona_churn"]] + [x["feature"] for x in d["protege_churn"]]
            vals = [x["values"] for x in d["pressiona_churn"]] + [x["values"] for x in d["protege_churn"]]
            cores = ["#c7417b"] * len(d["pressiona_churn"]) + ["#2e86ab"] * len(d["protege_churn"])
            tabela = pd.DataFrame({"feature": motivos, "values": vals, "cor": cores})
            tabela = tabela.iloc[tabela["values"].abs().sort_values(ascending=False).index]
            fig3 = px.bar(
                tabela,
                x="values", y="feature", orientation="h", color="cor",
                color_discrete_map={"#c7417b": "#c7417b", "#2e86ab": "#2e86ab"},
            )
            fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, width="stretch")
            st.caption(f"Vermelho: empurra o churn | Azul: protege do churn — cliente **{escolha}**")
        else:
            st.info("Sem laudo SHAP para este cliente na rodada atual.")
