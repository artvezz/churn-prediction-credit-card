"""Testes do dashboard (executa o script de verdade via AppTest do Streamlit).

Cobre o caminho demo/local (sem rede). O caminho Supabase e exercitado por
`test_dashboard_le_supabase_quando_houver_chaves` e
`test_dashboard_baixa_a_base_inteira_do_supabase`, que so rodam se as variaveis
estiverem no ambiente.
"""

from __future__ import annotations

import contextlib
import os

import pytest

from conftest import DEMO_SCORES, ROOT

pytest.importorskip("streamlit", reason="dashboard exige streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not DEMO_SCORES.exists(),
    reason="demo de scoring ausente: rode `python src/train.py`",
)

APP = str(ROOT / "dashboard" / "app.py")
CHAVES = ("SUPABASE_URL", "SUPABASE_ANON_KEY")


@contextlib.contextmanager
def _sem_supabase():
    """Forca o modo demo durante a execucao e devolve as chaves no fim.

    Sem o `finally`, os testes seguintes que dependem das chaves seriam pulados.
    """
    salvas = {k: os.environ.pop(k, None) for k in CHAVES}
    try:
        yield
    finally:
        os.environ.update({k: v for k, v in salvas.items() if v is not None})


def _run(timeout: int = 120) -> AppTest:
    with _sem_supabase():
        at = AppTest.from_file(APP, default_timeout=timeout)
        at.run()
    return at


def _rerun(at: AppTest) -> None:
    """Re-executa o app mantendo o modo demo."""
    with _sem_supabase():
        at.run()


def test_dashboard_sobe_sem_excecao():
    at = _run()
    assert not at.exception, f"erro no app: {at.exception}"


def test_dashboard_carrega_a_base_demo():
    at = _run()
    assert "Demo local" in at.caption[0].value
    assert at.metric[0].value == "10,127"


def test_dashboard_mostra_kpis_e_segmentos():
    at = _run()
    rotulos = [m.label for m in at.metric]
    assert rotulos == ["Clientes", "Em risco (alta prob.)", "CLV estimado em risco", "Valor médio (CLV)", "Score médio"]

    at.sidebar.multiselect[0].set_value(["retencao_prioritaria"])
    _rerun(at)
    assert not at.exception
    assert at.metric[0].value == "476"
    assert "Filtros da barra lateral aplicados" in at.info[0].value


def test_dashboard_sobrevive_a_filtro_sem_resultado():
    at = _run()
    at.sidebar.slider[0].set_value(1.0)
    _rerun(at)
    assert not at.exception, f"erro no app: {at.exception}"
    assert at.warning[0].value.startswith("Nenhum cliente")


def test_dashboard_filtra_por_score_minimo():
    at = _run()
    at.sidebar.slider[0].set_value(0.9)
    _rerun(at)
    assert not at.exception
    total = int(at.metric[0].value.replace(".", "").replace(",", ""))
    assert 0 < total < 10127


def test_dashboard_exibe_explicabilidade_shap():
    at = _run()
    _rerun(at)
    assert len(at.tabs) == 3
    assert at.selectbox, "aba de explicabilidade sem seletor de cliente"


def test_dashboard_prefere_o_laudo_da_ultima_rodada():
    """Quando existe laudo do lote, o dashboard diz que ele e a origem."""
    laudo_do_lote = ROOT / "data" / "processed" / "shap_by_client.json"
    if not laudo_do_lote.exists():
        pytest.skip("sem laudo de lote: rode o scoring (`POST /run-scoring`)")

    at = _run()
    _rerun(at)
    origem = [c.value for c in at.caption if "Origem do laudo" in c.value]
    assert origem, "dashboard nao informa a origem do laudo SHAP"
    assert origem[0].startswith("Origem do laudo: **lote")


def test_dashboard_le_supabase_quando_houver_chaves():
    pytest.importorskip("requests", reason="dashboard exige requests")
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_ANON_KEY")):
        pytest.skip("defina SUPABASE_URL e SUPABASE_ANON_KEY para testar o caminho Supabase")

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, f"erro no app: {at.exception}"
    assert "Supabase" in at.caption[0].value


def test_dashboard_baixa_a_base_inteira_do_supabase():
    """O PostgREST corta em 1000 linhas: sem paginar, o dashboard esconde a base."""
    pytest.importorskip("requests", reason="dashboard exige requests")
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_ANON_KEY")):
        pytest.skip("defina SUPABASE_URL e SUPABASE_ANON_KEY para testar o caminho Supabase")

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, f"erro no app: {at.exception}"
    total = int(at.metric[0].value.replace(".", "").replace(",", ""))
    assert total > 1000, f"base truncada: o dashboard mostrou apenas {total} clientes"

    at.sidebar.multiselect[0].set_value(["retencao_prioritaria"])
    at.run()
    assert not at.exception
    assert at.metric[0].value == "476"


def test_paginacao_do_supabase_nao_duplica_nem_perde_clientes():
    """Ordenar so por score (que tem empates) faz a paginacao repetir e perder linhas."""
    pytest.importorskip("requests", reason="dashboard exige requests")
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_ANON_KEY")):
        pytest.skip("defina SUPABASE_URL e SUPABASE_ANON_KEY para testar o caminho Supabase")

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, f"erro no app: {at.exception}"

    ids = at.dataframe[0].value["ID"]
    total = int(at.metric[0].value.replace(".", "").replace(",", ""))
    assert len(ids) == total, f"a lista tem {len(ids)} linhas, mas o KPI conta {total}"
    assert ids.nunique() == len(ids), f"clientes repetidos na paginacao: {len(ids) - ids.nunique()}"
