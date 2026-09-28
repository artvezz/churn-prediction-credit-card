-- Schema inicial das tabelas do projeto (Supabase / PostgreSQL).
-- Rode no SQL Editor do Supabase (ou via `supabase db push`).

-- ---------------------------------------------------------------------------
-- clientes: dados cadastrais + comportamentais (entrada do modelo de churn).
-- Colunas espelham o dataset BankChurners (credit card customers), com as
-- 2 colunas de leak do target (Naive_Bayes_*) JÁ REMOVIDAS.
-- ---------------------------------------------------------------------------

create table if not exists public.clientes (
    id_cliente            bigint primary key,            -- CLIENTNUM
    attrition_flag        text not null default 'Existing Customer',
    customer_age          integer,
    gender                text,
    dependent_count       integer,
    education_level       text,
    marital_status        text,
    income_category       text,
    card_category         text,
    months_on_book        integer,
    total_relationship_count integer,
    months_inactive_12_mon  integer,
    contacts_count_12_mon   integer,
    credit_limit          numeric(14,2),
    total_revolving_bal   numeric(14,2),
    avg_open_to_buy       numeric(14,2),
    total_amt_chng_q4_q1  numeric(8,4),
    total_trans_amt       numeric(14,2),
    total_trans_ct        integer,
    total_ct_chng_q4_q1   numeric(8,4),
    avg_utilization_ratio numeric(8,4),
    data_carga            timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- churn_scores: resultado de cada rodada de scoring (lote agendado).
-- Escrita pelo serviço de scoring; lida pelo n8n e pelo dashboard.
-- ---------------------------------------------------------------------------

create table if not exists public.churn_scores (
    id_cliente      bigint not null references public.clientes (id_cliente) on delete cascade,
    score_churn     numeric(6,4) not null,     -- probabilidade de churn (0-1)
    clv_usd         numeric(14,2),             -- valor do cliente (USD)
    segmento        text not null,             -- retencao_prioritaria | acao_automatizada | monitorar_upsell | monitorar
    acao            text,
    data_execucao   timestamptz not null default now(),
    primary key (id_cliente, data_execucao)
);

create index if not exists idx_churn_scores_execucao
    on public.churn_scores (data_execucao desc);
create index if not exists idx_churn_scores_segmento
    on public.churn_scores (segmento);
create index if not exists idx_clientes_churn_prob
    on public.churn_scores (segmento, score_churn desc);

-- ---------------------------------------------------------------------------
-- View de priorização: última rodada de cada cliente, pronta para o dashboard.
-- ---------------------------------------------------------------------------

create or replace view public.v_priorizados as
select
    c.id_cliente,
    c.customer_age,
    c.gender,
    c.income_category,
    c.card_category,
    c.months_inactive_12_mon,
    c.contacts_count_12_mon,
    s.score_churn,
    s.clv_usd,
    s.segmento,
    s.acao,
    s.data_execucao
from public.churn_scores s
join public.clientes c using (id_cliente)
where s.data_execucao = (
    select max(s2.data_execucao) from public.churn_scores s2
    where s2.id_cliente = s.id_cliente
);

-- ---------------------------------------------------------------------------
-- Segurança: service_role grava/ler tudo; anon só lê a view (dashboard).
-- Ajuste conforme o nível de exposição desejado para o dashboard.
-- ---------------------------------------------------------------------------

alter table public.clientes enable row level security;
alter table public.churn_scores enable row level security;

drop policy if exists "service_read_clientes" on public.clientes;
create policy "service_read_clientes" on public.clientes
    for select using (auth.role() = 'service_role');

drop policy if exists "service_write_clientes" on public.clientes;
create policy "service_write_clientes" on public.clientes
    for all using (auth.role() = 'service_role') with check (auth.role() = 'service_role');

drop policy if exists "service_all_scores" on public.churn_scores;
create policy "service_all_scores" on public.churn_scores
    for all using (auth.role() = 'service_role') with check (auth.role() = 'service_role');

-- Dashboard lê apenas os resultados (não os dados completos dos clientes).
grant select on public.v_priorizados to anon;