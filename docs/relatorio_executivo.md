# Relatório Executivo — Previsão de Churn em Cartão de Crédito

> Resumo executivo (1–2 páginas) dos resultados técnicos traduzidos em decisão de negócio.
> Números de modelo referem-se à **holdout de 20%** (clientes não vistos no treino).

---

## Contexto

O banco perde, em média, **16% dos clientes de cartão de crédito por ciclo**. Cada perda envolve
custo de aquisição e a **receita futura do relacionamento** (anuidade, juros e interchange). O objetivo
do projeto é **antecipar o cancelamento**, explicar o motivo e **priorizar o orçamento de retenção**
em quem de fato compensa agir — não em "todos os que estão em risco".

## O que foi construído

Modelo de churn (XGBoost) que trabalha em lote agendado sobre a base de clientes:

```
clientes → [engenharia de features + modelo + CLV + segmentação] → priorização → time humano ou automação
```

| Etapa | Decisão |
|---|---|
| Sinais mais fortes | inatividade (meses), contatos com o SAC, queda de transações Q4×Q1 e utilização do limite |
| Valor do cliente | CLV ≈ lucro anual × ~3,79 (5 anos a 10% de desconto) |
| Quando agir | quando **lift·p·CLV > custo da ação** (regra por cliente) |
| O que fazer | 4 segmentos risco × valor com ação definida e ROI estimado |

## Resultados de negócio

### 1) Precisão na retenção

Na holdout, o modelo atinge **PR-AUC 0,97**. No corte operacional recomendado (**0,51**), a precisão é
**0,90** e o recall **0,90**: de cada 100 clientes acionados por risco, ~90 realmente cancelam, e a operação
enxerga ~90% dos que vão sair. Ampliando a base para recall ~0,94 exige baixar o corte a 0,30 (precisão 0,85).

### 2) Decisão de alocação — onde gastar

Comparação de políticas na holdout (ação personalizada, custo $80, lift 15%):

| Política | Clientes acionados | Lucro esperado |
|---|---|---|
| Não fazer nada | 0 | $0 |
| Corte fixo (threshold ótimo 0,51) | 246 | **$19,0k** |
| **Regra por-cliente** (`0,15·p·CLV > $80`) | 246 | **$20,7k** |

> Conclusão: **não** se deve reter "todos os em risco". O corte ótimo global fica ≈ **0,51** (perto do 0,5,
> porque o maior ganho vem de considerar o **valor** do cliente — por isso a regra por-cliente é
> superior: ela captura a heterogeneidade de CLV que um corte fixo ignora). O critério decisivo é
> **risco × valor**, não apenas risco.

### 3) Segmentação da base (10.127 clientes)

| Segmento | Clientes | Ação | ROI estimado da campanha |
|---|---|---|---|
| Retenção prioritária | 476 | contato personalizado | **2,1×** (retorno $119,3k / custo $38,1k) |
| Ação automatizada | 1.166 | oferta automática (e-mail/app) | **5,4×** (retorno $37,6k / custo $5,8k) |
| Monitorar + upsell | 4.588 | não gastar; expandir relação | — |
| Monitorar | 3.897 | sem ação | — |

Os segmentos de retenção concentram **~$1,6 milhão em CLV em risco**. Com o ROI positivo nos dois
segmentos de ação, o orçamento investido se paga e sobra receita protegida.

### 4) Explicabilidade (por quê)

Globalmente, o modelo aponta: **abandono de uso + contatos ao SAC + subutilização do limite** como
principais vetores. Em nível individual, cada cliente em risco recebe um **laudo** ("motivos que
empurram vs. motivos que protegem") disponível no dashboard — o que permite personalizar a oferta
(ex.: oferta de reativação para o ocioso; tratamento de cobrança para o insatisfeito).

## Premissas financeiras (configuráveis em `src/business.py`)

- Margem anual sobre transações: **4%**; juros efetivo sobre saldo rotativo: **12%**;
- Anuidade: Blue $50 · Silver $80 · Gold $120 · Platinum $250;
- CLV: 5 anos, taxa de desconto **10%**;
- Custo da ação: **$80** (personalizada) / **$5** (automatizada);
- Lift de retenção: **15%** (personalizada) / **5%** (automatizada) — valores típicos reportados para
  campanhas de retenção (sensibilidade testada no notebook `05`).

## Contextualização Brasil (referência)

Ao converter a 5,2 R$/US$: o lucro esperado da política ótima na holdout equivale a **~R$ 107 mil**
em valor protegido; a ação personalizada custaria ~R$ 415 por cliente. Em mercado local, a concorrência
de bancos digitais (portabilidade, cartão sem anuidade) amplifica o risco de churn — os mesmos sinais
(inatividade e contatos) continuam os melhores preditores e devem ser combinados com **saldo de conta
e uso de Pix** quando disponíveis.

## Recomendações operacionais

1. **Rodar o scoring semanal** (lote agendado no n8n) e atualizar `churn_scores`;
2. **Time humano** entra só nos `retencao_prioritaria` (476 ≈ 4,7% da base);
3. **Automação** dispara oferta de baixo custo para os `acao_automatizada`;
4. **Dashboard** como única fonte da lista priorizada com motivo (SHAP) e ROI;
5. **Revisar semestralmente** premissas de custo/lift com o resultado das campanhas reais
   (feedback para recalibrar o modelo).

## Limitações

- Dataset de banco norte-americano e snapshot (sem histórico temporal); churn é o rótulo fornecido;
- CLV e lift são premissas paramétricas, não estimativas de experimentos;
- Valores em US$ (conversão BRL apenas ilustrativa).

*Documentação técnica completa: `README.md`. Reprodução: `src/train.py`, notebooks `01–05`.*