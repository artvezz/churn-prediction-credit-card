"""Carrega o CSV raw (BankChurners) na tabela `clientes` do Supabase via REST.

Premissas:
- o schema.sql das tabelas já foi aplicado no projeto;
- SUPABASE_URL e SUPABASE_SERVICE_KEY estão no .env.
- o CSV usa camelCase (colunas do Kaggle); o script mapeia para o snake_case
  do schema (remove colunas de leak do target).

Uso:
    pip install python-dotenv   # ou exporte SUPABASE_URL / SUPABASE_SERVICE_KEY
    python supabase/load_clientes.py --csv data/raw/BankChurners.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

COLUMN_MAP = {
    "CLIENTNUM": "id_cliente",
    "Customer_Age": "customer_age",
    "Gender": "gender",
    "Dependent_count": "dependent_count",
    "Education_Level": "education_level",
    "Marital_Status": "marital_status",
    "Income_Category": "income_category",
    "Card_Category": "card_category",
    "Months_on_book": "months_on_book",
    "Total_Relationship_Count": "total_relationship_count",
    "Months_Inactive_12_mon": "months_inactive_12_mon",
    "Contacts_Count_12_mon": "contacts_count_12_mon",
    "Credit_Limit": "credit_limit",
    "Total_Revolving_Bal": "total_revolving_bal",
    "Avg_Open_To_Buy": "avg_open_to_buy",
    "Total_Amt_Chng_Q4_Q1": "total_amt_chng_q4_q1",
    "Total_Trans_Amt": "total_trans_amt",
    "Total_Trans_Ct": "total_trans_ct",
    "Total_Ct_Chng_Q4_Q1": "total_ct_chng_q4_q1",
    "Avg_Utilization_Ratio": "avg_utilization_ratio",
    "Attrition_Flag": "attrition_flag",
}

LEAK_COLUMNS = [
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]


def env(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        sys.exit(f"Faltando variável de ambiente: {key}")
    return val


def main() -> None:
    import csv
    import time

    try:
        from dotenv import load_dotenv

        load_dotenv(str(__import__("pathlib").Path(__file__).resolve().parent.parent / ".env"))
    except ImportError:  # python-dotenv opcional
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/BankChurners.csv")
    ap.add_argument("--chunk", type=int, default=500)
    args = ap.parse_args()

    url = env("SUPABASE_URL").rstrip("/")
    key = env("SUPABASE_SERVICE_KEY")

    rows = []
    with open(args.csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {COLUMN_MAP[k]: v for k, v in raw.items() if k in COLUMN_MAP}
            row["attrition_flag"] = row.get("attrition_flag", "Existing Customer").strip()
            rows.append(row)

    total = len(rows)
    print(f"Linhas a carregar: {total}")

    for i in range(0, total, args.chunk):
        batch = rows[i : i + args.chunk]
        req = urllib.request.Request(
            f"{url}/rest/v1/clientes",
            data=json.dumps(batch).encode(),
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates",
            },
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            status = resp.status
        time.sleep(0.3)
        print(f"  enviado {i + len(batch)}/{total} (HTTP {status})")
    print("Concluído.")


if __name__ == "__main__":
    main()