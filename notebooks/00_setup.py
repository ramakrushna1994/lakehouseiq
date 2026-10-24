# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Setup
# MAGIC
# MAGIC Creates the schemas and the assistant's own Delta tables. Safe to re-run: nothing here drops data.
# MAGIC
# MAGIC | Schema | Purpose |
# MAGIC |---|---|
# MAGIC | `sales`, `hr`, `finance` | Business data the assistant answers questions about |
# MAGIC | `app` | The assistant's own tables: conversation history, corrections, eval results |
# MAGIC
# MAGIC The catalog is the `catalog` widget (bundle variable; `workspace` on Free Edition).

# COMMAND ----------

SCHEMAS = ["sales", "hr", "finance", "app"]

APP_TABLES = {
    "conversation_history": {
        "comment": "One row per question asked in the app: SQL generated, verdict, latency and endpoints used.",
        "ddl": """
            session_id STRING, turn_id INT, ts TIMESTAMP, user_email STRING, question STRING,
            generated_sql STRING, source_view STRING, verdict STRING, row_count INT, error STRING,
            latency_ms BIGINT, generator_endpoint STRING, judge_endpoint STRING
        """,
    },
    "query_corrections": {
        "comment": "Corrections submitted by users via 'Report a mistake'; similar questions see them in the prompt.",
        "ddl": """
            correction_id STRING, ts TIMESTAMP, user_email STRING, question STRING, bad_sql STRING,
            corrected_sql STRING, reason STRING, bad_values STRING
        """,
    },
    "eval_results": {
        "comment": "One row per question per system (assistant or genie) per eval run.",
        "ddl": """
            run_id STRING, ts TIMESTAMP, system STRING, question_id STRING, question STRING,
            generated_sql STRING, exec_match BOOLEAN, judge_relevance INT, verdict STRING,
            latency_ms BIGINT, error STRING
        """,
    },
}


def setup_statements(catalog: str) -> list:
    """The DDL this notebook runs, as a list of SQL strings (pure, so it can be reviewed and tested)."""
    stmts = [f"CREATE SCHEMA IF NOT EXISTS {catalog}.{s}" for s in SCHEMAS]
    for name, spec in APP_TABLES.items():
        comment = spec["comment"].replace("'", "''")
        stmts.append(
            f"CREATE TABLE IF NOT EXISTS {catalog}.app.{name} ({' '.join(spec['ddl'].split())}) "
            f"USING DELTA COMMENT '{comment}'"
        )
    return stmts

# COMMAND ----------

if __name__ == "__main__":
    dbutils.widgets.text("catalog", "workspace")
    catalog = dbutils.widgets.get("catalog")
    if not catalog.replace("_", "").isalnum():
        raise ValueError(f"catalog must be a simple identifier, got {catalog!r}")

    try:
        spark.sql(f"USE CATALOG {catalog}")
    except Exception:
        # Not present: try to create it (not allowed on Free Edition, where `workspace` already exists).
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
        spark.sql(f"USE CATALOG {catalog}")

    for stmt in setup_statements(catalog):
        spark.sql(stmt)
        print(stmt.split("(")[0].strip())

# COMMAND ----------

if __name__ == "__main__":
    for row in spark.sql(f"SHOW TABLES IN {catalog}.app").collect():
        n = spark.table(f"{catalog}.app.{row.tableName}").count()
        print(f"{catalog}.app.{row.tableName}: {n} rows")
