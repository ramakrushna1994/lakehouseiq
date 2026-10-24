# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Grant the app's service principal read access
# MAGIC
# MAGIC The regex guardrails in the app are only the first layer. **This notebook is the enforced boundary**: the
# MAGIC app's service principal can
# MAGIC
# MAGIC * `SELECT` on `sales`, `hr` and `finance` (tables, views and metric views), and nothing else there
# MAGIC * `SELECT` and `MODIFY` on exactly two tables, `app.conversation_history` and `app.query_corrections`
# MAGIC * nothing on `app.eval_results`
# MAGIC
# MAGIC Residual risk: generated SQL runs as the same principal that can write the two app tables. The guardrails
# MAGIC block references to the `app` schema, but a second, write-only principal would remove the dependency on them.
# MAGIC
# MAGIC Idempotent: `GRANT` on an existing grant is a no-op. Run after the app exists (it needs the principal) and
# MAGIC after notebook 00.

# COMMAND ----------

import re

DATA_SCHEMAS = ["sales", "hr", "finance"]
WRITABLE_APP_TABLES = ["conversation_history", "query_corrections"]
_SAFE_PRINCIPAL = re.compile(r"^[A-Za-z0-9_.@-]+$")
_SAFE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def grant_statements(catalog: str, principal: str) -> list:
    """The exact GRANT statements for ``principal`` (pure, so the security model is reviewable and tested)."""
    if not _SAFE_IDENT.match(catalog):
        raise ValueError(f"catalog must be a simple identifier, got {catalog!r}")
    if not _SAFE_PRINCIPAL.match(principal):
        raise ValueError(f"unexpected characters in principal {principal!r}")
    p = f"`{principal}`"
    stmts = [f"GRANT USE CATALOG ON CATALOG {catalog} TO {p}"]
    for schema in DATA_SCHEMAS:
        stmts.append(f"GRANT USE SCHEMA, SELECT ON SCHEMA {catalog}.{schema} TO {p}")
    stmts.append(f"GRANT USE SCHEMA ON SCHEMA {catalog}.app TO {p}")
    for table in WRITABLE_APP_TABLES:
        stmts.append(f"GRANT SELECT, MODIFY ON TABLE {catalog}.app.{table} TO {p}")
    return stmts

# COMMAND ----------

if __name__ == "__main__":
    from databricks.sdk import WorkspaceClient

    dbutils.widgets.text("catalog", "workspace")
    dbutils.widgets.text("app_name", "lakehouseiq")
    catalog = dbutils.widgets.get("catalog")
    app_name = dbutils.widgets.get("app_name")

    app = WorkspaceClient().apps.get(app_name)
    principal = app.service_principal_client_id
    if not principal:
        raise RuntimeError(f"app {app_name!r} has no service principal yet; deploy the app first")
    print(f"App {app_name} runs as {principal}")

    # On the very first deploy the schemas don't exist yet (the setup job creates them and
    # re-runs this notebook as its last task), so a missing object is skipped, not fatal.
    skipped = 0
    for stmt in grant_statements(catalog, principal):
        try:
            spark.sql(stmt)
            print(stmt)
        except Exception as exc:  # AnalysisException; message carries the error class
            if "NOT_FOUND" in str(exc) or "does not exist" in str(exc):
                skipped += 1
                print(f"SKIPPED (object not created yet): {stmt}")
            else:
                raise
    if skipped:
        print(f"{skipped} grant(s) skipped; run the setup job to create the data, which re-applies grants.")

# COMMAND ----------

if __name__ == "__main__" and not skipped:
    spark.sql(f"SHOW GRANTS `{principal}` ON SCHEMA {catalog}.sales").show(truncate=False)
