# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Semantic layer
# MAGIC
# MAGIC Creates the governed definitions the assistant queries:
# MAGIC
# MAGIC 1. **Metric views** - one per file in `app/semantic/*.yaml`. The app reads the very same files to build its
# MAGIC    prompt, so the prompt can never drift from what exists in the catalog. Requires DBR 17.3+ / a current SQL
# MAGIC    warehouse (serverless is fine).
# MAGIC 2. **Cross-fact views** - plain views for questions that combine facts (return rate = returns / units sold,
# MAGIC    customer lifetime value with returns). Each fact is aggregated in its own CTE and joined at the end, so
# MAGIC    one-to-many joins never fan out and inflate sums.
# MAGIC
# MAGIC Placeholders substituted in the YAML: `${catalog}` and `${as_of_date}` (used only by `mv_invoices` to age
# MAGIC invoices against the data's reference date rather than the wall clock).
# MAGIC
# MAGIC Run after notebooks 01-03.

# COMMAND ----------

# MAGIC %pip install -q pyyaml

# COMMAND ----------

dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

PRODUCT_COLUMNS = {
    "product_id": "Primary key",
    "product_name": "Product name",
    "category": "Product category",
    "subcategory": "Product subcategory",
    "unit_price": "List price, USD",
    "unit_cost": "Unit cost, USD",
    "unit_margin": "unit_price - unit_cost, USD",
    "margin_pct": "Margin as a percent of price (0-100)",
    "times_ordered": "Order lines containing the product (non-cancelled orders)",
    "total_units_sold": "Units sold in non-cancelled orders",
    "gross_revenue": "quantity x list price before line discounts, USD",
    "net_revenue": "Line revenue after line discounts, USD",
    "total_returns": "Number of returns of the product",
    "return_rate_pct": "total_returns / total_units_sold x 100; null when nothing was sold",
}

CUSTOMER_COLUMNS = {
    "customer_id": "Primary key",
    "customer_name": "First and last name",
    "region": "Sales region",
    "state": "US state code",
    "segment": "Enterprise, Mid-Market or SMB",
    "acquisition_channel": "Channel the customer signed up through",
    "total_orders": "Non-cancelled orders",
    "lifetime_value": "Sum of line revenue over non-cancelled orders, USD",
    "avg_order_value": "Average order total, USD",
    "last_order_date": "Most recent non-cancelled order",
    "days_since_last_order": "Days from last_order_date to the data's as-of date (not the calendar date)",
    "total_returns": "Returns across all of the customer's orders",
}


def _columns_clause(columns: dict) -> str:
    return ", ".join(f"{name} COMMENT '{text.replace(chr(39), chr(39) * 2)}'" for name, text in columns.items())


def cross_fact_view_sql(catalog: str, as_of_date: str) -> dict:
    """DDL for the plain cross-fact views, keyed by view name (pure: no Spark needed)."""
    s = f"{catalog}.sales"
    product = f"""
CREATE OR REPLACE VIEW {s}.v_product_performance ({_columns_clause(PRODUCT_COLUMNS)})
COMMENT 'One row per product: units sold, revenue, margin and returns. Sales and returns are aggregated separately so joins never fan out.'
AS
WITH oi_agg AS (
  SELECT oi.product_id,
         COUNT(DISTINCT oi.order_item_id) AS times_ordered,
         SUM(oi.quantity)                 AS total_units_sold,
         SUM(oi.quantity * oi.unit_price) AS gross_revenue,
         SUM(oi.line_total)               AS net_revenue
  FROM {s}.order_items oi
  JOIN {s}.orders o ON oi.order_id = o.order_id
  WHERE o.status <> 'Cancelled'
  GROUP BY oi.product_id
),
ret_agg AS (
  SELECT product_id, COUNT(DISTINCT return_id) AS total_returns
  FROM {s}.returns
  GROUP BY product_id
)
SELECT p.product_id, p.product_name, p.category, p.subcategory, p.unit_price, p.unit_cost,
       p.unit_price - p.unit_cost AS unit_margin,
       ROUND((p.unit_price - p.unit_cost) / NULLIF(p.unit_price, 0) * 100, 2) AS margin_pct,
       COALESCE(oi.times_ordered, 0)    AS times_ordered,
       COALESCE(oi.total_units_sold, 0) AS total_units_sold,
       COALESCE(oi.gross_revenue, 0)    AS gross_revenue,
       COALESCE(oi.net_revenue, 0)      AS net_revenue,
       COALESCE(r.total_returns, 0)     AS total_returns,
       ROUND(COALESCE(r.total_returns, 0) / NULLIF(oi.total_units_sold, 0) * 100, 2) AS return_rate_pct
FROM {s}.products p
LEFT JOIN oi_agg oi ON p.product_id = oi.product_id
LEFT JOIN ret_agg r ON p.product_id = r.product_id
"""
    customer = f"""
CREATE OR REPLACE VIEW {s}.v_customer_segments ({_columns_clause(CUSTOMER_COLUMNS)})
COMMENT 'One row per customer: lifetime value, orders, returns and recency relative to the data as-of date. Facts are aggregated separately so joins never fan out.'
AS
WITH order_rev AS (
  SELECT o.customer_id,
         COUNT(DISTINCT o.order_id) AS total_orders,
         SUM(oi.line_total)         AS lifetime_value,
         MAX(o.order_date)          AS last_order_date
  FROM {s}.orders o
  JOIN {s}.order_items oi ON o.order_id = oi.order_id
  WHERE o.status <> 'Cancelled'
  GROUP BY o.customer_id
),
order_aov AS (
  SELECT customer_id, AVG(total_amount) AS avg_order_value
  FROM {s}.orders
  WHERE status <> 'Cancelled'
  GROUP BY customer_id
),
cust_returns AS (
  SELECT o.customer_id, COUNT(DISTINCT r.return_id) AS total_returns
  FROM {s}.orders o
  JOIN {s}.returns r ON o.order_id = r.order_id
  GROUP BY o.customer_id
)
SELECT c.customer_id,
       c.first_name || ' ' || c.last_name AS customer_name,
       c.region, c.state, c.segment,
       sc.channel_name AS acquisition_channel,
       COALESCE(orv.total_orders, 0)   AS total_orders,
       COALESCE(orv.lifetime_value, 0) AS lifetime_value,
       aov.avg_order_value             AS avg_order_value,
       orv.last_order_date             AS last_order_date,
       datediff(DATE'{as_of_date}', orv.last_order_date) AS days_since_last_order,
       COALESCE(cr.total_returns, 0)   AS total_returns
FROM {s}.customers c
LEFT JOIN order_rev orv     ON c.customer_id = orv.customer_id
LEFT JOIN order_aov aov     ON c.customer_id = aov.customer_id
LEFT JOIN cust_returns cr   ON c.customer_id = cr.customer_id
LEFT JOIN {s}.sales_channels sc ON c.channel_id = sc.channel_id
"""
    return {"v_product_performance": product.strip(), "v_customer_segments": customer.strip()}


def metric_view_sql(view_fqn: str, spec_yaml: str) -> str:
    """CREATE statement for one metric view. ``$$`` quoting keeps the YAML verbatim."""
    if "$$" in spec_yaml:
        raise ValueError("metric view YAML must not contain '$$'")
    return f"CREATE OR REPLACE VIEW {view_fqn} WITH METRICS LANGUAGE YAML AS $$\n{spec_yaml}$$"


def smoke_query(view_fqn: str, measures: list) -> str:
    """One query that evaluates every measure of a view, to prove the definition compiles and runs."""
    return f"SELECT {', '.join(f'MEASURE({m}) AS {m}' for m in measures)} FROM {view_fqn}"

# COMMAND ----------

if __name__ == "__main__":
    import os
    import sys

    dbutils.widgets.text("catalog", "workspace")  # noqa: F821
    dbutils.widgets.text("as_of_date", "2026-06-30")  # noqa: F821
    dbutils.widgets.text("semantic_dir", "")  # noqa: F821  # <bundle files root>/app/semantic
    catalog = dbutils.widgets.get("catalog")  # noqa: F821
    as_of_date = dbutils.widgets.get("as_of_date")  # noqa: F821
    semantic_dir = dbutils.widgets.get("semantic_dir")  # noqa: F821
    if not semantic_dir:
        raise ValueError("pass semantic_dir (the job sets it to ${workspace.file_path}/app/semantic)")

    # The registry module lives next to the YAML files: <app>/lakehouseiq/semantic.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(semantic_dir)))
    from lakehouseiq.semantic import fqn, load_registry, render_spec_yaml

    registry = load_registry(semantic_dir)
    print(f"{len(registry)} metric views found in {semantic_dir}")

# COMMAND ----------

if __name__ == "__main__":
    for schema in sorted({mv.schema for mv in registry}):
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")  # noqa: F821

    for mv in registry:
        name = fqn(catalog, mv)
        spark.sql(metric_view_sql(name, render_spec_yaml(mv, catalog, as_of_date)))  # noqa: F821
        print("created metric view", name)

# COMMAND ----------

if __name__ == "__main__":
    for name, ddl in cross_fact_view_sql(catalog, as_of_date).items():
        spark.sql(ddl)  # noqa: F821
        print("created view", f"{catalog}.sales.{name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test
# MAGIC
# MAGIC Evaluates every measure of every metric view and fails loudly if any definition does not compile or run.

# COMMAND ----------

if __name__ == "__main__":
    failures = {}
    for mv in registry:
        name = fqn(catalog, mv)
        try:
            row = spark.sql(smoke_query(name, mv.measure_names)).first()  # noqa: F821
            print(f"ok  {name}: " + ", ".join(f"{m}={row[m]}" for m in mv.measure_names[:3]) + " ...")
        except Exception as exc:
            failures[name] = str(exc)[:400]
    for view in ("v_product_performance", "v_customer_segments"):
        n = spark.table(f"{catalog}.sales.{view}").count()  # noqa: F821
        print(f"ok  {catalog}.sales.{view}: {n} rows")
    assert not failures, f"metric views that failed their smoke test: {failures}"
