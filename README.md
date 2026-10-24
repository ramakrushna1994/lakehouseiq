# LakehouseIQ

Natural-language questions in, governed SQL and a verified answer out, on Databricks.

LakehouseIQ turns plain-English business questions into Spark SQL over **Unity Catalog metric views**, so
"revenue" or "headcount" means exactly one thing everywhere. Every query passes **code-verified guardrails**
(a character-level SQL scanner, not prompt politeness), is scored by an **independent judge model** from a
different model family than the generator, and the whole system is measured by an **eval harness benchmarked
against Databricks Genie** on the same questions and the same data.

Built for Databricks Free Edition: serverless compute only, one SQL warehouse, Foundation Model APIs.

## How it works

```
question -> prompt (metric view context + live schema + enum values + saved corrections)
         -> generator model (Foundation Model API)
         -> guardrails (read-only SELECT, no system/app schemas, comment/literal aware)
         -> SQL warehouse (one automatic repair attempt on error)
         -> scoring (schema + value compliance computed in code, metric-view misuse check)
         -> judge model (relevance and SQL quality, different model family)
         -> verdict: GOOD / NEEDS REVIEW / RISKY / UNSCORED, plus a short summary
```

* `app/lakehouseiq/pipeline.py` is the only orchestration; the Streamlit app and the eval harness both call it,
  so the accuracy the eval reports is the accuracy users get.
* The **semantic registry** (`app/semantic/*.yaml`, 10 metric views) is the single source of truth: notebook 04
  renders it into `CREATE VIEW ... WITH METRICS`, and the prompt builder renders the same files into model context.
* Pure modules (`guardrails`, `parsing`, `scoring`, `semantic`, `prompts`, `eval/compare`) import with only the
  standard library, pyyaml and pandas, so the tests run offline for free.

## Data

Three synthetic domains seeded deterministically from a fixed `as_of_date` (no `current_date()` in the seed code,
so the data and every date-relative answer are reproducible):

| Schema | Content |
| --- | --- |
| `sales` | customers, products, stores, channels, promotions, orders, order items, returns, inventory |
| `hr` | departments, employees, payroll, performance reviews |
| `finance` | GL accounts, cost centers, budgets, expenses, invoices |
| `app` | `conversation_history`, `query_corrections`, `eval_results` |

Data integrity is enforced, not hoped for: each seed notebook ships assertion functions (ship date never before
order date, line prices equal product prices, order subtotals equal the sum of their lines, returns reference real
order lines, net pay includes overtime, one budget row per grain, invoice customer matches the order's customer,
and more). The tests run the seed builders in a local Spark session and prove the checks fire on corrupted data.

## Deploy

Prerequisites: a Databricks Free Edition workspace, the [Databricks CLI](https://docs.databricks.com/dev-tools/cli/install.html)
authenticated with a profile named `free_edition`, and the default `Serverless Starter Warehouse`.

```bash
databricks bundle validate -t dev -p free_edition
databricks bundle deploy -t dev -p free_edition      # creates the jobs, the app and its service principal

# 1. schemas, app tables, seed data, metric views, then grants for the app (a few minutes)
databricks bundle run lakehouseiq_setup -t dev -p free_edition

# 2. start the app
databricks bundle run lakehouseiq -t dev -p free_edition
```

The setup job ends by granting the app's service principal its minimal access. `lakehouseiq_grant_app` re-applies
just the grants (CI runs it after every deploy); on a workspace that hasn't been seeded yet it skips the missing
objects instead of failing.

Bundle variables (override with `--var name=value`): `catalog` (default `workspace`), `as_of_date`
(default `2026-06-30`; the fixed "today" for the seeded data and for date-relative logic such as overdue invoices),
`generator_endpoint`, `judge_endpoint`, `genie_space_id`. The warehouse is looked up by the name
`Serverless Starter Warehouse`.

## Evaluate against Genie

`eval/questions.yaml` holds 30 questions (10 per domain) with reference SQL.

1. Create a Genie space manually (Genie spaces are not part of the bundle): in the workspace, **Genie > New**,
   add the tables and metric views under `sales`, `hr` and `finance`, and select the same SQL warehouse.
2. Copy the space id from its URL and pass it: `databricks bundle run lakehouseiq_eval -t dev -p free_edition`
   after deploying with `--var genie_space_id=<id>`, or run locally with `GENIE_SPACE_ID` set.
3. Results land in `<catalog>.app.eval_results` and in an MLflow run (`exec_accuracy_assistant`,
   `exec_accuracy_genie`, `blocked_rate`, `repair_rate`, `p50_latency_ms`).

A question counts as correct when the **result set** matches the reference result (column names ignored,
numbers compared with tolerance, extra columns allowed, row order checked only when the question demands it).
Genie's SQL is validated by the same guardrails and run on the same warehouse. Re-running with the same `run_id`
resumes, skipping questions already stored.

## Security model

Layered, with the enforced boundary in Unity Catalog:

* Regex/scanner guardrails reject anything but one read-only `SELECT` and block `system`, `information_schema`
  and the `app` schema.
* `notebooks/05_grant_app.py` gives the app's service principal `SELECT` on `sales`, `hr`, `finance`, and
  `SELECT, MODIFY` on exactly `app.conversation_history` and `app.query_corrections`. Nothing on `app.eval_results`.
* No secrets in code or `app.yaml`; CI uses `DATABRICKS_HOST` and `DATABRICKS_TOKEN` repository secrets.
* User-supplied text reaches prompts inside `<<< >>>` delimiters with an instruction to treat it as data.

**Residual risk:** generated SQL executes as the same principal that can write the two app tables. The guardrails
block references to the `app` schema, but they are the only thing preventing a model-written statement from
touching those tables. A second, write-only principal would remove that dependency.

## Develop

```bash
pip install -r requirements-dev.txt
ruff check .
pytest -q
```

Repository layout: `app/` (Streamlit app and `lakehouseiq` package), `notebooks/` (Databricks source-format
notebooks 00-06), `eval/` (questions, comparison, runner), `tests/`, `docs/ARCHITECTURE.md`, `.github/workflows/`.
CI runs the tests on every push; `deploy.yml` deploys on pushes to `main`; `setup-data.yml` is manual and gated
by the `production` environment.

## Honest limits

* The reference SQL in `eval/questions.yaml` requires human review before any accuracy number is published.
* Free Edition has a daily fair-use quota; exceeding it shuts compute off for the day and can interrupt eval
  runs. Resume with the same `run_id`.
* Free Edition is not for commercial use.
* MLflow tracing is recorded for eval runs (they run as you). The app itself does not trace yet: its service
  principal has no experiment permission and no `MLFLOW_EXPERIMENT_ID`, so tracing is a silent no-op there.
* `sample_questions.txt` is a broader set of free-form questions for manual exploration; it is not scored.

## License

Apache-2.0, see `LICENSE`.
