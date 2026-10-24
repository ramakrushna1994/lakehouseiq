# LakehouseIQ Architecture

LakehouseIQ answers business questions in plain English by writing Spark SQL against a governed semantic
layer, checking that SQL in code before it runs, and having an independent model grade it afterwards. An eval
harness measures the whole system against Databricks Genie on the same questions and data.

## Design principles

1. **One definition per metric.** Revenue, headcount and budget variance are defined once, as Unity Catalog
   metric views. The model queries those definitions rather than re-deriving them with joins, which removes the
   most common source of wrong answers: join fan-out.
2. **Enforce in code, not in the prompt.** Prompts guide the model; code decides what runs. Guardrails parse
   the SQL; schema and value compliance are computed, not asked for.
3. **The grader is not the author.** The judge model comes from a different model family than the generator,
   so the two don't share the same blind spots.
4. **Measure, don't assert.** Accuracy is a number from the eval harness, reproducible because the data is
   deterministic.
5. **Grants are the real boundary.** Every in-app check is a first layer; Unity Catalog permissions are the one
   that can't be talked around.

## Components

| Layer | Implementation |
| --- | --- |
| Data | Delta tables in `sales`, `hr`, `finance`, seeded deterministically from a fixed `as_of_date` |
| Semantic layer | 10 metric views (`version: 1.1`) defined in `app/semantic/*.yaml`, plus two cross-fact views |
| Models | Foundation Model APIs; generator and judge are bundle variables |
| Execution | Serverless SQL warehouse via `databricks-sql-connector`, authenticated as the app's service principal |
| UI | Streamlit on Databricks Apps |
| Memory and audit | `app.conversation_history`, `app.query_corrections` (Delta, written with named parameters only) |
| Evaluation | `eval/` harness, results in `app.eval_results` and MLflow |
| Delivery | Databricks Asset Bundle + GitHub Actions |

## Request flow

```
question
  │
  ├─ context: metric-view registry + live schema + sampled enum values
  │           + relevant past corrections + last two turns
  ▼
generator model ──► JSON {sql, view}
  ▼
guardrails (character-level scanner)
  • exactly one SELECT / WITH statement
  • no DDL, DML or side-effecting Databricks commands
  • no reads from system, information_schema or app
  • Spark quoting rules: '…' and "…" literals, backslash escapes, `backtick` identifiers
  ▼
row cap (LIMIT appended unless the outermost clause already limits)
  ▼
SQL warehouse ── error? ──► one repair call ──► guardrails ──► warehouse
  ▼
scoring
  • schema compliance  — every identifier checked against the live schema and registry
  • value compliance   — filter literals checked against sampled column values
  • metric-view misuse — measures must go through MEASURE(); no SELECT * on a metric view
  • judge model        — relevance and SQL quality
  ▼
verdict: GOOD · NEEDS REVIEW · RISKY · UNSCORED  (+ BLOCKED / ERROR)
  ▼
summary (only when rows came back) ──► UI, history table
```

`app/lakehouseiq/pipeline.py` is the only orchestration. The Streamlit app and the eval harness both call
`answer_question()`, so the eval measures exactly what users get.

## Semantic registry

Each YAML file in `app/semantic/` holds a metric view's name, target schema, a routing description, and the
metric-view spec. The same files are read in two places:

- `notebooks/04_semantic_layer.py` substitutes `${catalog}` (and `${as_of_date}` for invoice aging) and issues
  `CREATE OR REPLACE VIEW … WITH METRICS LANGUAGE YAML`.
- `semantic.py` renders them into the model's context and into the set of known identifiers used for scoring.

Changing a metric therefore changes the view, the prompt and the scorer together.

Metric views are single-fact. Questions that combine facts, such as return rate (returns ÷ units sold), use
pre-aggregated cross-fact views (`v_product_performance`, `v_customer_segments`) that aggregate each fact in its
own CTE before joining.

## Data model and reproducibility

Every date is derived from `as_of_date` (default `2026-06-30`) and every random value comes from a fixed seed on
a single partition. Reruns produce identical tables, so reference answers in the eval never drift.

Integrity rules are asserted by the seed notebooks, and the tests prove the assertions fire on corrupted data:

- ship date ≥ order date; delivery date ≥ ship date; cancelled and processing orders have neither
- order-line price equals the product's price; order subtotal equals the sum of its lines
- every order has at least one line
- returns reference a real order line, inherit its order and product, and refund no more than the line total
- payroll department matches the employee's department; net pay = base + bonus + overtime − deductions
- one budget row per department × account × month
- an invoice's customer is the customer on its order

## Security model

| Principal | Access |
| --- | --- |
| App service principal | `SELECT` on `sales`, `hr`, `finance`; `SELECT, MODIFY` on `app.conversation_history` and `app.query_corrections` only |
| Workspace user (you) | Owner; runs setup and eval jobs |
| CI | Personal access token in GitHub secrets, used only for `bundle deploy` / `bundle run` |

User text reaches prompts inside `<<< >>>` delimiters with an instruction to treat it as data. Writes use named
SQL parameters, never string interpolation.

Known residual risk: generated SQL runs as the same principal that can write the two app tables. The guardrails
block any reference to the `app` schema; a separate write-only principal would remove the dependency on them.

## Evaluation

`eval/questions.yaml` holds 30 questions (10 per domain) with reference SQL. For each question the runner:

1. runs the assistant pipeline (no summary),
2. runs the reference SQL,
3. optionally asks a Genie space the same question and runs Genie's SQL through the same guardrails and warehouse,
4. compares result sets — values only, numbers rounded, column order and extra columns tolerated, row order
   checked only when the question requires it.

Metrics: `exec_accuracy_assistant`, `exec_accuracy_genie`, `blocked_rate`, `repair_rate`, `p50_latency_ms`.
Runs resume by `run_id`, which matters under Free Edition's daily quota. Swapping `generator_endpoint` and
re-running is how model choices are compared.

## Deliberate V1 exclusions

- **Streaming ingestion.** Free Edition restricts outbound network access, so an external Kafka cluster can't be
  reached from serverless compute. Batch seeding keeps eval answers fixed.
- **Vector-search routing / multi-agent router.** Ten metric views fit in one prompt; routing would add a failure
  point without improving accuracy at this scale.
- **App-side tracing.** Eval runs trace to MLflow; the app's service principal has no experiment permission yet.
