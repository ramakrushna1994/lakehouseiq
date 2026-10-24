# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Evaluation
# MAGIC
# MAGIC Runs the 30 curated questions in `eval/questions.yaml` the 10 held-out v1 questions in `eval/holdout.yaml` and the 30 frozen held-out v2 questions in `eval/holdout_v2.yaml` through the assistant (and through a Genie space when
# MAGIC `genie_space_id` is set), executes each system's SQL on the SQL warehouse, and compares the **result sets**
# MAGIC against the reference SQL. Rows go to `<catalog>.app.eval_results`; metrics and the results table go to an
# MAGIC MLflow run.
# MAGIC
# MAGIC * Re-running with the same `run_id` resumes: questions already stored for that run are skipped. This matters on
# MAGIC   Free Edition, where daily quotas can interrupt a run.
# MAGIC * The reference SQL needs human review before any number from this notebook is published.
# MAGIC * `limit` > 0 evaluates only the first N questions.
# MAGIC * Dependencies (pyyaml, databricks-sdk, databricks-sql-connector, openai, mlflow) come from the job's serverless
# MAGIC   `%pip` cell below.

# COMMAND ----------

# MAGIC %pip install -q pyyaml databricks-sql-connector openai mlflow-skinny

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("as_of_date", "")
dbutils.widgets.text("warehouse_id", "")
dbutils.widgets.text("generator_endpoint", "databricks-gpt-oss-120b")
dbutils.widgets.text("judge_endpoint", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("genie_space_id", "")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("limit", "0")

# COMMAND ----------

if __name__ == "__main__":
    import os
    import sys

    get = dbutils.widgets.get
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    repo_root = os.path.dirname(os.path.dirname("/Workspace" + notebook_path))
    sys.path[:0] = [repo_root, os.path.join(repo_root, "app")]

    env = {
        "CATALOG": get("catalog"),
        "DATABRICKS_WAREHOUSE_ID": get("warehouse_id"),
        "GENERATOR_ENDPOINT": get("generator_endpoint"),
        "JUDGE_ENDPOINT": get("judge_endpoint"),
        "GENIE_SPACE_ID": get("genie_space_id"),
        "AS_OF_DATE": get("as_of_date"),
    }
    os.environ.update({k: v for k, v in env.items() if v})

    from eval import runner

    argv = ["--sleep", "2"]
    if get("run_id"):
        argv += ["--run-id", get("run_id")]
    if int(get("limit") or 0) > 0:
        argv += ["--limit", get("limit")]
    runner.main(argv)

# COMMAND ----------

if __name__ == "__main__":
    display(
        spark.sql(
            f"SELECT system, COUNT(*) AS questions, ROUND(AVG(CAST(exec_match AS INT)), 3) AS exec_accuracy, "
            f"percentile(latency_ms, 0.5) AS p50_latency_ms "
            f"FROM {get('catalog')}.app.eval_results GROUP BY system ORDER BY system"
        )
    )
