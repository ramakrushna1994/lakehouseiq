# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Seed HR data
# MAGIC
# MAGIC Creates `hr.departments`, `employees`, `job_history`, `payroll`, `performance_reviews` and `attendance`.
# MAGIC Dates come from the `as_of_date` widget (never the clock); random columns use fixed seeds.
# MAGIC
# MAGIC **Integrity rules the data guarantees** (asserted at the end):
# MAGIC * every employee was hired at least 760 days before `as_of_date`, so payroll, reviews, attendance and expenses
# MAGIC   never predate a hire
# MAGIC * `payroll.dept_id` is the employee's department and
# MAGIC   `net_pay = base_salary + bonus_amount + overtime_pay - deductions`
# MAGIC * overtime pay is derived from overtime hours at 1.5x the hourly rate (`salary / 2080`)
# MAGIC * `performance_reviews.review_period` is derived from `review_date`, ratings stay within 1.0-5.0, and nobody
# MAGIC   reviews themselves
# MAGIC * attendance days present plus leave days never exceed 5, and `leave_type` is `None` exactly when no leave
# MAGIC   was taken

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# (dept_name, annual_budget, headcount_budget, cost_center, location)
DEPARTMENTS = [
    ('Engineering', 4500000, 50, 'CC-ENG', 'San Francisco'),
    ('Sales', 3200000, 40, 'CC-SAL', 'New York'),
    ('Marketing', 2000000, 25, 'CC-MKT', 'New York'),
    ('Human Resources', 1200000, 15, 'CC-HR', 'Chicago'),
    ('Finance', 1800000, 20, 'CC-FIN', 'Chicago'),
    ('Operations', 2800000, 35, 'CC-OPS', 'San Francisco'),
    ('Customer Success', 1500000, 20, 'CC-CSX', 'Austin'),
    ('Product Management', 1000000, 12, 'CC-PM', 'San Francisco'),
    ('Legal', 800000, 8, 'CC-LEG', 'New York'),
    ('Data & Analytics', 1200000, 15, 'CC-DA', 'San Francisco'),
]

FIRST_NAMES = ["James", "Sarah", "Michael", "Emily", "Daniel", "Jessica", "Robert", "Amanda", "Chris", "Laura",
               "Kevin", "Nicole", "Ryan", "Angela", "Tom", "Samantha", "Brandon", "Diana", "Marcus", "Priya",
               "Linda", "George", "Helen", "Victor", "Catherine"]
LAST_NAMES = ["Peterson", "Chen", "Kumar", "Rodriguez", "Park", "Lee", "Singh", "Foster", "Wang", "Martinez",
              "OBrien", "Thompson", "Davis", "Kim", "Baker", "Green", "Cole", "Patel", "Hall", "Sharma"]
JOB_TITLES = ["Junior Analyst", "Analyst", "Senior Analyst", "Lead", "Manager", "Senior Manager", "Director", "VP"]
JOB_LEVELS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
FROM_TITLES = ["Junior Analyst", "Analyst", "Senior Analyst", "Lead", "Manager"]
TO_TITLES = ["Analyst", "Senior Analyst", "Lead", "Manager", "Senior Manager"]
CHANGE_REASONS = ["Promotion", "Transfer", "Restructure", "Merit Raise", "Lateral Move"]
LEAVE_TYPES = ["Annual", "Sick", "Public Holiday"]

N_EMPLOYEES, N_JOB_HISTORY, N_PAY_MONTHS, N_REVIEW_QUARTERS, N_ATT_WEEKS = 200, 300, 24, 8, 104
MIN_TENURE_DAYS = 760  # > 24 months of payroll + margin

# COMMAND ----------

# MAGIC %md
# MAGIC ## Builders

# COMMAND ----------

def _pick(r, options):
    arr = F.array(*[F.lit(o) for o in options])
    return F.element_at(arr, (F.floor(r * len(options)) + 1).cast("int"))


def _between(r, lo, hi):
    return (F.floor(r * (hi - lo + 1)) + lo).cast("int")


def _hrand(*cols, seed):
    """Uniform [0,1) from a hash: independent of partitioning, safe after joins and cross joins."""
    return F.pmod(F.xxhash64(*cols, F.lit(seed)), F.lit(1000000)) / F.lit(1000000.0)


def build_departments(spark):
    return spark.createDataFrame(
        [(i + 1, r[0], float(r[1]), r[2], r[3], r[4], i + 1) for i, r in enumerate(DEPARTMENTS)],
        "dept_id int, dept_name string, annual_budget double, headcount_budget int, cost_center string, "
        "location string, manager_id int",
    ).withColumn("annual_budget", F.col("annual_budget").cast("decimal(12,2)"))


def build_employees(spark, as_of_date):
    asof = F.lit(as_of_date).cast("date")
    c = F.col("id")
    return spark.range(1, N_EMPLOYEES + 1, 1, 1).select(
        c.cast("int").alias("emp_id"),
        _pick(((c - 1) % 25 + 0.5) / 25, FIRST_NAMES).alias("first_name"),
        _pick(((c - 1) % 20 + 0.5) / 20, LAST_NAMES).alias("last_name"),
        F.concat(F.lit("emp"), (c - 1).cast("string"), F.lit("@company.com")).alias("email"),
        _between(F.rand(1001), 1, len(DEPARTMENTS)).alias("dept_id"),
        _pick(F.rand(1002), JOB_TITLES).alias("job_title"),
        _pick(F.rand(1003), JOB_LEVELS).alias("job_level"),
        F.date_sub(asof, _between(F.rand(1004), MIN_TENURE_DAYS, 2919)).alias("hire_date"),
        F.round(F.lit(55000) + F.rand(1005) * 195000, 2).cast("decimal(10,2)").alias("salary"),
        F.round(F.lit(5) + F.rand(1006) * 25, 2).cast("decimal(5,2)").alias("bonus_pct"),
        F.when(F.rand(1007) < 0.1, "Inactive").otherwise("Active").alias("status"),
        F.when(c <= len(DEPARTMENTS), F.lit(None).cast("int")).otherwise(_between(F.rand(1008), 1, len(DEPARTMENTS)))
        .alias("manager_id"),
    )


def build_job_history(spark, as_of_date, employees):
    asof = F.lit(as_of_date).cast("date")
    c = F.col("id")
    raw = spark.range(1, N_JOB_HISTORY + 1, 1, 1).select(
        c.cast("int").alias("history_id"),
        _between(F.rand(1101), 1, N_EMPLOYEES).alias("emp_id"),
        _between(F.rand(1102), 1, len(DEPARTMENTS)).alias("from_dept_id"),
        _between(F.rand(1103), 1, len(DEPARTMENTS)).alias("to_dept_raw"),
        _pick(F.rand(1104), FROM_TITLES).alias("from_title"),
        _pick(F.rand(1105), TO_TITLES).alias("to_title"),
        F.round(F.lit(55000) + F.rand(1106) * 125000, 2).alias("from_salary"),
        F.rand(1107).alias("r_raise"),
        F.rand(1108).alias("r_when"),
        _pick(F.rand(1109), CHANGE_REASONS).alias("change_reason"),
    )
    j = raw.join(F.broadcast(employees.select("emp_id", "hire_date")), "emp_id")
    same_dept = F.col("change_reason").isin("Promotion", "Merit Raise")
    span = F.datediff(asof, F.col("hire_date")) - 30
    return j.select(
        "history_id", "emp_id", "from_dept_id",
        F.when(same_dept, F.col("from_dept_id")).otherwise(F.col("to_dept_raw")).alias("to_dept_id"),
        "from_title", "to_title",
        F.col("from_salary").cast("decimal(10,2)").alias("from_salary"),
        F.round(F.col("from_salary") * F.when(same_dept, 1.03 + F.col("r_raise") * 0.17)
                .otherwise(0.95 + F.col("r_raise") * 0.25), 2).cast("decimal(10,2)").alias("to_salary"),
        # Every change happens after the hire and no later than the as-of date.
        F.date_add(F.col("hire_date"), (F.lit(30) + F.floor(F.col("r_when") * span)).cast("int")).alias("effective_date"),
        "change_reason",
    ).orderBy("history_id")


def build_payroll(spark, as_of_date, employees):
    asof = F.lit(as_of_date).cast("date")
    months = spark.range(0, N_PAY_MONTHS).select(F.col("id").cast("int").alias("mn"))
    p = employees.select("emp_id", "dept_id", "salary", "bonus_pct").crossJoin(months)
    p = p.withColumn("base_salary", F.round(F.col("salary") / 12, 2))
    p = p.withColumn("bonus_amount", F.when(F.col("mn").isin(0, 12), F.round(F.col("salary") * F.col("bonus_pct") / 200, 2))
                     .otherwise(F.lit(0)))
    p = p.withColumn("overtime_hours", F.when(_hrand("emp_id", "mn", seed=1201) < 0.2,
                                              F.round(F.lit(2) + _hrand("emp_id", "mn", seed=1202) * 18, 2))
                     .otherwise(F.lit(0)).cast("decimal(6,2)"))
    p = p.withColumn("overtime_pay", F.round(F.col("overtime_hours") * (F.col("salary") / 2080) * 1.5, 2))
    p = p.withColumn("deductions", F.round((F.col("base_salary") + F.col("bonus_amount") + F.col("overtime_pay")) * 0.22, 2))
    p = p.withColumn("net_pay", F.col("base_salary") + F.col("bonus_amount") + F.col("overtime_pay") - F.col("deductions"))
    w = Window.orderBy(F.col("mn").desc(), "emp_id")
    return p.select(
        F.row_number().over(w).cast("int").alias("payroll_id"),
        "emp_id", "dept_id",
        F.add_months(F.trunc(asof, "month"), -F.col("mn")).alias("pay_period"),
        F.col("base_salary").cast("decimal(10,2)"), F.col("bonus_amount").cast("decimal(10,2)"),
        F.col("overtime_hours"), F.col("overtime_pay").cast("decimal(10,2)"),
        F.col("deductions").cast("decimal(10,2)"), F.col("net_pay").cast("decimal(10,2)"),
        F.when(F.col("mn") == 0, "Pending").otherwise("Processed").alias("pay_status"),
    )


def build_performance_reviews(spark, as_of_date, employees):
    asof = F.lit(as_of_date).cast("date")
    qs = spark.range(0, N_REVIEW_QUARTERS).select(F.col("id").cast("int").alias("qn"))
    r = employees.select("emp_id", "dept_id").crossJoin(qs)
    r = r.withColumn("review_date", F.add_months(asof, -3 * F.col("qn")))
    h = lambda s: _hrand("emp_id", "qn", seed=s)  # noqa: E731
    w = Window.orderBy(F.col("qn").desc(), "emp_id")
    return r.select(
        F.row_number().over(w).cast("int").alias("review_id"),
        "emp_id", "dept_id",
        F.concat(F.year("review_date").cast("string"), F.lit("-Q"), F.quarter("review_date").cast("string"))
        .alias("review_period"),
        "review_date",
        F.round(F.lit(1.0) + F.floor(h(1301) * 41) / 10, 1).cast("decimal(3,1)").alias("rating"),
        F.round(F.lit(50) + h(1302) * 50, 2).cast("decimal(5,2)").alias("goals_met_pct"),
        # reviewer is never the employee: shift by 1..199 positions around the 200 employees
        ((F.col("emp_id") - 1 + 1 + F.floor(h(1303) * (N_EMPLOYEES - 1))) % N_EMPLOYEES + 1).cast("int").alias("reviewer_id"),
        F.lit(None).cast("string").alias("comments"),
    )


def build_attendance(spark, as_of_date, employees):
    asof = F.lit(as_of_date).cast("date")
    ws = spark.range(0, N_ATT_WEEKS).select(F.col("id").cast("int").alias("wn"))
    a = employees.select("emp_id", "dept_id").crossJoin(ws)
    h = lambda s: _hrand("emp_id", "wn", seed=s)  # noqa: E731
    days = _between(h(1401), 3, 5)
    leave_raw = F.when(h(1402) < 0.2, _between(h(1403), 1, 2)).otherwise(F.lit(0))
    leave = F.least(leave_raw, F.lit(5) - days)
    a = a.withColumn("days_present", days).withColumn("leave_days", leave.cast("int"))
    w = Window.orderBy(F.col("wn").desc(), "emp_id")
    return a.select(
        F.row_number().over(w).cast("int").alias("attendance_id"),
        "emp_id", "dept_id",
        F.date_sub(F.date_trunc("week", asof).cast("date"), F.col("wn") * 7).alias("week_start_date"),
        "days_present",
        F.round(F.col("days_present") * 8 + F.lit(-2) + h(1404) * 5, 2).cast("decimal(6,2)").alias("hours_worked"),
        "leave_days",
        F.when(F.col("leave_days") > 0, _pick(h(1405), LEAVE_TYPES)).otherwise(F.lit("None")).alias("leave_type"),
    )


def build_hr(spark: SparkSession, as_of_date: str) -> dict:
    employees = build_employees(spark, as_of_date)
    return {
        "departments": build_departments(spark),
        "employees": employees,
        "job_history": build_job_history(spark, as_of_date, employees),
        "payroll": build_payroll(spark, as_of_date, employees),
        "performance_reviews": build_performance_reviews(spark, as_of_date, employees),
        "attendance": build_attendance(spark, as_of_date, employees),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Integrity checks

# COMMAND ----------

EXPECTED_ROWS = {
    "departments": 10, "employees": 200, "job_history": 300, "payroll": 200 * 24,
    "performance_reviews": 200 * 8, "attendance": 200 * 104,
}


def check_hr(spark, t: dict, as_of_date: str) -> dict:
    asof = F.lit(as_of_date).cast("date")
    e, p, r, a, jh = t["employees"], t["payroll"], t["performance_reviews"], t["attendance"], t["job_history"]

    def n(df):
        return df.count()

    pe = p.join(e.select("emp_id", F.col("dept_id").alias("e_dept"), "hire_date", "salary"), "emp_id")
    violations = {
        "hired_too_recently": n(e.where(F.datediff(asof, F.col("hire_date")) < MIN_TENURE_DAYS)),
        "payroll_dept_not_employee_dept": n(pe.where(F.col("dept_id") != F.col("e_dept"))),
        "net_pay_formula": n(p.where(F.abs(F.col("net_pay") - (F.col("base_salary") + F.col("bonus_amount")
                                                              + F.col("overtime_pay") - F.col("deductions"))) > 0.005)),
        "overtime_pay_without_hours": n(p.where((F.col("overtime_hours") == 0) & (F.col("overtime_pay") != 0))),
        "overtime_hours_without_pay": n(p.where((F.col("overtime_hours") > 0) & (F.col("overtime_pay") <= 0))),
        "paid_before_hire": n(pe.where(F.col("pay_period") < F.trunc(F.col("hire_date"), "month"))),
        "duplicate_pay_period": n(p.groupBy("emp_id", "pay_period").count().where("count > 1")),
        "review_period_not_from_date": n(r.where(F.col("review_period") != F.concat(
            F.year("review_date").cast("string"), F.lit("-Q"), F.quarter("review_date").cast("string")))),
        "rating_out_of_range": n(r.where((F.col("rating") < 1.0) | (F.col("rating") > 5.0))),
        "self_review": n(r.where(F.col("emp_id") == F.col("reviewer_id"))),
        "reviewer_not_an_employee": n(r.join(e, r["reviewer_id"] == e["emp_id"], "left_anti")),
        "review_after_as_of": n(r.where(F.col("review_date") > asof)),
        "attendance_over_five_days": n(a.where(F.col("days_present") + F.col("leave_days") > 5)),
        "leave_type_mismatch": n(a.where(((F.col("leave_days") > 0) & (F.col("leave_type") == "None"))
                                         | ((F.col("leave_days") == 0) & (F.col("leave_type") != "None")))),
        "attendance_week_after_as_of": n(a.where(F.col("week_start_date") > asof)),
        "job_change_before_hire": n(jh.join(e.select("emp_id", "hire_date"), "emp_id")
                                    .where(F.col("effective_date") <= F.col("hire_date"))),
        "job_change_after_as_of": n(jh.where(F.col("effective_date") > asof)),
        "orphan_dept": n(e.join(t["departments"], "dept_id", "left_anti")),
    }
    counts = {name: t[name].count() for name in EXPECTED_ROWS}
    bad_counts = {k: (counts[k], v) for k, v in EXPECTED_ROWS.items() if counts[k] != v}
    failed = {k: v for k, v in violations.items() if v}
    assert not bad_counts, f"unexpected row counts (actual, expected): {bad_counts}"
    assert not failed, f"integrity violations: {failed}"
    return {"row_counts": counts, "violations": violations}

# COMMAND ----------

TABLE_COMMENTS = {
    "departments": "Departments with annual budget, headcount budget, cost center code and office location.",
    "employees": "Employee master. One row per employee; status is Active or Inactive. Salary is annual, USD.",
    "job_history": "Position changes over time: promotions, transfers and merit raises with before/after titles and salaries.",
    "payroll": "Monthly payroll. One row per employee per month for 24 months; net_pay = base + bonus + overtime - deductions.",
    "performance_reviews": "Quarterly performance reviews, 8 quarters per employee, rating 1.0-5.0.",
    "attendance": "Weekly attendance, 104 weeks per employee.",
}

COLUMN_COMMENTS = {
    "departments": {
        "dept_id": "Primary key", "dept_name": "Department name", "annual_budget": "Annual budget, USD",
        "headcount_budget": "Planned headcount", "cost_center": "Cost center code", "location": "Office location",
        "manager_id": "Employee id of the department head",
    },
    "employees": {
        "emp_id": "Primary key", "first_name": "First name", "last_name": "Last name", "email": "Work email",
        "dept_id": "References departments.dept_id", "job_title": "Job title", "job_level": "Level L1 (junior) to L7",
        "hire_date": "Hire date, at least 760 days before the data's as-of date", "salary": "Annual salary, USD",
        "bonus_pct": "Target bonus percent of salary", "status": "Active or Inactive",
        "manager_id": "Employee id of the manager; null for department heads",
    },
    "job_history": {
        "history_id": "Primary key", "emp_id": "References employees.emp_id", "from_dept_id": "Department before the change",
        "to_dept_id": "Department after the change", "from_title": "Title before", "to_title": "Title after",
        "from_salary": "Annual salary before, USD", "to_salary": "Annual salary after, USD",
        "effective_date": "Date the change took effect, after the hire date",
        "change_reason": "Promotion, Transfer, Restructure, Merit Raise or Lateral Move",
    },
    "payroll": {
        "payroll_id": "Primary key", "emp_id": "References employees.emp_id",
        "dept_id": "The employee's department", "pay_period": "First day of the pay month",
        "base_salary": "Monthly base salary (annual / 12), USD", "bonus_amount": "Bonus paid, USD; paid in months 0 and 12",
        "overtime_hours": "Overtime hours worked", "overtime_pay": "Overtime hours x hourly rate x 1.5, USD",
        "deductions": "22 percent of gross pay, USD",
        "net_pay": "base_salary + bonus_amount + overtime_pay - deductions, USD",
        "pay_status": "Processed for closed months, Pending for the current month",
    },
    "performance_reviews": {
        "review_id": "Primary key", "emp_id": "Employee reviewed", "dept_id": "The employee's department",
        "review_period": "Quarter label derived from review_date, e.g. 2026-Q2", "review_date": "Date of the review",
        "rating": "1.0 to 5.0, higher is better", "goals_met_pct": "Percent of goals met, 50-100",
        "reviewer_id": "Employee who reviewed; never the same employee", "comments": "Free-text comments (unused)",
    },
    "attendance": {
        "attendance_id": "Primary key", "emp_id": "References employees.emp_id", "dept_id": "The employee's department",
        "week_start_date": "Monday of the week", "days_present": "Days present, 3-5",
        "hours_worked": "Hours worked in the week", "leave_days": "Leave days taken; days_present + leave_days <= 5",
        "leave_type": "Annual, Sick or Public Holiday; None when no leave",
    },
}


def write_tables(spark, catalog: str, schema: str, tables: dict, table_comments: dict, column_comments: dict):
    """Write each DataFrame as a Delta table, then apply table and column comments."""
    for name, df in tables.items():
        fq = f"{catalog}.{schema}.{name}"
        df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq)
        if name in table_comments:
            spark.sql(f"COMMENT ON TABLE {fq} IS '{table_comments[name].replace(chr(39), chr(39) * 2)}'")
        for col, text in column_comments.get(name, {}).items():
            spark.sql(f"ALTER TABLE {fq} ALTER COLUMN {col} COMMENT '{text.replace(chr(39), chr(39) * 2)}'")

# COMMAND ----------

if __name__ == "__main__":
    dbutils.widgets.text("catalog", "workspace")  # noqa: F821
    dbutils.widgets.text("as_of_date", "2026-06-30")  # noqa: F821
    catalog = dbutils.widgets.get("catalog")  # noqa: F821
    as_of_date = dbutils.widgets.get("as_of_date")  # noqa: F821

    tables = build_hr(spark, as_of_date)  # noqa: F821
    write_tables(spark, catalog, "hr", tables, TABLE_COMMENTS, COLUMN_COMMENTS)  # noqa: F821

    written = {name: spark.table(f"{catalog}.hr.{name}") for name in tables}  # noqa: F821
    for name, df in written.items():
        print(f"{name:20s} {df.count():>8,d} rows")

# COMMAND ----------

if __name__ == "__main__":
    result = check_hr(spark, written, as_of_date)  # noqa: F821
    for rule, count in result["violations"].items():
        print(f"  ok  {rule}: {count}")
    print("All HR integrity checks passed.")
