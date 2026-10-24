# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Seed finance data
# MAGIC
# MAGIC Creates `finance.gl_accounts`, `cost_centers`, `budget_lines`, `invoices`, `invoice_lines` and `expenses`.
# MAGIC Run **after** notebooks 01 and 02: invoices are issued against real orders and expenses against real employees.
# MAGIC
# MAGIC **Integrity rules the data guarantees** (asserted at the end):
# MAGIC * `budget_lines` has exactly one row per department x account x year x month, for the previous year and the
# MAGIC   current year up to the as-of month, on expense accounts only
# MAGIC * an invoice belongs to a real non-cancelled order and to the customer who placed it; its amount is the sum
# MAGIC   of its lines, `total_due = invoice_amount + tax_amount`, and `paid_amount` is the full amount exactly when
# MAGIC   the status is Paid
# MAGIC * status follows the dates: Overdue means unpaid and past due at the as-of date, Sent means unpaid and not yet
# MAGIC   due; `payment_date` exists only for Paid invoices and is never before the invoice date
# MAGIC * an expense is claimed by a real employee, carries that employee's department, uses a cost center of that
# MAGIC   department, is booked to the GL account for its category, and is reimbursed only if Approved

# COMMAND ----------


from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

# (account_code, account_name, account_type, account_category)
GL_ACCOUNTS = [
    ('4001', 'Product Revenue', 'Revenue', 'Sales'),
    ('4002', 'Service Revenue', 'Revenue', 'Sales'),
    ('4003', 'Subscription Revenue', 'Revenue', 'Sales'),
    ('4004', 'Other Revenue', 'Revenue', 'Other'),
    ('5001', 'Cost of Goods Sold', 'Expense', 'COGS'),
    ('5002', 'Shipping & Fulfilment', 'Expense', 'COGS'),
    ('5003', 'Packaging', 'Expense', 'COGS'),
    ('6001', 'Salaries & Wages', 'Expense', 'Payroll'),
    ('6002', 'Bonuses', 'Expense', 'Payroll'),
    ('6003', 'Benefits & Insurance', 'Expense', 'Payroll'),
    ('6004', 'Payroll Taxes', 'Expense', 'Payroll'),
    ('6005', 'Contract Labour', 'Expense', 'Payroll'),
    ('7001', 'Rent & Facilities', 'Expense', 'Facilities'),
    ('7002', 'Utilities', 'Expense', 'Facilities'),
    ('7003', 'Office Supplies', 'Expense', 'Facilities'),
    ('7004', 'Equipment Lease', 'Expense', 'Facilities'),
    ('7005', 'Maintenance & Repair', 'Expense', 'Facilities'),
    ('8001', 'Software & SaaS', 'Expense', 'Technology'),
    ('8002', 'Hardware', 'Expense', 'Technology'),
    ('8003', 'Cloud Infrastructure', 'Expense', 'Technology'),
    ('8004', 'IT Support', 'Expense', 'Technology'),
    ('9001', 'Marketing & Advertising', 'Expense', 'Marketing'),
    ('9002', 'Events & Sponsorships', 'Expense', 'Marketing'),
    ('9003', 'PR & Communications', 'Expense', 'Marketing'),
    ('9004', 'Market Research', 'Expense', 'Marketing'),
    ('10001', 'Travel & Entertainment', 'Expense', 'T&E'),
    ('10002', 'Meals & Entertainment', 'Expense', 'T&E'),
    ('10003', 'Conference & Training', 'Expense', 'T&E'),
    ('11001', 'Legal & Compliance', 'Expense', 'G&A'),
    ('11002', 'Audit & Accounting', 'Expense', 'G&A'),
    ('11003', 'Insurance', 'Expense', 'G&A'),
    ('11004', 'Banking & Finance Charges', 'Expense', 'G&A'),
    ('12001', 'Depreciation', 'Expense', 'Non-Cash'),
    ('12002', 'Amortisation', 'Expense', 'Non-Cash'),
    ('13001', 'Interest Income', 'Revenue', 'Other'),
    ('13002', 'Interest Expense', 'Expense', 'Financing'),
    ('14001', 'Cash & Equivalents', 'Asset', 'Current'),
    ('14002', 'Accounts Receivable', 'Asset', 'Current'),
    ('14003', 'Inventory', 'Asset', 'Current'),
    ('14004', 'Prepaid Expenses', 'Asset', 'Current'),
    ('15001', 'Property & Equipment', 'Asset', 'Non-Current'),
    ('15002', 'Intangible Assets', 'Asset', 'Non-Current'),
    ('16001', 'Accounts Payable', 'Liability', 'Current'),
    ('16002', 'Accrued Liabilities', 'Liability', 'Current'),
    ('16003', 'Deferred Revenue', 'Liability', 'Current'),
    ('16004', 'Short-Term Debt', 'Liability', 'Current'),
    ('17001', 'Long-Term Debt', 'Liability', 'Non-Current'),
    ('17002', 'Equity', 'Liability', 'Equity'),
    ('18001', 'Tax Payable', 'Liability', 'Current'),
    ('18002', 'Income Tax Expense', 'Expense', 'Tax'),
]

# (cc_code, cc_name, dept_id, cc_manager)
COST_CENTERS = [
    ('CC-ENG-1', 'Engineering - Core Platform', 1, 'Sarah Chen'),
    ('CC-ENG-2', 'Engineering - Data Infra', 1, 'Robert Singh'),
    ('CC-ENG-3', 'Engineering - Product Dev', 1, 'Emily Rodriguez'),
    ('CC-SAL-1', 'Sales - Enterprise', 2, 'Chris Wang'),
    ('CC-SAL-2', 'Sales - SMB', 2, 'Ryan Davis'),
    ('CC-MKT-1', 'Marketing - Digital', 3, 'Samantha Green'),
    ('CC-MKT-2', 'Marketing - Brand & Events', 3, 'Brandon Cole'),
    ('CC-HR-1', 'Human Resources - All', 4, 'Linda Wright'),
    ('CC-FIN-1', 'Finance - FP&A', 5, 'Catherine Young'),
    ('CC-FIN-2', 'Finance - Accounting', 5, 'Derek Phillips'),
    ('CC-OPS-1', 'Operations - Logistics', 6, 'Patrick Reed'),
    ('CC-OPS-2', 'Operations - Facilities', 6, 'Monica Flores'),
    ('CC-CSX-1', 'Customer Success', 7, 'Angela Kim'),
    ('CC-PM-1', 'Product Management', 8, 'Priya Sharma'),
    ('CC-DA-1', 'Data & Analytics', 10, 'Andrew Carter'),
]

EXPENSE_CATEGORIES = ["Travel", "Meals & Entertainment", "Software", "Office Supplies", "Training", "Marketing"]
CATEGORY_ACCOUNT = {
    "Travel": "Travel & Entertainment", "Meals & Entertainment": "Meals & Entertainment",
    "Software": "Software & SaaS", "Office Supplies": "Office Supplies",
    "Training": "Conference & Training", "Marketing": "Marketing & Advertising",
}
VENDORS = ["Delta Airlines", "Marriott Hotels", "AWS", "Salesforce", "Zoom", "Uber", "Amex Travel", "Office Depot"]
ACCOUNT_ID = {a[1]: i + 1 for i, a in enumerate(GL_ACCOUNTS)}
# Per-department tendency to over/under-spend, so budget-vs-actual questions have a real answer.
DEPT_SPEND_BIAS = [1.02, 1.12, 0.95, 0.88, 1.15, 1.05, 0.92, 1.18, 0.97, 1.08]

N_DEPTS, N_PRODUCTS, N_EMPLOYEES = 10, 96, 200
N_BUDGET, N_INVOICES, N_EXPENSES = 3000, 2000, 3000

# COMMAND ----------

# MAGIC %md
# MAGIC ## Builders
# MAGIC
# MAGIC `build_finance(spark, as_of_date, orders=..., products=..., departments=..., employees=...)` takes the already
# MAGIC built sales and HR DataFrames, so it can run in a notebook (read from tables) or in a unit test (pass builders'
# MAGIC output).

# COMMAND ----------

def _pick(r, options):
    arr = F.array(*[F.lit(o) for o in options])
    return F.element_at(arr, (F.floor(r * len(options)) + 1).cast("int"))


def _between(r, lo, hi):
    return (F.floor(r * (hi - lo + 1)) + lo).cast("int")


def _hrand(*cols, seed):
    return F.pmod(F.xxhash64(*cols, F.lit(seed)), F.lit(1000000)) / F.lit(1000000.0)


def _lookup(col, mapping: dict, default=None):
    """CASE WHEN col = k THEN v ... (a map lookup that cannot raise under ANSI mode)."""
    expr = F.lit(default)
    for k, v in reversed(list(mapping.items())):
        expr = F.when(col == F.lit(k), F.lit(v)).otherwise(expr)
    return expr


def build_gl_accounts(spark):
    return spark.createDataFrame(
        [(i + 1, *a, True) for i, a in enumerate(GL_ACCOUNTS)],
        "account_id int, account_code string, account_name string, account_type string, account_category string, "
        "is_active boolean",
    )


def build_cost_centers(spark):
    return spark.createDataFrame(
        [(i + 1, *c) for i, c in enumerate(COST_CENTERS)],
        "cc_id int, cc_code string, cc_name string, dept_id int, cc_manager string",
    )


def build_budget_lines(spark, as_of_date):
    year, month = int(as_of_date[:4]), int(as_of_date[5:7])
    periods = [(year - 1, m) for m in range(1, 13)] + [(year, m) for m in range(1, month + 1)]
    accounts = [i + 1 for i, a in enumerate(GL_ACCOUNTS) if a[2] == "Expense"]
    grid = spark.createDataFrame(
        [(d, a, y, m) for d in range(1, N_DEPTS + 1) for a in accounts for (y, m) in periods],
        "dept_id int, account_id int, budget_year int, budget_month int",
    )
    first_cc = {}
    for i, c in enumerate(COST_CENTERS):
        first_cc.setdefault(c[2], i + 1)
    h = lambda s: _hrand("dept_id", "account_id", "budget_year", "budget_month", seed=s)  # noqa: E731
    # The grid is unique per (dept, account, year, month); sampling it keeps that grain unique.
    sampled = grid.withColumn("k", h(1501)).orderBy("k").limit(N_BUDGET)
    budget = F.round(F.lit(5000) + h(1502) * 195000, 2)
    bias = _lookup(F.col("dept_id"), dict(zip(range(1, N_DEPTS + 1), DEPT_SPEND_BIAS, strict=True)), 1.0)
    w = Window.orderBy("budget_year", "budget_month", "dept_id", "account_id")
    return sampled.select(
        F.row_number().over(w).cast("int").alias("budget_id"),
        "dept_id", "account_id",
        _lookup(F.col("dept_id"), first_cc).cast("int").alias("cc_id"),
        "budget_year", "budget_month",
        budget.cast("decimal(14,2)").alias("budget_amount"),
        F.round(budget * (F.lit(0.9) + h(1503) * 0.2), 2).cast("decimal(14,2)").alias("revised_amount"),
        F.round(budget * bias * (F.lit(0.85) + h(1504) * 0.3), 2).cast("decimal(14,2)").alias("actual_amount"),
    )


def build_invoices(spark, as_of_date, orders, products):
    asof = F.lit(as_of_date).cast("date")
    ok = orders.where(F.col("status") != "Cancelled").select("order_id", "customer_id", "order_date")
    picked = ok.withColumn("k", F.xxhash64("order_id", F.lit(1601))).orderBy("k").limit(N_INVOICES)
    w = Window.orderBy("order_date", "order_id")
    inv0 = picked.select(F.row_number().over(w).cast("int").alias("invoice_id"), "order_id", "customer_id", "order_date")

    n_lines = _between(_hrand("invoice_id", seed=1602), 1, 5)
    lines = (
        inv0.select("invoice_id", F.explode(F.sequence(F.lit(1), n_lines)).alias("line_no"))
        .select(
            "invoice_id", "line_no",
            _between(_hrand("invoice_id", "line_no", seed=1603), 1, N_PRODUCTS).alias("product_id"),
            _between(_hrand("invoice_id", "line_no", seed=1604), 1, 10).alias("quantity"),
            _between(_hrand("invoice_id", "line_no", seed=1605), 1, 3).alias("account_id"),
        )
        .join(F.broadcast(products.select("product_id", "unit_price")), "product_id")
        .withColumn("line_amount", F.round(F.col("quantity") * F.col("unit_price"), 2).cast("decimal(12,2)"))
    )
    lw = Window.orderBy("invoice_id", "line_no")
    invoice_lines = lines.select(
        F.row_number().over(lw).cast("int").alias("line_id"), "invoice_id", "product_id",
        F.concat(F.lit("Product / Service item "), F.col("line_no").cast("string")).alias("description"),
        "quantity", "unit_price", "line_amount", "account_id",
    )
    amounts = lines.groupBy("invoice_id").agg(F.sum("line_amount").cast("decimal(12,2)").alias("invoice_amount"))

    j = inv0.join(amounts, "invoice_id")
    inv_date = F.least(F.date_add(F.col("order_date"), _between(_hrand("invoice_id", seed=1606), 0, 5)), asof)
    j = j.withColumn("invoice_date", inv_date)
    j = j.withColumn("due_date", F.date_add(F.col("invoice_date"), _between(_hrand("invoice_id", seed=1607), 15, 60)))
    r = _hrand("invoice_id", seed=1608)
    cls = F.when(r < 0.45, "Paid").when(r < 0.80, "Unpaid").when(r < 0.90, "Draft").otherwise("Void")
    status = F.when(cls == "Unpaid", F.when(F.col("due_date") < asof, "Overdue").otherwise("Sent")).otherwise(cls)
    j = j.withColumn("invoice_status", status)
    j = j.withColumn("tax_amount", F.round(F.col("invoice_amount") * 0.08, 2).cast("decimal(10,2)"))
    j = j.withColumn("total_due", (F.col("invoice_amount") + F.col("tax_amount")).cast("decimal(12,2)"))
    paid = F.col("invoice_status") == "Paid"
    pay_date = F.least(F.date_add(F.col("invoice_date"), _between(_hrand("invoice_id", seed=1609), 1, 45)), asof)
    invoices = j.select(
        "invoice_id", F.concat(F.lit("INV-"), F.lpad(F.col("invoice_id").cast("string"), 6, "0")).alias("invoice_number"),
        "customer_id", "order_id", "invoice_date", "due_date", "invoice_amount", "tax_amount", "total_due",
        F.when(paid, F.col("total_due")).otherwise(F.lit(0)).cast("decimal(12,2)").alias("paid_amount"),
        "invoice_status",
        F.when(paid, pay_date).cast("date").alias("payment_date"),
    ).orderBy("invoice_id")
    return invoices, invoice_lines


def build_expenses(spark, as_of_date, departments, employees):
    asof = F.lit(as_of_date).cast("date")
    first_cc = {}
    for i, c in enumerate(COST_CENTERS):
        first_cc.setdefault(c[2], i + 1)
    c = F.col("id")
    r_appr = F.rand(1707)
    raw = spark.range(1, N_EXPENSES + 1, 1, 1).select(
        c.cast("int").alias("expense_id"),
        _between(F.rand(1701), 1, N_EMPLOYEES).alias("emp_id"),
        F.date_sub(asof, _between(F.rand(1702), 0, 729)).alias("expense_date"),
        F.round(F.lit(10) + F.rand(1703) * 4990, 2).cast("decimal(10,2)").alias("amount"),
        _pick(F.rand(1704), EXPENSE_CATEGORIES).alias("category"),
        _pick(F.rand(1705), VENDORS).alias("vendor"),
        F.when(r_appr < 0.2, "Pending").when(r_appr < 0.4, "Rejected").otherwise("Approved").alias("approval_status"),
        (F.rand(1708) < 2 / 3).alias("reimb_draw"),
    )
    e = raw.join(F.broadcast(employees.select("emp_id", "dept_id", "hire_date")), "emp_id")
    return e.select(
        "expense_id", "emp_id", "dept_id",
        _lookup(F.col("dept_id"), first_cc).cast("int").alias("cc_id"),
        _lookup(F.col("category"), {k: ACCOUNT_ID[v] for k, v in CATEGORY_ACCOUNT.items()}).cast("int").alias("account_id"),
        "expense_date", "amount", F.lit("USD").alias("currency"), "category", "vendor",
        F.concat(F.lit("Expense item "), F.col("expense_id").cast("string")).alias("description"),
        "approval_status",
        (F.col("reimb_draw") & (F.col("approval_status") == "Approved")).alias("reimbursed"),
    ).orderBy("expense_id")


def build_finance(spark: SparkSession, as_of_date: str, *, orders, products, departments, employees) -> dict:
    invoices, invoice_lines = build_invoices(spark, as_of_date, orders, products)
    return {
        "gl_accounts": build_gl_accounts(spark),
        "cost_centers": build_cost_centers(spark),
        "budget_lines": build_budget_lines(spark, as_of_date),
        "invoices": invoices,
        "invoice_lines": invoice_lines,
        "expenses": build_expenses(spark, as_of_date, departments, employees),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Integrity checks

# COMMAND ----------

EXPECTED_ROWS = {"gl_accounts": 50, "cost_centers": 15, "budget_lines": 3000, "invoices": 2000, "expenses": 3000}


def check_finance(spark, t: dict, as_of_date: str, *, orders, employees) -> dict:
    asof = F.lit(as_of_date).cast("date")
    year, month = int(as_of_date[:4]), int(as_of_date[5:7])
    b, inv, il, ex = t["budget_lines"], t["invoices"], t["invoice_lines"], t["expenses"]

    def n(df):
        return df.count()

    line_sum = il.groupBy("invoice_id").agg(F.sum("line_amount").alias("s"), F.count("*").alias("lines"))
    ij = inv.join(line_sum, "invoice_id", "left").join(
        orders.select("order_id", F.col("customer_id").alias("o_customer"), F.col("status").alias("o_status"),
                      "order_date"), "order_id", "left")
    ee = ex.join(employees.select("emp_id", F.col("dept_id").alias("e_dept"), "hire_date"), "emp_id", "left")
    ee = ee.join(t["cost_centers"].select("cc_id", F.col("dept_id").alias("cc_dept")), "cc_id", "left")
    ba = b.join(t["gl_accounts"].select("account_id", "account_type"), "account_id", "left") \
          .join(t["cost_centers"].select("cc_id", F.col("dept_id").alias("cc_dept")), "cc_id", "left")
    expected_account = _lookup(F.col("category"), {k: ACCOUNT_ID[v] for k, v in CATEGORY_ACCOUNT.items()})

    violations = {
        "budget_duplicate_grain": n(b.groupBy("dept_id", "account_id", "budget_year", "budget_month").count().where("count > 1")),
        "budget_year_out_of_range": n(b.where(~F.col("budget_year").isin(year - 1, year))),
        "budget_month_after_as_of": n(b.where((F.col("budget_year") == year) & (F.col("budget_month") > month))),
        "budget_non_expense_account": n(ba.where(F.col("account_type") != "Expense")),
        "budget_cost_center_wrong_dept": n(ba.where(F.col("cc_id").isNotNull() & (F.col("cc_dept") != F.col("dept_id")))),
        "invoice_customer_not_order_customer": n(ij.where(F.col("customer_id") != F.col("o_customer"))),
        "invoice_for_missing_or_cancelled_order": n(ij.where(F.col("o_status").isNull() | (F.col("o_status") == "Cancelled"))),
        "invoice_before_order": n(ij.where(F.col("invoice_date") < F.col("order_date"))),
        "duplicate_invoice_per_order": n(inv.groupBy("order_id").count().where("count > 1")),
        "due_before_invoice": n(inv.where(F.col("due_date") < F.col("invoice_date"))),
        "invoice_amount_not_sum_of_lines": n(ij.where(F.abs(F.col("invoice_amount") - F.col("s")) > 0.005)),
        "invoice_without_lines": n(ij.where(F.col("lines").isNull())),
        "total_due_formula": n(inv.where(F.abs(F.col("total_due") - F.col("invoice_amount") - F.col("tax_amount")) > 0.005)),
        "paid_amount_vs_status": n(inv.where(((F.col("invoice_status") == "Paid") & (F.col("paid_amount") != F.col("total_due")))
                                             | ((F.col("invoice_status") != "Paid") & (F.col("paid_amount") != 0)))),
        "payment_date_vs_status": n(inv.where(((F.col("invoice_status") == "Paid") & F.col("payment_date").isNull())
                                              | ((F.col("invoice_status") != "Paid") & F.col("payment_date").isNotNull()))),
        "payment_before_invoice": n(inv.where(F.col("payment_date") < F.col("invoice_date"))),
        "payment_after_as_of": n(inv.where(F.col("payment_date") > asof)),
        "overdue_but_not_past_due": n(inv.where((F.col("invoice_status") == "Overdue") & (F.col("due_date") >= asof))),
        "sent_but_past_due": n(inv.where((F.col("invoice_status") == "Sent") & (F.col("due_date") < asof))),
        "expense_dept_not_employee_dept": n(ee.where(F.col("dept_id") != F.col("e_dept"))),
        "expense_cost_center_wrong_dept": n(ee.where(F.col("cc_id").isNotNull() & (F.col("cc_dept") != F.col("dept_id")))),
        "expense_before_hire": n(ee.where(F.col("expense_date") < F.col("hire_date"))),
        "expense_after_as_of": n(ex.where(F.col("expense_date") > asof)),
        "expense_wrong_gl_account": n(ex.where(F.col("account_id") != expected_account)),
        "reimbursed_but_not_approved": n(ex.where(F.col("reimbursed") & (F.col("approval_status") != "Approved"))),
    }
    counts = {name: t[name].count() for name in EXPECTED_ROWS}
    lines = counts.setdefault("invoice_lines", il.count())
    bad_counts = {k: (counts[k], v) for k, v in EXPECTED_ROWS.items() if counts[k] != v}
    failed = {k: v for k, v in violations.items() if v}
    assert not bad_counts, f"unexpected row counts (actual, expected): {bad_counts}"
    assert 5500 <= lines <= 6500, f"invoice_lines should be ~6,000 (1-5 lines per invoice), got {lines}"
    assert not failed, f"integrity violations: {failed}"
    return {"row_counts": counts, "violations": violations}

# COMMAND ----------

TABLE_COMMENTS = {
    "gl_accounts": "General ledger chart of accounts: code, name, type (Revenue, Expense, Asset, Liability) and category.",
    "cost_centers": "Cost centers, each owned by one department.",
    "budget_lines": "Monthly budget vs actual by department and expense account. Unique per department, account, year "
                    "and month. Variance = actual_amount - budget_amount; positive means over budget.",
    "invoices": "Customer invoices, one per non-cancelled order. Status: Draft, Sent, Paid, Overdue or Void. "
                "Outstanding balance means status Sent or Overdue.",
    "invoice_lines": "Invoice line items; invoice_amount is the sum of line_amount.",
    "expenses": "Employee expense claims, booked to the GL account of their category. Reimbursed only when Approved.",
}

COLUMN_COMMENTS = {
    "gl_accounts": {
        "account_id": "Primary key", "account_code": "Ledger code", "account_name": "Account name",
        "account_type": "Revenue, Expense, Asset or Liability", "account_category": "Grouping such as Payroll, Technology",
        "is_active": "Whether the account is in use",
    },
    "cost_centers": {
        "cc_id": "Primary key", "cc_code": "Cost center code", "cc_name": "Cost center name",
        "dept_id": "Owning department, references hr.departments.dept_id", "cc_manager": "Manager name",
    },
    "budget_lines": {
        "budget_id": "Primary key", "dept_id": "References hr.departments.dept_id",
        "account_id": "References gl_accounts.account_id (expense accounts only)",
        "cc_id": "Cost center of the department; null for departments without one",
        "budget_year": "Calendar year", "budget_month": "Month number 1-12",
        "budget_amount": "Original budget, USD", "revised_amount": "Revised budget, USD", "actual_amount": "Actual spend, USD",
    },
    "invoices": {
        "invoice_id": "Primary key", "invoice_number": "Display number such as INV-000123",
        "customer_id": "Customer who placed the order", "order_id": "The order being invoiced",
        "invoice_date": "Date issued; never before the order date", "due_date": "Payment due date",
        "invoice_amount": "Sum of line amounts before tax, USD", "tax_amount": "8 percent tax, USD",
        "total_due": "invoice_amount + tax_amount, USD", "paid_amount": "Amount received; equals total_due when Paid, else 0",
        "invoice_status": "Draft, Sent, Paid, Overdue or Void", "payment_date": "Date paid; null unless Paid",
    },
    "invoice_lines": {
        "line_id": "Primary key", "invoice_id": "References invoices.invoice_id",
        "product_id": "References sales.products.product_id", "description": "Line description",
        "quantity": "Units", "unit_price": "Product list price, USD", "line_amount": "quantity * unit_price, USD",
        "account_id": "Revenue GL account",
    },
    "expenses": {
        "expense_id": "Primary key", "emp_id": "Employee who claimed", "dept_id": "The employee's department",
        "cc_id": "Cost center of the department; null when it has none",
        "account_id": "GL account for the category", "expense_date": "Date of the expense", "amount": "Claim amount, USD",
        "currency": "Always USD", "category": "Travel, Meals & Entertainment, Software, Office Supplies, Training or Marketing",
        "vendor": "Vendor paid", "description": "Line description", "approval_status": "Pending, Approved or Rejected",
        "reimbursed": "True only when approved and paid back to the employee",
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

    orders = spark.table(f"{catalog}.sales.orders")  # noqa: F821
    products = spark.table(f"{catalog}.sales.products")  # noqa: F821
    departments = spark.table(f"{catalog}.hr.departments")  # noqa: F821
    employees = spark.table(f"{catalog}.hr.employees")  # noqa: F821

    tables = build_finance(spark, as_of_date, orders=orders, products=products,  # noqa: F821
                           departments=departments, employees=employees)
    write_tables(spark, catalog, "finance", tables, TABLE_COMMENTS, COLUMN_COMMENTS)  # noqa: F821

    written = {name: spark.table(f"{catalog}.finance.{name}") for name in tables}  # noqa: F821
    for name, df in written.items():
        print(f"{name:16s} {df.count():>8,d} rows")

# COMMAND ----------

if __name__ == "__main__":
    result = check_finance(spark, written, as_of_date, orders=orders, employees=employees)  # noqa: F821
    print("invoice_lines", result["row_counts"]["invoice_lines"], "rows")
    for rule, count in result["violations"].items():
        print(f"  ok  {rule}: {count}")
    print("All finance integrity checks passed.")
