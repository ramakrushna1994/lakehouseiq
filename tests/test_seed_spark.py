"""Seed-data builders and their integrity checks, run in a local Spark session.

Skipped automatically when pyspark or a JVM is unavailable (as in CI, which installs only
requirements-dev.txt). The notebooks are plain Python in Databricks source format, so they
import as modules; only the ``if __name__ == "__main__"`` cells need a workspace.
"""

import importlib.util
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("no JVM available for local Spark", allow_module_level=True)

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"
AS_OF = "2026-06-30"


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"), NOTEBOOKS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    from pyspark.sql import SparkSession

    s = (
        SparkSession.builder.master("local[1]").appName("plumbline-tests")
        .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.warehouse.dir", str(tmp_path_factory.mktemp("wh"))).getOrCreate()
    )
    s.sparkContext.setLogLevel("ERROR")
    yield s
    s.stop()


def persist(spark, tables, root):
    """Write to parquet and read back, so checks run against stored data like the notebooks do."""
    for name, df in tables.items():
        df.write.mode("overwrite").parquet(str(root / name))
    return {name: spark.read.parquet(str(root / name)) for name in tables}


@pytest.fixture(scope="module")
def warehouse(spark, tmp_path_factory):
    sales_nb, hr_nb, fin_nb = load("01_seed_sales.py"), load("02_seed_hr.py"), load("03_seed_finance.py")
    root = tmp_path_factory.mktemp("data")
    sales = persist(spark, sales_nb.build_sales(spark, AS_OF), root / "sales")
    hr = persist(spark, hr_nb.build_hr(spark, AS_OF), root / "hr")
    fin_tables = fin_nb.build_finance(
        spark, AS_OF, orders=sales["orders"], products=sales["products"],
        departments=hr["departments"], employees=hr["employees"],
    )
    fin = persist(spark, fin_tables, root / "fin")
    return {"nb": (sales_nb, hr_nb, fin_nb), "sales": sales, "hr": hr, "fin": fin}


def test_sales_row_counts_and_integrity(spark, warehouse):
    sales_nb = warehouse["nb"][0]
    result = sales_nb.check_sales(spark, warehouse["sales"], AS_OF)
    assert result["row_counts"]["orders"] == 5000 and result["row_counts"]["returns"] == 1500
    assert 14000 <= result["row_counts"]["order_items"] <= 16000
    assert not any(result["violations"].values())


def test_hr_row_counts_and_integrity(spark, warehouse):
    result = warehouse["nb"][1].check_hr(spark, warehouse["hr"], AS_OF)
    assert result["row_counts"]["payroll"] == 4800 and result["row_counts"]["attendance"] == 20800
    assert not any(result["violations"].values())


def test_finance_row_counts_and_integrity(spark, warehouse):
    result = warehouse["nb"][2].check_finance(
        spark, warehouse["fin"], AS_OF, orders=warehouse["sales"]["orders"], employees=warehouse["hr"]["employees"]
    )
    assert result["row_counts"]["budget_lines"] == 3000 and result["row_counts"]["invoices"] == 2000
    assert not any(result["violations"].values())


def test_checks_actually_detect_violations(spark, warehouse):
    """A check that cannot fail proves nothing: corrupt the data and expect the assertion to fire."""
    from pyspark.sql import functions as F

    sales_nb = warehouse["nb"][0]
    broken = dict(warehouse["sales"])
    broken["orders"] = broken["orders"].withColumn("ship_date", F.date_sub("order_date", 3))
    with pytest.raises(AssertionError, match="ship_before_order"):
        sales_nb.check_sales(spark, broken, AS_OF)

    broken = dict(warehouse["sales"])
    broken["order_items"] = broken["order_items"].withColumn("unit_price", F.col("unit_price") + 1)
    with pytest.raises(AssertionError, match="price_not_product_price"):
        sales_nb.check_sales(spark, broken, AS_OF)

    broken = dict(warehouse["sales"])
    broken["returns"] = broken["returns"].withColumn("product_id", (F.col("product_id") % 96) + 1)
    with pytest.raises(AssertionError, match="return_product_mismatch"):
        sales_nb.check_sales(spark, broken, AS_OF)

    hr_nb = warehouse["nb"][1]
    broken = dict(warehouse["hr"])
    broken["payroll"] = broken["payroll"].withColumn("net_pay", F.col("net_pay") + 1)
    with pytest.raises(AssertionError, match="net_pay_formula"):
        hr_nb.check_hr(spark, broken, AS_OF)

    fin_nb = warehouse["nb"][2]
    broken = dict(warehouse["fin"])
    broken["invoices"] = broken["invoices"].withColumn("customer_id", (F.col("customer_id") % 500) + 1)
    with pytest.raises(AssertionError, match="invoice_customer_not_order_customer"):
        fin_nb.check_finance(spark, broken, AS_OF, orders=warehouse["sales"]["orders"], employees=warehouse["hr"]["employees"])


def test_data_is_deterministic(spark, warehouse):
    sales_nb = warehouse["nb"][0]
    again = sales_nb.build_sales(spark, AS_OF)
    for name in ("orders", "order_items", "customers", "returns", "inventory"):
        stored = warehouse["sales"][name]
        assert again[name].exceptAll(stored).count() == 0, name
        assert stored.exceptAll(again[name]).count() == 0, name


def test_as_of_date_drives_every_date(spark, warehouse):
    sales_nb = warehouse["nb"][0]
    other = sales_nb.build_sales(spark, "2025-12-31")
    from pyspark.sql import functions as F

    latest = other["orders"].agg(F.max("order_date")).first()[0].isoformat()
    assert latest <= "2025-12-31"
    assert other["inventory"].agg(F.max("snapshot_date")).first()[0].isoformat() == "2025-12-31"


def test_no_wall_clock_in_notebooks():
    for path in NOTEBOOKS.glob("0[0-3]_*.py"):
        text = path.read_text().lower()
        assert "current_date" not in text and "current_timestamp" not in text and "datetime.now" not in text, path.name
