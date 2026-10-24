# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Seed sales data
# MAGIC
# MAGIC Creates `sales.products`, `customers`, `stores`, `sales_channels`, `promotions`, `orders`,
# MAGIC `order_items`, `returns` and `inventory` in the catalog given by the `catalog` widget.
# MAGIC
# MAGIC **Deterministic.** Every date is computed from the `as_of_date` widget, never from the clock, and every
# MAGIC random column uses a fixed seed on a single-partition range, so reruns produce identical data. Tables that
# MAGIC are derived after a join use a hash-based draw instead, which does not depend on partitioning.
# MAGIC
# MAGIC **Integrity rules the data guarantees** (asserted at the end of this notebook):
# MAGIC * `ship_date` is never before `order_date` (normally 1-3 days after), and `delivery_date` is never before
# MAGIC   `ship_date` (normally 2-7 days after); both are null for Processing
# MAGIC   and Cancelled orders, and delivery is null while an order is only Shipped
# MAGIC * `order_items.unit_price` equals the product's list price, and nothing is sold before the product launched
# MAGIC * `orders.subtotal` is the sum of its lines, `total_amount = subtotal - discount + tax`, and every order has at
# MAGIC   least one line
# MAGIC * a promotion is only applied inside its date window and above its minimum order value
# MAGIC * every return points at a real order line of a non-cancelled order, inherits that line's order and product,
# MAGIC   happens 1-30 days after delivery (or the order date), and never refunds more than the line total
# MAGIC * inventory is one row per product, store and weekly snapshot, and reserved stock never exceeds stock on hand

# COMMAND ----------

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

CHANNELS = [
    ('Website', 'Online'),
    ('Mobile App', 'Online'),
    ('In-Store', 'Retail'),
    ('Distributor', 'Wholesale'),
    ('Partner Portal', 'Partner'),
]

STORES = [
    ('NYC Flagship', 'New York', 'NY', 'East', '2018-03-01'),
    ('Brooklyn Hub', 'Brooklyn', 'NY', 'East', '2019-07-15'),
    ('Boston Outlet', 'Boston', 'MA', 'East', '2020-01-10'),
    ('Chicago Main', 'Chicago', 'IL', 'Central', '2018-06-01'),
    ('Houston Store', 'Houston', 'TX', 'South', '2019-03-20'),
    ('Dallas Outlet', 'Dallas', 'TX', 'South', '2021-05-01'),
    ('Atlanta Hub', 'Atlanta', 'GA', 'South', '2020-09-15'),
    ('Miami Store', 'Miami', 'FL', 'South', '2022-02-01'),
    ('LA Flagship', 'Los Angeles', 'CA', 'West', '2017-11-01'),
    ('San Francisco', 'San Francisco', 'CA', 'West', '2019-01-20'),
    ('Seattle Hub', 'Seattle', 'WA', 'West', '2020-04-01'),
    ('Denver Outlet', 'Denver', 'CO', 'West', '2021-08-15'),
    ('Phoenix Store', 'Phoenix', 'AZ', 'West', '2022-01-10'),
    ('Minneapolis Hub', 'Minneapolis', 'MN', 'Central', '2021-03-01'),
    ('Kansas City', 'Kansas City', 'MO', 'Central', '2022-06-01'),
    ('Charlotte', 'Charlotte', 'NC', 'East', '2021-11-01'),
    ('Nashville', 'Nashville', 'TN', 'South', '2022-03-15'),
    ('Portland', 'Portland', 'OR', 'West', '2023-01-01'),
    ('Detroit', 'Detroit', 'MI', 'Central', '2022-09-01'),
    ('Columbus', 'Columbus', 'OH', 'Central', '2023-04-01'),
]

# (promo_code, promo_name, discount_type, discount_value, min_order_value, start_date, end_date, channel_id)
PROMOTIONS = [
    ('SAVE10', '10% Off Sitewide', 'PERCENT', 10, 50, '2024-01-01', '2024-01-31', 1),
    ('SAVE15', '15% Off Electronics', 'PERCENT', 15, 200, '2024-02-01', '2024-02-28', 1),
    ('FLASH20', 'Flash Sale 20%', 'PERCENT', 20, 100, '2024-03-15', '2024-03-17', 2),
    ('SPRING25', 'Spring Sale', 'PERCENT', 25, 150, '2024-04-01', '2024-04-30', 1),
    ('FIXED50', '$50 Off Orders >$500', 'FIXED', 50, 500, '2024-05-01', '2024-05-31', 1),
    ('SUMMER10', 'Summer Discount', 'PERCENT', 10, 75, '2024-06-01', '2024-08-31', 3),
    ('BACK2SCH', 'Back to School', 'PERCENT', 12, 100, '2024-08-01', '2024-09-15', 2),
    ('FALL15', 'Fall Collection', 'PERCENT', 15, 200, '2024-09-01', '2024-10-31', 1),
    ('BFCM30', 'Black Friday 30% Off', 'PERCENT', 30, 200, '2024-11-29', '2024-11-30', 1),
    ('CYBER25', 'Cyber Monday', 'PERCENT', 25, 150, '2024-12-02', '2024-12-02', 2),
    ('HOLIDAY20', 'Holiday Season', 'PERCENT', 20, 100, '2024-12-10', '2024-12-25', 1),
    ('NEWYEAR', 'New Year Sale', 'PERCENT', 15, 100, '2025-01-01', '2025-01-07', 1),
    ('VIP30', 'VIP Customer 30% Off', 'PERCENT', 30, 300, '2024-01-01', '2025-12-31', 5),
    ('WHOLESALE10', 'Wholesale Discount', 'PERCENT', 10, 1000, '2024-01-01', '2025-12-31', 4),
    ('APP15', 'App-Only 15% Off', 'PERCENT', 15, 50, '2024-01-01', '2025-12-31', 2),
]

# (product_name, category, subcategory, unit_cost, unit_price, supplier, launch_date)
PRODUCTS = [
    ('Laptop Pro 15', 'Electronics', 'Computers', 700, 1299.99, 'TechCorp', '2022-01-01'),
    ('Laptop Air 13', 'Electronics', 'Computers', 500, 999.99, 'TechCorp', '2022-06-01'),
    ('Laptop Budget 11', 'Electronics', 'Computers', 250, 499.99, 'BudgetTech', '2023-01-01'),
    ('Wireless Mouse Pro', 'Electronics', 'Peripherals', 12, 49.99, 'PeriphHub', '2021-03-01'),
    ('Wireless Mouse Basic', 'Electronics', 'Peripherals', 6, 24.99, 'PeriphHub', '2021-03-01'),
    ('USB-C Hub 7-port', 'Electronics', 'Peripherals', 18, 79.99, 'ConnectPro', '2022-05-01'),
    ('USB-C Hub 4-port', 'Electronics', 'Peripherals', 10, 39.99, 'ConnectPro', '2022-05-01'),
    ('Mechanical Keyboard RGB', 'Electronics', 'Peripherals', 40, 129.99, 'KeyMasters', '2021-08-01'),
    ('Mechanical Keyboard TKL', 'Electronics', 'Peripherals', 35, 99.99, 'KeyMasters', '2022-01-01'),
    ('Membrane Keyboard', 'Electronics', 'Peripherals', 10, 39.99, 'KeyMasters', '2023-03-01'),
    ('Monitor 27" 4K', 'Electronics', 'Displays', 200, 599.99, 'DisplayCo', '2021-01-01'),
    ('Monitor 27" FHD', 'Electronics', 'Displays', 130, 349.99, 'DisplayCo', '2021-01-01'),
    ('Monitor 24" FHD', 'Electronics', 'Displays', 100, 249.99, 'DisplayCo', '2022-03-01'),
    ('Webcam 4K', 'Electronics', 'Peripherals', 35, 119.99, 'CamTech', '2022-01-01'),
    ('Webcam 1080p', 'Electronics', 'Peripherals', 20, 69.99, 'CamTech', '2021-06-01'),
    ('Headphones ANC Pro', 'Electronics', 'Audio', 80, 299.99, 'SoundWave', '2021-04-01'),
    ('Headphones ANC Basic', 'Electronics', 'Audio', 50, 179.99, 'SoundWave', '2022-01-01'),
    ('Earbuds Wireless Pro', 'Electronics', 'Audio', 40, 149.99, 'SoundWave', '2022-06-01'),
    ('Earbuds Wireless Basic', 'Electronics', 'Audio', 20, 79.99, 'SoundWave', '2023-01-01'),
    ('Tablet 12"', 'Electronics', 'Tablets', 150, 499.99, 'TechCorp', '2022-03-01'),
    ('Tablet 10"', 'Electronics', 'Tablets', 100, 349.99, 'TechCorp', '2022-03-01'),
    ('Tablet 8"', 'Electronics', 'Tablets', 70, 249.99, 'TechCorp', '2023-01-01'),
    ('Smartwatch Pro', 'Electronics', 'Wearables', 80, 299.99, 'WearTech', '2022-01-01'),
    ('Smartwatch Basic', 'Electronics', 'Wearables', 40, 149.99, 'WearTech', '2022-06-01'),
    ('Portable Charger 20000', 'Electronics', 'Accessories', 15, 59.99, 'PowerUp', '2021-01-01'),
    ('Portable Charger 10000', 'Electronics', 'Accessories', 8, 34.99, 'PowerUp', '2021-01-01'),
    ('Phone Stand Adjustable', 'Electronics', 'Accessories', 5, 19.99, 'DeskGear', '2022-01-01'),
    ('Laptop Stand Foldable', 'Electronics', 'Accessories', 12, 44.99, 'DeskGear', '2022-01-01'),
    ('Cable USB-C 2m', 'Electronics', 'Accessories', 3, 14.99, 'ConnectPro', '2021-01-01'),
    ('Cable HDMI 2m', 'Electronics', 'Accessories', 5, 19.99, 'ConnectPro', '2021-01-01'),
    ('Ergonomic Chair Pro', 'Furniture', 'Seating', 280, 799.99, 'ErgoFurn', '2020-06-01'),
    ('Ergonomic Chair Standard', 'Furniture', 'Seating', 180, 499.99, 'ErgoFurn', '2021-01-01'),
    ('Task Chair Basic', 'Furniture', 'Seating', 90, 249.99, 'ErgoFurn', '2022-01-01'),
    ('Standing Desk Electric 60"', 'Furniture', 'Desks', 350, 999.99, 'DeskMakers', '2021-01-01'),
    ('Standing Desk Manual 48"', 'Furniture', 'Desks', 200, 599.99, 'DeskMakers', '2021-06-01'),
    ('Fixed Desk 60"', 'Furniture', 'Desks', 120, 349.99, 'DeskMakers', '2022-01-01'),
    ('Bookshelf 5-Tier', 'Furniture', 'Storage', 100, 299.99, 'ShelfCo', '2021-01-01'),
    ('Bookshelf 3-Tier', 'Furniture', 'Storage', 60, 179.99, 'ShelfCo', '2022-01-01'),
    ('Filing Cabinet 4-Drawer', 'Furniture', 'Storage', 120, 349.99, 'ShelfCo', '2021-01-01'),
    ('Filing Cabinet 2-Drawer', 'Furniture', 'Storage', 70, 199.99, 'ShelfCo', '2022-01-01'),
    ('Whiteboard 4x3', 'Furniture', 'Office Accessories', 50, 149.99, 'OfficePro', '2021-01-01'),
    ('Whiteboard 6x4', 'Furniture', 'Office Accessories', 80, 249.99, 'OfficePro', '2021-01-01'),
    ('Desk Lamp LED Dimmable', 'Furniture', 'Lighting', 25, 79.99, 'BrightHome', '2021-01-01'),
    ('Desk Lamp Basic', 'Furniture', 'Lighting', 10, 29.99, 'BrightHome', '2022-01-01'),
    ('Monitor Arm Dual', 'Furniture', 'Mounts', 40, 129.99, 'MountPro', '2022-01-01'),
    ('Monitor Arm Single', 'Furniture', 'Mounts', 25, 79.99, 'MountPro', '2022-01-01'),
    ('Notebook Premium A4 (Pack 5)', 'Stationery', 'Paper', 8, 24.99, 'PaperCo', '2021-01-01'),
    ('Notebook Basic A4 (Pack 10)', 'Stationery', 'Paper', 6, 14.99, 'PaperCo', '2021-01-01'),
    ('Pen Set Luxury 12', 'Stationery', 'Writing', 12, 34.99, 'InkMasters', '2021-01-01'),
    ('Pen Set Standard 24', 'Stationery', 'Writing', 6, 14.99, 'InkMasters', '2022-01-01'),
    ('Marker Set 20 Colors', 'Stationery', 'Writing', 8, 19.99, 'InkMasters', '2022-01-01'),
    ('Sticky Notes 12-pack', 'Stationery', 'Office Supplies', 4, 9.99, 'PaperCo', '2021-01-01'),
    ('Planner 2025 Premium', 'Stationery', 'Planning', 15, 39.99, 'PaperCo', '2025-01-01'),
    ('Planner 2025 Basic', 'Stationery', 'Planning', 7, 17.99, 'PaperCo', '2025-01-01'),
    ('Binder A4 Pack 5', 'Stationery', 'Office Supplies', 8, 19.99, 'PaperCo', '2022-01-01'),
    ('Stapler Heavy Duty', 'Stationery', 'Office Supplies', 12, 29.99, 'OfficePro', '2021-01-01'),
    ('Coffee Maker Espresso Pro', 'Appliances', 'Coffee', 60, 199.99, 'BrewMaster', '2021-06-01'),
    ('Coffee Maker Drip', 'Appliances', 'Coffee', 35, 99.99, 'BrewMaster', '2022-01-01'),
    ('Coffee Maker Pod', 'Appliances', 'Coffee', 45, 149.99, 'BrewMaster', '2022-06-01'),
    ('Water Purifier UV', 'Appliances', 'Water', 120, 349.99, 'PureWater', '2021-01-01'),
    ('Water Purifier RO', 'Appliances', 'Water', 180, 499.99, 'PureWater', '2022-01-01'),
    ('Air Purifier HEPA Large', 'Appliances', 'Air', 90, 299.99, 'CleanAir', '2021-01-01'),
    ('Air Purifier HEPA Small', 'Appliances', 'Air', 50, 149.99, 'CleanAir', '2022-01-01'),
    ('Desk Fan 12"', 'Appliances', 'Climate', 18, 59.99, 'CoolBreeze', '2021-01-01'),
    ('Desk Fan 8"', 'Appliances', 'Climate', 10, 34.99, 'CoolBreeze', '2022-01-01'),
    ('Mini Fridge 20L', 'Appliances', 'Refrigeration', 80, 229.99, 'CoolTech', '2022-01-01'),
    ('Mini Fridge 10L', 'Appliances', 'Refrigeration', 50, 149.99, 'CoolTech', '2023-01-01'),
    ('Microwave Compact', 'Appliances', 'Kitchen', 60, 189.99, 'HomeApp', '2022-01-01'),
    ('Electric Kettle', 'Appliances', 'Kitchen', 20, 59.99, 'HomeApp', '2021-01-01'),
    ('Toaster 4-Slice', 'Appliances', 'Kitchen', 25, 69.99, 'HomeApp', '2022-01-01'),
    ('Blender Pro', 'Appliances', 'Kitchen', 40, 129.99, 'HomeApp', '2022-06-01'),
    ('Shredder 10-Sheet', 'Electronics', 'Office Equipment', 50, 149.99, 'OfficeTech', '2021-01-01'),
    ('Shredder 6-Sheet', 'Electronics', 'Office Equipment', 30, 89.99, 'OfficeTech', '2022-01-01'),
    ('Printer Laser Mono', 'Electronics', 'Printing', 150, 449.99, 'PrintPro', '2021-01-01'),
    ('Printer Inkjet Color', 'Electronics', 'Printing', 80, 249.99, 'PrintPro', '2022-01-01'),
    ('Scanner Flatbed A4', 'Electronics', 'Printing', 70, 199.99, 'PrintPro', '2022-01-01'),
    ('External SSD 1TB', 'Electronics', 'Storage Devices', 70, 199.99, 'DataDrive', '2022-01-01'),
    ('External SSD 500GB', 'Electronics', 'Storage Devices', 40, 119.99, 'DataDrive', '2022-01-01'),
    ('External HDD 2TB', 'Electronics', 'Storage Devices', 50, 129.99, 'DataDrive', '2021-06-01'),
    ('USB Flash Drive 128GB', 'Electronics', 'Storage Devices', 8, 29.99, 'DataDrive', '2022-01-01'),
    ('USB Flash Drive 64GB', 'Electronics', 'Storage Devices', 4, 14.99, 'DataDrive', '2022-01-01'),
    ('Smart Speaker Large', 'Electronics', 'Smart Home', 60, 199.99, 'SmartHome', '2022-01-01'),
    ('Smart Speaker Mini', 'Electronics', 'Smart Home', 30, 99.99, 'SmartHome', '2022-06-01'),
    ('Smart Plug 4-Pack', 'Electronics', 'Smart Home', 20, 49.99, 'SmartHome', '2022-01-01'),
    ('Security Camera Indoor', 'Electronics', 'Smart Home', 40, 129.99, 'SmartHome', '2022-06-01'),
    ('Security Camera Outdoor', 'Electronics', 'Smart Home', 60, 189.99, 'SmartHome', '2023-01-01'),
    ('Desk Organizer Premium', 'Furniture', 'Office Accessories', 18, 49.99, 'DeskGear', '2022-01-01'),
    ('Desk Organizer Basic', 'Furniture', 'Office Accessories', 8, 22.99, 'DeskGear', '2022-01-01'),
    ('Cable Management Kit', 'Electronics', 'Accessories', 10, 29.99, 'ConnectPro', '2022-01-01'),
    ('Webcam Ring Light', 'Electronics', 'Accessories', 15, 44.99, 'CamTech', '2022-06-01'),
    ('Green Screen Collapsible', 'Electronics', 'Accessories', 25, 69.99, 'CamTech', '2023-01-01'),
    ('Ergonomic Mouse Vertical', 'Electronics', 'Peripherals', 25, 79.99, 'PeriphHub', '2022-06-01'),
    ('Trackpad Wireless', 'Electronics', 'Peripherals', 30, 89.99, 'PeriphHub', '2023-01-01'),
    ('Numeric Keypad Wireless', 'Electronics', 'Peripherals', 15, 44.99, 'KeyMasters', '2023-01-01'),
    ('Document Holder', 'Furniture', 'Office Accessories', 10, 27.99, 'DeskGear', '2022-01-01'),
    ('Footrest Ergonomic', 'Furniture', 'Seating', 30, 89.99, 'ErgoFurn', '2022-06-01'),
]

FIRST_NAMES = ["Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry", "Iris", "Jack",
               "Karen", "Leo", "Mia", "Nathan", "Olivia", "Paul", "Quinn", "Rachel", "Steve", "Tina"]
LAST_NAMES = ["Johnson", "Smith", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
              "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin"]
REGIONS = ["North", "South", "East", "West"]
STATES = ["NY", "CA", "TX", "FL", "IL", "WA", "GA", "NC", "OH", "CO"]
SEGMENTS = ["Enterprise", "Mid-Market", "SMB"]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer", "Buy Now Pay Later"]
RETURN_REASONS = ["Defective", "Wrong Item", "Changed Mind", "Damaged in Shipping", "Not as Described",
                  "Better Price Found"]
RETURN_STATUSES = ["Pending", "Approved", "Completed"]

N_CUSTOMERS, N_ORDERS, N_RETURNS, N_INVENTORY = 500, 5000, 1500, 10000
INVENTORY_SNAPSHOTS = 13  # weekly

# COMMAND ----------

# MAGIC %md
# MAGIC ## Builders
# MAGIC
# MAGIC `build_sales(spark, as_of_date)` returns every table as a lazy DataFrame. Nothing is written here, which keeps
# MAGIC the logic importable and testable without a Databricks workspace.

# COMMAND ----------

def _pick(r, options):
    """Choose from a python list using a uniform [0,1) column."""
    arr = F.array(*[F.lit(o) for o in options])
    return F.element_at(arr, (F.floor(r * len(options)) + 1).cast("int"))


def _between(r, lo, hi):
    """Integer in [lo, hi] from a uniform [0,1) column."""
    return (F.floor(r * (hi - lo + 1)) + lo).cast("int")


def _hrand(*cols, seed):
    """Uniform [0,1) derived from a hash: stable regardless of partitioning or join order."""
    return F.pmod(F.xxhash64(*cols, F.lit(seed)), F.lit(1000000)) / F.lit(1000000.0)


def _range(spark, n):
    return spark.range(1, n + 1, 1, 1)  # ids 1..n, one partition


def build_static(spark, as_of_date):
    channels = spark.createDataFrame(
        [(i + 1, *row) for i, row in enumerate(CHANNELS)], "channel_id int, channel_name string, channel_type string")
    stores = spark.createDataFrame(
        [(i + 1, *row) for i, row in enumerate(STORES)],
        "store_id int, store_name string, city string, state string, region string, opened_date string",
    ).withColumn("opened_date", F.col("opened_date").cast("date"))
    promotions = spark.createDataFrame(
        [(i + 1, r[0], r[1], r[2], float(r[3]), float(r[4]), r[5], r[6], r[7]) for i, r in enumerate(PROMOTIONS)],
        "promo_id int, promo_code string, promo_name string, discount_type string, discount_value double, "
        "min_order_value double, start_date string, end_date string, channel_id int",
    ).select(
        "promo_id", "promo_code", "promo_name", "discount_type",
        F.col("discount_value").cast("decimal(8,2)"), F.col("min_order_value").cast("decimal(10,2)"),
        F.col("start_date").cast("date"), F.col("end_date").cast("date"), "channel_id",
    )
    products = spark.createDataFrame(
        [(i + 1, r[0], r[1], r[2], float(r[3]), float(r[4]), r[5], r[6]) for i, r in enumerate(PRODUCTS)],
        "product_id int, product_name string, category string, subcategory string, unit_cost double, "
        "unit_price double, supplier string, launch_date string",
    ).select(
        "product_id", "product_name", "category", "subcategory",
        F.col("unit_cost").cast("decimal(10,2)"), F.col("unit_price").cast("decimal(10,2)"), "supplier",
        F.lit(True).alias("is_active"), F.col("launch_date").cast("date").alias("launch_date"),
        F.col("launch_date").cast("timestamp").alias("created_at"),
    )
    return {"sales_channels": channels, "stores": stores, "promotions": promotions, "products": products}


def build_customers(spark, as_of_date):
    asof = F.lit(as_of_date).cast("date")
    c = F.col("id")
    return _range(spark, N_CUSTOMERS).select(
        c.cast("int").alias("customer_id"),
        _pick(((c - 1) % 20 + 0.5) / 20, FIRST_NAMES).alias("first_name"),
        _pick(((c - 1) % 15 + 0.5) / 15, LAST_NAMES).alias("last_name"),
        F.concat(F.lit("customer"), c.cast("string"), F.lit("@email.com")).alias("email"),
        F.concat(F.lit("555-"), F.lpad(_between(F.rand(101), 1000, 9999).cast("string"), 4, "0")).alias("phone"),
        _pick(F.rand(102), REGIONS).alias("region"),
        _pick(F.rand(103), STATES).alias("state"),
        _pick(F.rand(104), SEGMENTS).alias("segment"),
        _between(F.rand(105), 1, len(CHANNELS)).alias("channel_id"),
        # Every customer signed up before the first order in the data (orders span 730 days).
        F.date_sub(asof, _between(F.rand(106), 740, 1460)).alias("signup_date"),
    ).withColumn("created_at", F.col("signup_date").cast("timestamp"))


def build_orders_and_items(spark, as_of_date, products, promotions):
    """Lines first, then order totals from the lines, then promotions from the totals."""
    asof = F.lit(as_of_date).cast("date")
    oid = F.col("id")
    r_status = F.rand(204)
    base = _range(spark, N_ORDERS).select(
        oid.cast("int").alias("order_id"),
        _between(F.rand(201), 1, N_CUSTOMERS).alias("customer_id"),
        _between(F.rand(202), 1, len(STORES)).alias("store_id"),
        _between(F.rand(203), 1, len(CHANNELS)).alias("channel_id"),
        F.date_sub(asof, _between(F.rand(205), 0, 729)).alias("order_date"),
        F.when(r_status < 0.1, "Processing").when(r_status < 0.2, "Shipped")
        .when(r_status < 0.3, "Cancelled").otherwise("Completed").alias("status"),
        _pick(F.rand(206), PAYMENT_METHODS).alias("payment_method"),
        _between(F.rand(207), 1, 5).alias("line_count"),
        (F.rand(208) < 0.2).alias("wants_promo"),
        _between(F.rand(209), 1, 3).alias("ship_lag"),
        _between(F.rand(210), 2, 7).alias("delivery_lag"),
    )

    # ---- order_items: 1..5 lines per order -------------------------------------------------
    launch_arr = F.array(*[F.lit(p[6]).cast("date") for p in PRODUCTS])
    lines = (
        base.select("order_id", "order_date", F.explode(F.sequence(F.lit(1), F.col("line_count"))).alias("line_no"))
        .select(
            "order_id", "order_date", "line_no",
            _between(F.rand(301), 1, len(PRODUCTS)).alias("cand_product"),
            _between(F.rand(302), 1, 5).alias("quantity"),
            F.when(F.rand(303) < 0.25, _between(F.rand(304), 5, 30)).otherwise(F.lit(0)).cast("decimal(5,2)")
            .alias("discount_pct"),
        )
        # Nothing is sold before it launched: fall back to an early-catalogue product.
        .withColumn(
            "product_id",
            F.when(F.element_at(launch_arr, F.col("cand_product")) > F.col("order_date"), (F.col("cand_product") % 30) + 1)
            .otherwise(F.col("cand_product")),
        )
        .join(F.broadcast(products.select("product_id", F.col("unit_price"))), "product_id")
        .withColumn(
            "line_total",
            F.round(F.col("quantity") * F.col("unit_price") * (F.lit(100) - F.col("discount_pct")) / F.lit(100), 2)
            .cast("decimal(12,2)"),
        )
    )
    w = Window.orderBy("order_id", "line_no")
    order_items = lines.withColumn("order_item_id", F.row_number().over(w)).select(
        F.col("order_item_id").cast("int"), "order_id", "product_id", "quantity", "unit_price", "discount_pct", "line_total"
    )

    # ---- orders: totals derived from lines ---------------------------------------------------
    sub = order_items.groupBy("order_id").agg(F.sum("line_total").cast("decimal(12,2)").alias("subtotal"))
    pre = base.join(sub, "order_id")

    promo = (
        pre.where("wants_promo")
        .join(
            F.broadcast(promotions.select("promo_id", "discount_type", "discount_value", "min_order_value",
                                          "start_date", "end_date")),
            (F.col("order_date") >= F.col("start_date")) & (F.col("order_date") <= F.col("end_date"))
            & (F.col("subtotal") >= F.col("min_order_value")),
        )
        .withColumn("pick", F.row_number().over(Window.partitionBy("order_id").orderBy(F.xxhash64("order_id", "promo_id"))))
        .where("pick = 1")
        .select("order_id", "promo_id", "discount_type", "discount_value")
    )
    o = pre.join(promo, "order_id", "left")
    discount = (
        F.when(F.col("promo_id").isNull(), F.lit(0))
        .when(F.col("discount_type") == "PERCENT", F.round(F.col("subtotal") * F.col("discount_value") / 100, 2))
        .otherwise(F.col("discount_value"))
    )
    o = o.withColumn("discount_amount", F.least(discount.cast("decimal(10,2)"), F.col("subtotal").cast("decimal(10,2)")))
    o = o.withColumn("tax_amount", F.round(F.lit(0.08) * (F.col("subtotal") - F.col("discount_amount")), 2).cast("decimal(10,2)"))
    o = o.withColumn("total_amount", (F.col("subtotal") - F.col("discount_amount") + F.col("tax_amount")).cast("decimal(12,2)"))

    shipped_out = F.col("status").isin("Processing", "Cancelled")
    ship = F.least(F.date_add(F.col("order_date"), F.col("ship_lag")), asof)
    deliver = F.least(F.date_add(ship, F.col("delivery_lag")), asof)
    orders = o.select(
        "order_id", "customer_id", "store_id", "channel_id", "promo_id", "order_date",
        F.when(shipped_out, F.lit(None).cast("date")).otherwise(ship).alias("ship_date"),
        F.when(shipped_out | (F.col("status") == "Shipped"), F.lit(None).cast("date")).otherwise(deliver)
        .alias("delivery_date"),
        "status", "subtotal", "discount_amount", "tax_amount", "total_amount", "payment_method",
    ).orderBy("order_id")
    return orders, order_items.orderBy("order_item_id")


def build_returns(spark, as_of_date, orders, order_items):
    asof = F.lit(as_of_date).cast("date")
    ok_orders = orders.where(F.col("status") != "Cancelled").select(
        "order_id", "order_date", "delivery_date")
    cand = (
        order_items.join(F.broadcast(ok_orders), "order_id")
        .withColumn("sample_key", F.xxhash64("order_item_id", F.lit(401)))
        .orderBy("sample_key").limit(N_RETURNS)
    )
    anchor = F.coalesce(F.col("delivery_date"), F.col("order_date"))
    ret_date = F.least(F.date_add(anchor, _between(_hrand("order_item_id", seed=402), 1, 30)), asof)
    r_status = _hrand("order_item_id", seed=403)
    w = Window.orderBy("order_item_id")
    return cand.select(
        F.row_number().over(w).cast("int").alias("return_id"),
        "order_item_id", "order_id", "product_id",
        ret_date.alias("return_date"),
        _pick(_hrand("order_item_id", seed=404), RETURN_REASONS).alias("return_reason"),
        _pick(r_status, RETURN_STATUSES).alias("return_status"),
        F.round(F.col("line_total") * (F.lit(0.3) + F.lit(0.7) * _hrand("order_item_id", seed=405)), 2)
        .cast("decimal(10,2)").alias("refund_amount"),
    )


def build_inventory(spark, as_of_date, products, stores):
    asof = F.lit(as_of_date).cast("date")
    weeks = spark.range(0, INVENTORY_SNAPSHOTS).select(F.col("id").cast("int").alias("wk"))
    cand = (
        products.select("product_id")
        .crossJoin(stores.select("store_id"))
        .crossJoin(weeks)
        .withColumn("snapshot_date", F.date_sub(asof, F.col("wk") * 7))
        .withColumn("sample_key", F.xxhash64("product_id", "store_id", "wk", F.lit(501)))
        .orderBy("sample_key").limit(N_INVENTORY)
    )
    on_hand = _between(_hrand("product_id", "store_id", "wk", seed=502), 0, 500)
    stamp = ["product_id", "store_id", "wk"]
    inv = cand.select(
        "product_id", "store_id", "snapshot_date",
        on_hand.alias("quantity_on_hand"),
        _between(_hrand(*stamp, seed=503), 0, 50).alias("reserved_raw"),
        _between(_hrand(*stamp, seed=504), 10, 50).alias("reorder_point"),
        F.date_sub(F.col("snapshot_date"), _between(_hrand(*stamp, seed=505), 0, 30)).alias("last_restocked"),
    )
    w = Window.orderBy("product_id", "store_id", "snapshot_date")
    return inv.select(
        F.row_number().over(w).cast("int").alias("inventory_id"),
        "product_id", "store_id", "snapshot_date", "quantity_on_hand",
        F.least(F.col("reserved_raw"), F.col("quantity_on_hand")).alias("quantity_reserved"),
        "reorder_point", "last_restocked",
    )


def build_sales(spark: SparkSession, as_of_date: str) -> dict:
    static = build_static(spark, as_of_date)
    customers = build_customers(spark, as_of_date)
    orders, order_items = build_orders_and_items(spark, as_of_date, static["products"], static["promotions"])
    returns = build_returns(spark, as_of_date, orders, order_items)
    inventory = build_inventory(spark, as_of_date, static["products"], static["stores"])
    return {
        **static, "customers": customers, "orders": orders, "order_items": order_items,
        "returns": returns, "inventory": inventory,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Integrity checks
# MAGIC
# MAGIC Each check counts violations and fails loudly. They run against the tables as written, so a regression in the
# MAGIC builders cannot slip into the eval's reference answers unnoticed.

# COMMAND ----------

EXPECTED_ROWS = {
    "products": 96, "customers": 500, "orders": 5000, "returns": 1500, "inventory": 10000,
    "stores": 20, "sales_channels": 5, "promotions": 15,
}


def check_sales(spark, t: dict, as_of_date: str) -> dict:
    """Return violation counts per rule; raise AssertionError if any is non-zero."""
    asof = F.lit(as_of_date).cast("date")
    o, i, r, inv, p, c = t["orders"], t["order_items"], t["returns"], t["inventory"], t["products"], t["customers"]

    def n(df):
        return df.count()

    line_sum = i.groupBy("order_id").agg(F.sum("line_total").alias("s"), F.count("*").alias("lines"))
    ojl = o.join(line_sum, "order_id", "left")
    ri = r.join(i.select("order_item_id", F.col("order_id").alias("i_order"), F.col("product_id").alias("i_product"),
                         "line_total"), "order_item_id", "left")
    ri = ri.join(o.select("order_id", F.col("status").alias("o_status"), F.col("order_date").alias("o_date"),
                          "delivery_date"), "order_id", "left")
    ij = i.join(p.select("product_id", F.col("unit_price").alias("p_price"), "launch_date"), "product_id", "left") \
          .join(o.select("order_id", "order_date"), "order_id", "left")
    inv_key = inv.groupBy("product_id", "store_id", "snapshot_date").count().where("count > 1")

    violations = {
        "ship_before_order": n(o.where(F.col("ship_date") < F.col("order_date"))),
        "delivery_before_ship": n(o.where(F.col("delivery_date") < F.col("ship_date"))),
        "dates_in_future": n(o.where((F.col("ship_date") > asof) | (F.col("delivery_date") > asof)
                                     | (F.col("order_date") > asof))),
        "dates_set_when_not_shipped": n(o.where(F.col("status").isin("Processing", "Cancelled")
                                                & (F.col("ship_date").isNotNull() | F.col("delivery_date").isNotNull()))),
        "delivery_set_when_only_shipped": n(o.where((F.col("status") == "Shipped") & F.col("delivery_date").isNotNull())),
        "missing_ship_date": n(o.where(F.col("status").isin("Shipped", "Completed") & F.col("ship_date").isNull())),
        "missing_delivery_date": n(o.where((F.col("status") == "Completed") & F.col("delivery_date").isNull())),
        "price_not_product_price": n(ij.where(F.col("unit_price") != F.col("p_price"))),
        "sold_before_launch": n(ij.where(F.col("order_date") < F.col("launch_date"))),
        "orders_without_lines": n(ojl.where(F.col("lines").isNull())),
        "subtotal_not_sum_of_lines": n(ojl.where(F.abs(F.col("subtotal") - F.col("s")) > 0.005)),
        "total_formula": n(o.where(F.abs(F.col("total_amount") - (F.col("subtotal") - F.col("discount_amount")
                                                                  + F.col("tax_amount"))) > 0.005)),
        "tax_formula": n(o.where(F.abs(F.col("tax_amount") - F.round(0.08 * (F.col("subtotal") - F.col("discount_amount")), 2)) > 0.005)),
        "promo_outside_window_or_below_minimum": n(
            o.where(F.col("promo_id").isNotNull())
            .join(t["promotions"].select("promo_id", "start_date", "end_date", "min_order_value"), "promo_id")
            .where((F.col("order_date") < F.col("start_date")) | (F.col("order_date") > F.col("end_date"))
                   | (F.col("subtotal") < F.col("min_order_value")))),
        "discount_without_promo": n(o.where(F.col("promo_id").isNull() & (F.col("discount_amount") != 0))),
        "return_line_missing": n(ri.where(F.col("i_order").isNull())),
        "return_order_mismatch": n(ri.where(F.col("order_id") != F.col("i_order"))),
        "return_product_mismatch": n(ri.where(F.col("product_id") != F.col("i_product"))),
        "return_on_cancelled_order": n(ri.where(F.col("o_status") == "Cancelled")),
        "return_not_after_delivery": n(ri.where(F.col("return_date") <= F.coalesce(F.col("delivery_date"), F.col("o_date"))
                                                ).where(F.col("return_date") < asof)),
        "return_too_late": n(ri.where(F.datediff("return_date", F.coalesce("delivery_date", "o_date")) > 30)),
        "refund_over_line_total": n(ri.where(F.col("refund_amount") > F.col("line_total"))),
        "duplicate_returns_per_line": n(r.groupBy("order_item_id").count().where("count > 1")),
        "inventory_duplicate_grain": n(inv_key),
        "reserved_over_on_hand": n(inv.where(F.col("quantity_reserved") > F.col("quantity_on_hand"))),
        "orphan_customer": n(o.join(c, "customer_id", "left_anti")),
        "orphan_store": n(o.join(t["stores"], "store_id", "left_anti")),
        "orphan_channel": n(o.join(t["sales_channels"], "channel_id", "left_anti")),
        "orphan_item_order": n(i.join(o, "order_id", "left_anti")),
        "customer_signup_after_first_order": n(
            o.join(c.select("customer_id", "signup_date"), "customer_id").where(F.col("order_date") < F.col("signup_date"))),
    }
    counts = {name: t[name].count() for name in EXPECTED_ROWS}
    bad_counts = {k: (counts[k], v) for k, v in EXPECTED_ROWS.items() if counts[k] != v}
    items = counts.setdefault("order_items", i.count())
    failed = {k: v for k, v in violations.items() if v}
    assert not bad_counts, f"unexpected row counts (actual, expected): {bad_counts}"
    assert 14000 <= items <= 16000, f"order_items should be ~15,000 (1-5 lines per order), got {items}"
    assert not failed, f"integrity violations: {failed}"
    return {"row_counts": counts, "violations": violations}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Comments
# MAGIC
# MAGIC Table and column comments are read by Genie and by the assistant's prompt, so they say what each field means.

# COMMAND ----------

TABLE_COMMENTS = {
    "products": "Product catalogue: list price, cost, category and supplier. One row per product.",
    "customers": "Customer master: region, segment and acquisition channel. One row per customer.",
    "stores": "Physical stores with city, state and region.",
    "sales_channels": "Sales channels (Website, Mobile App, In-Store, Distributor, Partner Portal) and their type.",
    "promotions": "Promotion definitions with discount type/value, minimum order value and date window.",
    "orders": "Customer orders. One row per order; totals are derived from order_items. Status is Completed, "
              "Shipped, Processing or Cancelled. Exclude Cancelled orders for revenue.",
    "order_items": "Order lines. One row per product on an order; unit_price equals the product list price.",
    "returns": "Product returns. One row per returned order line, 1-30 days after delivery.",
    "inventory": "Weekly stock snapshots per product and store. Quantities are point-in-time: filter to one "
                 "snapshot_date before summing.",
}

COLUMN_COMMENTS = {
    "products": {
        "product_id": "Primary key", "product_name": "Product name", "category": "Top-level category",
        "subcategory": "Subcategory within the category", "unit_cost": "Cost to the company per unit, USD",
        "unit_price": "List price per unit, USD", "supplier": "Supplier name", "is_active": "Whether the product is sold",
        "launch_date": "Date the product launched; no sales before this date", "created_at": "Record creation time",
    },
    "customers": {
        "customer_id": "Primary key", "first_name": "First name", "last_name": "Last name", "email": "Email address",
        "phone": "Phone number", "region": "Sales region: North, South, East or West", "state": "US state code",
        "segment": "Customer segment: Enterprise, Mid-Market or SMB",
        "channel_id": "Acquisition channel, references sales_channels.channel_id",
        "signup_date": "Date the customer signed up", "created_at": "Record creation time",
    },
    "stores": {
        "store_id": "Primary key", "store_name": "Store name", "city": "City", "state": "US state code",
        "region": "Store region: East, Central, South or West", "opened_date": "Date the store opened",
    },
    "sales_channels": {
        "channel_id": "Primary key", "channel_name": "Channel name", "channel_type": "Online, Retail, Wholesale or Partner",
    },
    "promotions": {
        "promo_id": "Primary key", "promo_code": "Code customers enter", "promo_name": "Display name",
        "discount_type": "PERCENT or FIXED", "discount_value": "Percent off, or dollars off for FIXED",
        "min_order_value": "Minimum order subtotal for the promotion to apply, USD",
        "start_date": "First day the promotion is valid", "end_date": "Last day the promotion is valid",
        "channel_id": "Channel the promotion is aimed at",
    },
    "orders": {
        "order_id": "Primary key", "customer_id": "References customers.customer_id",
        "store_id": "References stores.store_id", "channel_id": "References sales_channels.channel_id",
        "promo_id": "References promotions.promo_id; null when no promotion was applied",
        "order_date": "Date the order was placed",
        "ship_date": "Ship date, 1-3 days after order_date; null for Processing and Cancelled orders",
        "delivery_date": "Delivery date, 2-7 days after ship_date; null unless the order is Completed",
        "status": "Completed, Shipped, Processing or Cancelled",
        "subtotal": "Sum of line_total over the order's lines, USD",
        "discount_amount": "Promotion discount, USD", "tax_amount": "8 percent tax on (subtotal - discount), USD",
        "total_amount": "subtotal - discount_amount + tax_amount, USD", "payment_method": "How the customer paid",
    },
    "order_items": {
        "order_item_id": "Primary key", "order_id": "References orders.order_id",
        "product_id": "References products.product_id", "quantity": "Units ordered",
        "unit_price": "List price per unit at time of order (equals products.unit_price), USD",
        "discount_pct": "Line discount percent, 0-30", "line_total": "quantity * unit_price after line discount, USD",
    },
    "returns": {
        "return_id": "Primary key", "order_item_id": "The returned order line", "order_id": "Order of that line",
        "product_id": "Product of that line", "return_date": "Date the return was recorded",
        "return_reason": "Why the item came back", "return_status": "Pending, Approved or Completed",
        "refund_amount": "Refund in USD; never more than the line total",
    },
    "inventory": {
        "inventory_id": "Primary key", "product_id": "References products.product_id",
        "store_id": "References stores.store_id", "snapshot_date": "Weekly snapshot date",
        "quantity_on_hand": "Units physically on hand", "quantity_reserved": "Units reserved; at most quantity_on_hand",
        "reorder_point": "Reorder when on-hand falls to this level", "last_restocked": "Date of last restock",
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
    dbutils.widgets.text("catalog", "workspace")
    dbutils.widgets.text("as_of_date", "2026-06-30")
    catalog = dbutils.widgets.get("catalog")
    as_of_date = dbutils.widgets.get("as_of_date")

    tables = build_sales(spark, as_of_date)
    write_tables(spark, catalog, "sales", tables, TABLE_COMMENTS, COLUMN_COMMENTS)

    written = {name: spark.table(f"{catalog}.sales.{name}") for name in tables}
    for name, df in written.items():
        print(f"{name:16s} {df.count():>8,d} rows")

# COMMAND ----------

if __name__ == "__main__":
    result = check_sales(spark, written, as_of_date)
    print("order_items", result["row_counts"]["order_items"], "rows")
    for rule, count in result["violations"].items():
        print(f"  ok  {rule}: {count}")
    print("All sales integrity checks passed.")
