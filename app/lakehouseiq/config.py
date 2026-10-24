"""Runtime settings, read from environment variables.

Nothing here touches the network and nothing is required at import time:
``DATABRICKS_WAREHOUSE_ID`` is only demanded when a connection is actually made.
Catalog and endpoint names are configuration; no other module hardcodes them.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_CATALOG = "workspace"
DEFAULT_GENERATOR_ENDPOINT = "databricks-gpt-oss-120b"
DEFAULT_JUDGE_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Settings:
    catalog: str = DEFAULT_CATALOG
    warehouse_id: str | None = None
    generator_endpoint: str = DEFAULT_GENERATOR_ENDPOINT
    judge_endpoint: str = DEFAULT_JUDGE_ENDPOINT
    max_rows: int = 500
    query_timeout_s: int = 120
    mlflow_experiment_id: str | None = None
    as_of_date: str | None = None

    def __post_init__(self) -> None:
        # The catalog is interpolated into SQL identifiers, so it must be a plain name.
        if not _IDENT_RE.match(self.catalog):
            raise ValueError(f"CATALOG must be a simple identifier, got {self.catalog!r}")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ

        def get(name: str, default: str | None = None) -> str | None:
            value = env.get(name)
            return value.strip() if value and value.strip() else default

        return cls(
            catalog=get("CATALOG", DEFAULT_CATALOG),
            warehouse_id=get("DATABRICKS_WAREHOUSE_ID"),
            generator_endpoint=get("GENERATOR_ENDPOINT", DEFAULT_GENERATOR_ENDPOINT),
            judge_endpoint=get("JUDGE_ENDPOINT", DEFAULT_JUDGE_ENDPOINT),
            max_rows=int(get("MAX_ROWS", "500")),
            query_timeout_s=int(get("QUERY_TIMEOUT_S", "120")),
            mlflow_experiment_id=get("MLFLOW_EXPERIMENT_ID"),
            as_of_date=get("AS_OF_DATE"),
        )

    def require_warehouse_id(self) -> str:
        if not self.warehouse_id:
            raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not set; bind a SQL warehouse resource to the app.")
        return self.warehouse_id

    def fq(self, schema: str, name: str) -> str:
        """Fully qualified ``catalog.schema.name``."""
        return f"{self.catalog}.{schema}.{name}"
