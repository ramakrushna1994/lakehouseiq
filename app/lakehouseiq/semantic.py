"""Metric-view registry: the single source of truth for the semantic layer.

Each ``app/semantic/*.yaml`` file describes one Unity Catalog metric view. The same
file is read by ``notebooks/04_semantic_layer.py`` (to create the view) and by the
app (to build prompt context and the set of known identifiers), so the prompt can
never drift from what was actually created.

Two placeholders are substituted when a spec is rendered:

* ``${catalog}``     the Unity Catalog catalog (a bundle variable, never hardcoded)
* ``${as_of_date}``  the data's reference date, used only by ``mv_invoices`` to age
                     invoices against the data rather than the wall clock

PURE module: standard library + PyYAML only.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PLACEHOLDER_RE = re.compile(r"\$\{[^}]*\}")

# Pre-aggregated plain views (created by notebook 04) for questions that combine
# facts. They use the CTE-per-fact pattern so joins never fan out.
CROSS_FACT_VIEWS: dict[str, str] = {
    "sales.v_product_performance": (
        "One row per product combining units sold, revenue, margin and returns. "
        "Use for return rate by product, top/bottom products by margin, and "
        "any question that needs sales AND returns together."
    ),
    "sales.v_customer_segments": (
        "One row per customer combining lifetime value, order count, average order "
        "value, returns and days since last order. Use for customer lifetime value, "
        "top customers, and spend by region/segment/acquisition channel."
    ),
}


@dataclass(frozen=True)
class MetricView:
    name: str
    schema: str
    description: str
    dimensions: list[dict] = field(default_factory=list)
    measures: list[dict] = field(default_factory=list)
    spec: dict = field(default_factory=dict)

    @property
    def dimension_names(self) -> list[str]:
        return [d["name"] for d in self.dimensions]

    @property
    def measure_names(self) -> list[str]:
        return [m["name"] for m in self.measures]


def _check_ident(value: str, what: str) -> str:
    if not isinstance(value, str) or not _IDENT_RE.match(value):
        raise ValueError(f"{what} must be a simple identifier, got {value!r}")
    return value


def load_registry(path: str | Path) -> list[MetricView]:
    """Load every ``*.yaml`` in ``path`` (or the single file ``path``), sorted by file name."""
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no metric view specs found under {p}")
    views: list[MetricView] = []
    for f in files:
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        for key in ("name", "schema", "description", "spec"):
            if key not in data:
                raise ValueError(f"{f.name}: missing required key {key!r}")
        spec = data["spec"]
        views.append(
            MetricView(
                name=data["name"],
                schema=data["schema"],
                description=" ".join(str(data["description"]).split()),
                dimensions=[
                    {"name": d["name"], "expr": d["expr"], "comment": d.get("comment", "")}
                    for d in spec.get("dimensions", [])
                ],
                measures=[
                    {
                        "name": m["name"],
                        "expr": m["expr"],
                        "comment": m.get("comment", ""),
                        "synonyms": list(m.get("synonyms", [])),
                    }
                    for m in spec.get("measures", [])
                ],
                spec=spec,
            )
        )
    return views


def fqn(catalog: str, mv: MetricView) -> str:
    return f"{_check_ident(catalog, 'catalog')}.{mv.schema}.{mv.name}"


def _substitute(node, mapping: dict[str, str]):
    if isinstance(node, str):
        for k, v in mapping.items():
            node = node.replace("${" + k + "}", v)
        return node
    if isinstance(node, list):
        return [_substitute(x, mapping) for x in node]
    if isinstance(node, dict):
        return {k: _substitute(v, mapping) for k, v in node.items()}
    return node


def render_spec_yaml(mv: MetricView, catalog: str, as_of_date: str) -> str:
    """Return the metric-view YAML body with placeholders substituted.

    Both values are validated because they end up inside DDL: ``catalog`` must be a
    plain identifier and ``as_of_date`` an ISO date.
    """
    _check_ident(catalog, "catalog")
    if not _DATE_RE.match(str(as_of_date)):
        raise ValueError(f"as_of_date must be YYYY-MM-DD, got {as_of_date!r}")
    spec = _substitute(copy.deepcopy(mv.spec), {"catalog": catalog, "as_of_date": str(as_of_date)})
    left = _PLACEHOLDER_RE.findall(yaml.safe_dump(spec))
    if left:
        raise ValueError(f"{mv.name}: unresolved placeholders {sorted(set(left))}")
    if "comment" not in spec:
        spec = {**spec, "comment": mv.description}
    return yaml.safe_dump(spec, sort_keys=False, width=120, allow_unicode=True)


def render_prompt_context(registry: list[MetricView], catalog: str) -> str:
    """Prompt text: per view ``FQN - description`` then its dimensions and measures."""
    blocks: list[str] = []
    for mv in registry:
        lines = [f"{fqn(catalog, mv)} - {mv.description}", "  dimensions:"]
        for d in mv.dimensions:
            lines.append(f"    - {d['name']}" + (f": {d['comment']}" if d["comment"] else ""))
        lines.append("  measures (use with MEASURE()):")
        for m in mv.measures:
            lines.append(f"    - {m['name']}" + (f": {m['comment']}" if m["comment"] else ""))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def known_view_fqns(registry: list[MetricView], catalog: str) -> list[str]:
    """Fully qualified names of every view the product publishes (metric + cross-fact)."""
    names = [fqn(catalog, mv) for mv in registry]
    names += [f"{_check_ident(catalog, 'catalog')}.{k}" for k in CROSS_FACT_VIEWS]
    return names


def view_descriptions(registry: list[MetricView], catalog: str) -> dict[str, str]:
    out = {fqn(catalog, mv): mv.description for mv in registry}
    out.update({f"{catalog}.{k}": v for k, v in CROSS_FACT_VIEWS.items()})
    return out


def known_identifiers(registry: list[MetricView], catalog: str) -> set[str]:
    """Upper-cased identifiers a query against these views may legitimately use."""
    ids: set[str] = set()
    for mv in registry:
        base = [fqn(catalog, mv), f"{mv.schema}.{mv.name}", mv.name]
        for b in base:
            ids.add(b.upper())
        for col in mv.dimension_names + mv.measure_names:
            ids.add(col.upper())
            for b in base:
                ids.add(f"{b}.{col}".upper())
    return ids
