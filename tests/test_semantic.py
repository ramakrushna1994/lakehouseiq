"""Offline tests for lakehouseiq.semantic and the shipped registry."""

from pathlib import Path

import pytest
import yaml

from lakehouseiq import semantic
from lakehouseiq.semantic import (
    CROSS_FACT_VIEWS,
    fqn,
    known_identifiers,
    known_view_fqns,
    load_registry,
    render_prompt_context,
    render_spec_yaml,
)

SEMANTIC_DIR = Path(__file__).resolve().parent.parent / "app" / "semantic"
EXPECTED = {
    "mv_orders", "mv_order_lines", "mv_returns", "mv_inventory", "mv_employees",
    "mv_payroll", "mv_performance", "mv_budget", "mv_expenses", "mv_invoices",
}


@pytest.fixture(scope="module")
def registry():
    return load_registry(SEMANTIC_DIR)


def test_ten_views_in_the_expected_schemas(registry):
    assert {mv.name for mv in registry} == EXPECTED
    schemas = {mv.name: mv.schema for mv in registry}
    assert schemas["mv_orders"] == "sales" and schemas["mv_payroll"] == "hr" and schemas["mv_budget"] == "finance"


def test_every_view_has_dimensions_and_measures_with_unique_names(registry):
    for mv in registry:
        assert mv.dimensions and mv.measures, mv.name
        names = mv.dimension_names + mv.measure_names
        assert len(names) == len(set(names)), f"duplicate names in {mv.name}"


def test_every_field_has_a_comment_and_an_expression(registry):
    for mv in registry:
        for f in mv.dimensions + mv.measures:
            assert f["comment"].strip(), f"{mv.name}.{f['name']} lacks a comment"
            assert f["expr"].strip()


def test_spec_version_and_dimensions_key(registry):
    for mv in registry:
        assert str(mv.spec["version"]) == "1.1"
        assert "dimensions" in mv.spec and "measures" in mv.spec and "fields" not in mv.spec


def test_joins_quote_the_on_key(registry):
    # an unquoted `on` key parses as boolean True under YAML 1.1
    def walk(joins):
        for j in joins:
            assert "on" in j and True not in j, j["name"]
            walk(j.get("joins", []))

    for mv in registry:
        walk(mv.spec.get("joins", []))


def test_catalog_placeholder_fully_substituted(registry):
    for mv in registry:
        rendered = render_spec_yaml(mv, "my_catalog", "2026-06-30")
        assert "${" not in rendered
        assert "my_catalog." in rendered or mv.name == "mv_invoices"
        body = yaml.safe_load(rendered)
        assert body["source"].startswith("my_catalog.")
        assert body["comment"]


def test_as_of_date_only_in_invoices(registry):
    for mv in registry:
        raw = yaml.safe_dump(mv.spec)
        assert ("${as_of_date}" in raw) == (mv.name == "mv_invoices"), mv.name
    inv = next(m for m in registry if m.name == "mv_invoices")
    assert "DATE'2026-06-30'" in render_spec_yaml(inv, "c", "2026-06-30")


def test_no_other_placeholders_exist(registry):
    import re

    for mv in registry:
        found = set(re.findall(r"\$\{(\w+)\}", yaml.safe_dump(mv.spec)))
        assert found <= {"catalog", "as_of_date"}, (mv.name, found)


def test_render_rejects_bad_inputs(registry):
    mv = registry[0]
    with pytest.raises(ValueError):
        render_spec_yaml(mv, "x; DROP", "2026-06-30")
    with pytest.raises(ValueError):
        render_spec_yaml(mv, "workspace", "yesterday")
    with pytest.raises(ValueError):
        fqn("a.b", mv)


def test_unresolved_placeholder_raises(tmp_path):
    f = tmp_path / "x.yaml"
    f.write_text(
        "name: mv_x\nschema: sales\ndescription: d\nspec:\n  version: 1.1\n  source: ${catalog}.sales.${oops}\n"
        "  dimensions: [{name: a, expr: a, comment: c}]\n  measures: [{name: m, expr: COUNT(1), comment: c}]\n"
    )
    with pytest.raises(ValueError, match="unresolved"):
        render_spec_yaml(load_registry(f)[0], "workspace", "2026-06-30")


def test_missing_required_key_raises(tmp_path):
    (tmp_path / "bad.yaml").write_text("name: mv_x\nschema: sales\n")
    with pytest.raises(ValueError, match="missing required key"):
        load_registry(tmp_path)


def test_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path)


def test_prompt_context_lists_every_field(registry):
    text = render_prompt_context(registry, "workspace")
    for mv in registry:
        assert f"workspace.{mv.schema}.{mv.name} - " in text
        for f in mv.dimensions + mv.measures:
            assert f"- {f['name']}" in text
    assert "MEASURE()" in text


def test_known_identifiers_cover_views_and_fields(registry):
    ids = known_identifiers(registry, "workspace")
    assert "WORKSPACE.SALES.MV_ORDERS" in ids
    assert "MV_ORDERS.ORDER_REVENUE" in ids
    assert "ORDER_MONTH" in ids and "SALES.MV_ORDERS.TOTAL_ORDERS" in ids


def test_known_view_fqns_include_cross_fact(registry):
    names = known_view_fqns(registry, "workspace")
    assert "workspace.finance.mv_budget" in names
    for k in CROSS_FACT_VIEWS:
        assert f"workspace.{k}" in names
    assert len(names) == 12


def test_composed_measures_reference_existing_measures(registry):
    import re

    for mv in registry:
        for m in mv.measures:
            for ref in re.findall(r"MEASURE\((\w+)\)", m["expr"]):
                assert ref in mv.measure_names, (mv.name, m["name"], ref)


def test_semantic_module_is_pure():
    import ast

    tree = ast.parse(Path(semantic.__file__).read_text())
    mods = {n.module if isinstance(n, ast.ImportFrom) else a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert {m.split(".")[0] for m in mods if m} <= {"__future__", "copy", "re", "dataclasses", "pathlib", "yaml"}
