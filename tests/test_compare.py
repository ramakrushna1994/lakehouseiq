"""Offline tests for eval.compare."""

import datetime as dt
from decimal import Decimal

import pandas as pd

from eval.compare import results_match


def df(**cols):
    return pd.DataFrame(cols)


def test_identical_frames_match():
    a = df(x=["a", "b"], y=[1, 2])
    assert results_match(a, a.copy(), True) == (True, "match")


def test_column_names_are_ignored():
    assert results_match(df(a=[1, 2]), df(zzz=[1, 2]), True)[0]


def test_numbers_rounded_to_two_places():
    assert results_match(df(v=[1.234]), df(v=[1.2349]), True)[0]
    assert not results_match(df(v=[1.23]), df(v=[1.25]), True)[0]


def test_decimal_and_int_and_float_equal():
    assert results_match(df(v=[Decimal("5.00")]), df(v=[5]), True)[0]
    assert results_match(df(v=[5]), df(v=[5.0]), True)[0]


def test_strings_trimmed_and_casefolded():
    assert results_match(df(s=["  Alice "]), df(s=["alice"]), True)[0]


def test_dates_compare_as_iso_strings():
    assert results_match(df(d=[dt.date(2026, 1, 1)]), df(d=["2026-01-01"]), True)[0]
    assert results_match(df(d=[dt.date(2026, 1, 1)]), df(d=[pd.Timestamp("2026-01-01")]), True)[0]
    assert not results_match(df(d=[dt.date(2026, 1, 1)]), df(d=[dt.date(2026, 1, 2)]), True)[0]


def test_none_equals_nan():
    assert results_match(df(v=[None, 1.0]), df(v=[float("nan"), 1.0]), True)[0]
    assert results_match(df(v=["a", None]), df(v=["a", float("nan")]), True)[0]


def test_row_count_mismatch():
    ok, reason = results_match(df(v=[1, 2]), df(v=[1]), False)
    assert not ok and "row count" in reason


def test_order_matters_true_detects_reordering():
    exp, act = df(v=[1, 2, 3]), df(v=[3, 2, 1])
    assert not results_match(exp, act, True)[0]
    assert results_match(exp, act, False)[0]


def test_unordered_compares_multisets():
    assert not results_match(df(v=[1, 1, 2]), df(v=[1, 2, 2]), False)[0]


def test_column_permutation_matches():
    exp = df(a=["x", "y"], b=[1, 2])
    act = df(p=[1, 2], q=["x", "y"])
    assert results_match(exp, act, True)[0]


def test_column_permutation_keeps_rows_aligned():
    exp = df(a=["x", "y"], b=[1, 2])
    act = df(p=[2, 1], q=["x", "y"])  # values per column fine, rows swapped
    ok, reason = results_match(exp, act, False)
    assert not ok and "line up" in reason


def test_extra_actual_columns_allowed():
    exp = df(a=["x", "y"])
    act = df(extra=[9, 9], name=["x", "y"], more=[0, 1])
    assert results_match(exp, act, True)[0]


def test_missing_expected_column_fails():
    ok, reason = results_match(df(a=[1], b=[2]), df(a=[1]), True)
    assert not ok and "column count" in reason
    ok, reason = results_match(df(a=[1], b=[2]), df(a=[1], c=[3]), True)
    assert not ok and "no actual column" in reason


def test_duplicate_valued_columns_need_distinct_actuals():
    exp = df(a=[1, 2], b=[1, 2])
    assert not results_match(exp, df(a=[1, 2], c=[5, 6]), True)[0]
    assert results_match(exp, df(a=[1, 2], c=[1, 2]), True)[0]


def test_more_than_six_columns_requires_same_order():
    cols = {f"c{i}": [i, i + 10] for i in range(7)}
    exp = pd.DataFrame(cols)
    assert results_match(exp, exp.copy(), True)[0]
    swapped = exp[["c1", "c0", *[f"c{i}" for i in range(2, 7)]]]
    assert not results_match(exp, swapped, True)[0]


def test_empty_results_match():
    assert results_match(pd.DataFrame({"a": []}), pd.DataFrame({"b": []}), False)[0]


def test_none_dataframe_is_a_mismatch():
    assert not results_match(None, df(a=[1]), True)[0]


def test_numpy_scalars_and_bools():
    import numpy as np

    assert results_match(df(v=np.array([1, 2], dtype="int64")), df(v=[1.0, 2.0]), True)[0]
    assert results_match(df(v=[True, False]), df(v=[1, 0]), True)[0]


def test_ignore_zero_rows_accepts_extra_zero_bucket():
    import pandas as pd

    from eval.compare import results_match

    expected = pd.DataFrame({"bucket": ["0-30", "31-60"], "bal": [100.0, 50.0]})
    actual = pd.DataFrame({"bucket": ["0-30", "31-60", "Not outstanding"], "bal": [100.0, 50.0, 0.0]})
    assert results_match(expected, actual, order_matters=False)[0] is False
    assert results_match(expected, actual, order_matters=False, ignore_zero_rows=True)[0] is True


def test_ignore_zero_rows_still_rejects_nonzero_extra_row():
    import pandas as pd

    from eval.compare import results_match

    expected = pd.DataFrame({"bucket": ["0-30"], "bal": [100.0]})
    actual = pd.DataFrame({"bucket": ["0-30", "Not outstanding"], "bal": [100.0, 7.0]})
    assert results_match(expected, actual, order_matters=False, ignore_zero_rows=True)[0] is False


def test_month_label_equals_first_of_month_timestamp():
    import pandas as pd

    from eval.compare import results_match

    expected = pd.DataFrame({"m": pd.to_datetime(["2026-01-01", "2026-02-01"]), "v": [1.0, 2.0]})
    for actual_month in (["2026-01", "2026-02"], ["2026-01-01", "2026-02-01"], ["2026-01-01 00:00:00", "2026-02-01 00:00:00"]):
        actual = pd.DataFrame({"m": actual_month, "v": [1.0, 2.0]})
        assert results_match(expected, actual, order_matters=True)[0], actual_month
