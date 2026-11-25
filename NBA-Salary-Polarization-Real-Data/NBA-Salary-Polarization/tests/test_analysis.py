"""Unit tests for the inequality / polarization measures and data cleaning.

Run from the project folder:  python -m pytest
"""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.analysis import FEATURES, clean_data, gini, inequality, lorenz_at, middle_share, wolfson


def test_gini_equal():
    assert gini([10, 10, 10]) == 0


def test_gini_unequal():
    assert abs(gini([0, 0, 10]) - 2 / 3) < 1e-9


def test_lorenz_bottom_half():
    # Bottom half of [1, 1, 3, 3] holds 2 of 8 = 25% of the total.
    assert lorenz_at([3, 1, 3, 1], 0.5) == pytest.approx(0.25)


def test_wolfson_equal_is_zero():
    assert wolfson([5, 5, 5, 5]) == pytest.approx(0.0)


def test_wolfson_hand_computed():
    # [1, 1, 3, 3]: mean = median = 2, L(0.5) = 0.25 -> T = 0.25, Gini = 0.25.
    # P = 2 * (2 * 0.25 - 0.25) * 2 / 2 = 0.5
    assert wolfson([1, 1, 3, 3]) == pytest.approx(0.5)


def test_wolfson_increases_when_middle_empties():
    clustered_middle = [4, 5, 5, 5, 5, 6]
    two_poles = [1, 1, 5, 5, 9, 9]
    assert wolfson(two_poles) > wolfson(clustered_middle)


def test_middle_share():
    # median = 10; band is [5, 15] -> 10, 12 and 5 are inside, 1 and 40 are not.
    assert middle_share([1, 5, 10, 12, 40]) == pytest.approx(0.6)


def make_frame():
    row = {f: 1.0 for f in FEATURES}
    rows = [dict(row, player='A', season=2025, salary=100, salary_cap=1000, cap_share=0.1),
            dict(row, player='A', season=2025, salary=100, salary_cap=1000, cap_share=0.1),  # exact copy
            dict(row, player='A', season=2025, salary=200, salary_cap=1000, cap_share=0.2),  # same name, different player
            dict(row, player='B', season=2025, salary=None, salary_cap=1000, cap_share=None)]
    return pd.DataFrame(rows)


def test_clean_data_keeps_same_name_players_and_drops_bad_rows():
    out = clean_data(make_frame())
    assert len(out) == 2
    assert out.season.iloc[0] == '2025'


def test_clean_data_reports_missing_columns():
    with pytest.raises(ValueError, match='prepare_data'):
        clean_data(make_frame().drop(columns=['cap_share']))


def test_inequality_columns():
    result = inequality(clean_data(make_frame()))
    assert {'gini', 'top_10_pct_share', 'wolfson', 'middle_share'} <= set(result.columns)
