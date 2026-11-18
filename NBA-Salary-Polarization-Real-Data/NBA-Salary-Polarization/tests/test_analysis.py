from utils.analysis import gini

def test_gini_equal():
    assert gini([10, 10, 10]) == 0

def test_gini_unequal():
    assert abs(gini([0, 0, 10]) - 2/3) < 1e-9
