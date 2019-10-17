from isofit.common import max_table_size, binary_table, eps, combos


def test_max_table_size():
    assert max_table_size == 500


def test_binary_table():
    assert len(binary_table) == 7


def test_eps():
    assert eps == 1e-5


def test_combos():
    inds = [[1, 2], [3, 4, 5]]
    result = [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]
    assert all([a == b for a, b in zip(combos(inds), result)])
