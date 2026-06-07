from tsfoundation.evaluation.metrics import mae, rmse


def test_mae():
    assert mae([1.0, 2.0, 3.0], [1.0, 1.0, 5.0]) == 1.0


def test_rmse():
    assert round(rmse([1.0, 2.0], [2.0, 4.0]), 6) == round((2.5) ** 0.5, 6)

