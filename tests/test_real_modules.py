from datetime import datetime, timedelta

import networkx as nx
import pytest

from src.algorithms.eval.metrics import (
    direction_of_change_agreement,
    mae,
    pearson,
    spearman,
)
from src.common.types import SignedEdge, Window


def test_metrics_return_expected_values() -> None:
    truth = [1, 2, 3]
    prediction = [1, 2, 4]
    assert mae(truth, prediction) == pytest.approx(1 / 3)
    assert pearson(truth, prediction) == pytest.approx(0.9819805)
    assert spearman(truth, prediction) == pytest.approx(1.0)
    assert direction_of_change_agreement(truth, prediction) == 1.0


def test_shared_types_validate_contract() -> None:
    window = Window(datetime(2025, 1, 1), datetime(2025, 1, 2))
    edge = SignedEdge("a", "b", -1, window.start, 0.8)
    assert edge.sign == -1
    assert window.end - window.start == timedelta(days=1)


def test_shared_types_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        Window(datetime(2025, 1, 2), datetime(2025, 1, 1))
    with pytest.raises(ValueError):
        SignedEdge("a", "b", 0)
