"""Tests for :mod:`models.baseline` edge cases."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder


def test_classification_report_when_test_split_lacks_a_class():
    """Regression: sklearn raises if target_names outnumber labels present in y_test."""
    le = LabelEncoder()
    le.fit(["healthy", "monitor", "warning", "critical"])
    # Test fold missing "critical" — the failure mode seen on small datasets.
    y_true = le.transform(["healthy", "monitor", "warning"])
    y_pred = le.transform(["healthy", "monitor", "warning"])
    n_classes = len(le.classes_)
    all_labels = np.arange(n_classes)

    f1m = float(
        f1_score(y_true, y_pred, average="macro", labels=all_labels, zero_division=0)
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=all_labels,
        target_names=list(le.classes_),
        zero_division=0,
        output_dict=True,
    )

    assert f1m > 0.0
    assert "critical" in report
    assert report["critical"]["support"] == 0.0
