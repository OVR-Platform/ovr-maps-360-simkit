"""Tests for how probe outcomes are aggregated into a verdict.

The aggregation has to fail loudly on degenerate input: a report with no probes
must not read as "no leaks detected".
"""

import numpy as np
import pytest

from simkit.physics.probes import PhysicsReport, ProbeResult


def _probe(**overrides) -> ProbeResult:
    defaults = {
        "x": 0.0,
        "y": 0.0,
        "ground_z": 0.0,
        "settled": True,
        "fell_through": False,
        "penetration_m": 0.002,
        "final_z": 0.058,
        "settle_time_s": 0.6,
    }
    return ProbeResult(**{**defaults, **overrides})


def test_empty_report_is_a_failure_not_a_clean_sheet():
    report = PhysicsReport()

    assert report.leak_rate == 1.0  # not 0.0
    assert report.settle_rate == 0.0


def test_leak_rate_counts_probes_that_fell_through():
    report = PhysicsReport(probes=[_probe(), _probe(fell_through=True, settled=False), _probe()])

    assert report.leak_rate == pytest.approx(1 / 3)


def test_penetration_ignores_probes_that_fell_through():
    """A probe in free fall has no meaningful penetration depth."""
    report = PhysicsReport(
        probes=[
            _probe(penetration_m=0.004),
            _probe(fell_through=True, settled=False, penetration_m=float("nan")),
            _probe(penetration_m=0.011),
        ]
    )

    assert report.max_penetration_m == pytest.approx(0.011)


def test_settle_rate_reflects_unsettled_probes():
    report = PhysicsReport(probes=[_probe(), _probe(settled=False), _probe(), _probe()])

    assert report.settle_rate == pytest.approx(0.75)


def test_summary_dict_reports_nan_when_nothing_settled():
    report = PhysicsReport(probes=[_probe(settled=False, settle_time_s=float("nan"))])
    summary = report.as_dict()

    assert summary["probes"] == 1
    assert np.isnan(summary["median_settle_time_s"])
