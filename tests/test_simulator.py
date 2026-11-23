import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator import (
    build_distribution_from_history,
    build_distribution_from_table,
    build_monte_carlo_interval_table,
    read_uploaded_dataset,
    run_monte_carlo,
    simulate_one_run,
    suggest_inventory_parameters,
    validate_distribution,
)


VALID = pd.DataFrame({
    "demand": [0, 2, 4],
    "probability": [0.2, 0.3, 0.5],
})


def test_valid_distribution():
    result = validate_distribution(VALID)
    assert result["probability"].sum() == pytest.approx(1.0)


def test_invalid_probability_total():
    bad = pd.DataFrame({"demand": [1, 2], "probability": [0.2, 0.2]})
    with pytest.raises(ValueError):
        validate_distribution(bad)


def test_simulation_has_expected_number_of_days():
    result = simulate_one_run(VALID, days=12, seed=7)
    assert len(result) == 12
    assert (result["ending_stock"] >= 0).all()


def test_reproducible_monte_carlo():
    summary_a, results_a = run_monte_carlo(VALID, days=5, replications=10, seed=99)
    summary_b, results_b = run_monte_carlo(VALID, days=5, replications=10, seed=99)
    assert summary_a == summary_b
    pd.testing.assert_frame_equal(results_a, results_b)


def test_import_csv_semicolon_and_history_distribution():
    csv_bytes = b"tanggal;permintaan\n2026-01-01;10\n2026-01-02;20\n2026-01-03;10\n2026-01-04;20\n"
    df, sheets = read_uploaded_dataset(csv_bytes, "riwayat.csv")
    assert sheets == []
    dist, stats = build_distribution_from_history(df, "permintaan")
    assert stats["valid_rows"] == 4
    assert list(dist["demand"]) == [10, 20]
    assert dist["probability"].sum() == pytest.approx(1.0)


def test_import_table_with_indonesian_decimals_and_normalization():
    df = pd.DataFrame({"permintaan": ["5", "10"], "frekuensi": ["20", "30"]})
    dist, info = build_distribution_from_table(df, "permintaan", "frekuensi", auto_normalize=True)
    assert info["was_normalized"] is True
    assert list(dist["probability"]) == pytest.approx([0.4, 0.6])


def test_interval_table_and_parameter_suggestion():
    intervals = build_monte_carlo_interval_table(VALID)
    assert len(intervals) == len(VALID)
    assert intervals["Probabilitas Kumulatif"].iloc[-1] == pytest.approx(1.0)
    suggestion = suggest_inventory_parameters(VALID, lead_time=3)
    assert suggestion["initial_stock"] > 0
    assert suggestion["reorder_point"] > 0
    assert suggestion["order_quantity"] > 0
