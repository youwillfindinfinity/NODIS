"""
Unit tests for nodis/cli.py using click.testing.CliRunner.
"""
import pathlib

import pytest
from click.testing import CliRunner

from nodis.cli import main


# ---------------------------------------------------------------------------
# Help output tests
# ---------------------------------------------------------------------------

def test_main_help():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0


def test_simulate_help():
    result = CliRunner().invoke(main, ["simulate", "--help"])
    assert result.exit_code == 0
    assert "topology" in result.output.lower()


def test_run_help():
    result = CliRunner().invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "method" in result.output.lower()


def test_evaluate_help():
    result = CliRunner().invoke(main, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "predicted" in result.output.lower()


def test_enrich_help():
    result = CliRunner().invoke(main, ["enrich", "--help"])
    assert result.exit_code == 0
    output_lower = result.output.lower()
    assert "adjacency" in output_lower or "level" in output_lower


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

def test_simulate_runs(tmp_path):
    """simulate with small n/p/reps writes pkl files to the output directory."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "simulate",
            "--n", "20",
            "--p", "5",
            "--reps", "2",
            "--topology", "random",
            "--out", str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    pkl_files = list(tmp_path.glob("*.pkl"))
    assert len(pkl_files) == 2


def test_run_requires_data():
    """run without --data should exit non-zero (missing required option)."""
    result = CliRunner().invoke(main, ["run"])
    assert result.exit_code != 0


def test_evaluate_requires_predicted():
    """evaluate without args should exit non-zero (missing required options)."""
    result = CliRunner().invoke(main, ["evaluate"])
    assert result.exit_code != 0
