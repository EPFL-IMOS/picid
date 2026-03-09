"""Tests for picid.cli.assemble."""

from picid.cli.assemble import main, run_assemble


def test_assemble_imports():
    """run_assemble is importable and callable."""
    from picid.cli.assemble import run_assemble

    assert callable(run_assemble)


def test_main_exists_and_callable():
    """main exists and is callable."""
    assert callable(main)
