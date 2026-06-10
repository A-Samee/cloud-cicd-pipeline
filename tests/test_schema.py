"""
tests/test_schema.py — Unit tests for schema-change detection.

These tests exercise the pure-logic function `compare_schemas` from
ingestion.ingestion rather than hitting the network.
"""

import sys
from pathlib import Path

# Ensure the project root is importable regardless of where pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.ingestion import compare_schemas  # noqa: E402


def test_feature_added():
    """
    When the new schema contains a column that the old schema did not,
    it should appear in the 'added' list while 'removed' stays empty.
    """
    old_schema = ["a", "b"]
    new_schema = ["a", "b", "c"]

    result = compare_schemas(old_schema, new_schema)

    assert result["added"] == ["c"], f"Expected added=['c'], got {result['added']}"
    assert result["removed"] == [], f"Expected removed=[], got {result['removed']}"


def test_feature_removed():
    """
    When a column disappears from the schema, it should appear in
    'removed' while 'added' stays empty.
    """
    old_schema = ["a", "b", "c"]
    new_schema = ["a", "c"]

    result = compare_schemas(old_schema, new_schema)

    assert result["removed"] == ["b"], f"Expected removed=['b'], got {result['removed']}"
    assert result["added"] == [], f"Expected added=[], got {result['added']}"
