import pytest

from science_agent.jsonutil import JSONExtractionError, extract_json


def test_fenced_block():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```\nDone.') == {"a": 1}


def test_object_in_prose_with_nested_braces_and_strings():
    text = 'Here: {"a": {"b": "x } y"}, "c": [1, 2]} and more text'
    assert extract_json(text) == {"a": {"b": "x } y"}, "c": [1, 2]}


def test_trailing_comma_repaired():
    assert extract_json('{"a": [1, 2,], "b": 3,}') == {"a": [1, 2], "b": 3}


def test_prefers_largest_object():
    assert extract_json('{"x": 1} then {"verdict": "wrong", "confidence": 0.9}') == {
        "verdict": "wrong",
        "confidence": 0.9,
    }


def test_no_json_raises():
    with pytest.raises(JSONExtractionError):
        extract_json("no json here")
