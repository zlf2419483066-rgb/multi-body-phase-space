from __future__ import annotations

from collections.abc import Mapping
from enum import Enum, IntEnum
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import phase_space_graph5.catalog_checkpoint as checkpoint


def _reference_jsonable(value):
    if isinstance(value, tuple):
        return [_reference_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_reference_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _reference_jsonable(item) for key, item in value.items()}
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


class _DictSubclass(dict):
    pass


class _ListSubclass(list):
    pass


class _TupleSubclass(tuple):
    pass


class _MappingSubclass(Mapping):
    def __init__(self, value):
        self._value = dict(value)

    def __getitem__(self, key):
        return self._value[key]

    def __iter__(self):
        return iter(self._value)

    def __len__(self):
        return len(self._value)


class _BrokenMapping(Mapping):
    def __getitem__(self, key):
        raise RuntimeError("synthetic mapping failure")

    def __iter__(self):
        raise RuntimeError("synthetic mapping failure")

    def __len__(self):
        return 1


class _StringEnum(Enum):
    VALUE = "value"


class _NumericEnum(IntEnum):
    VALUE = 3


def _capture(serializer, value):
    try:
        converted = serializer(value)
        result = {
            "ok": True,
            "type": (type(converted).__module__, type(converted).__qualname__),
            "repr": repr(converted),
        }
        try:
            result["json"] = json.dumps(converted, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":"))
        except Exception as exc:
            result["json_error"] = (type(exc).__name__, str(exc))
        return result
    except Exception as exc:
        return {"ok": False, "error": (type(exc).__name__, str(exc))}


class CheckpointJsonableTests(unittest.TestCase):
    def test_fast_jsonable_matches_reference_bytes_and_types(self):
        payload = {
            "primitive": (None, True, 3, 1.25, "x"),
            "nested": {2: ("a", [1, {"b": 2}])},
            "subclasses": (_DictSubclass({"x": 1}), _ListSubclass([1, 2]),
                           _TupleSubclass((1, 2)), _MappingSubclass({"x": 1})),
            "enums": (_StringEnum.VALUE, _NumericEnum.VALUE),
        }
        expected = _reference_jsonable(payload)
        actual = checkpoint._jsonable(payload)
        self.assertEqual(actual, expected)
        expected_bytes = json.dumps(expected, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")
        actual_bytes = checkpoint._json_bytes(payload)
        self.assertEqual(actual_bytes, expected_bytes)
        self.assertEqual(type(checkpoint._jsonable(_DictSubclass({"x": 1}))), dict)
        self.assertEqual(type(checkpoint._jsonable(_ListSubclass([1]))), list)
        self.assertEqual(checkpoint._jsonable(_StringEnum.VALUE), "value")
        self.assertIs(checkpoint._jsonable(_NumericEnum.VALUE), _NumericEnum.VALUE)

    def test_mapping_failure_behavior_is_preserved(self):
        expected = _capture(_reference_jsonable, _BrokenMapping())
        actual = _capture(checkpoint._jsonable, _BrokenMapping())
        self.assertEqual(actual, expected)

    def test_checkpoint_json_bytes_match_reference_serializer(self):
        payload = {
            "schema_version": "F5-catalog-checkpoint-v1",
            "topology_key": ("topology", 5, 7, 0),
            "completion": {"fixed_labels": True, "particle_relabeling": True},
            "identity": {
                "key": ("candidate", "fixed_labels", ((0, 1), (1, 2))),
                "digest": "abc123",
                "source_index": 0,
            },
        }
        expected = json.dumps(_reference_jsonable(payload), ensure_ascii=False,
                              sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(checkpoint._json_bytes(payload), expected)


if __name__ == "__main__":
    unittest.main()
