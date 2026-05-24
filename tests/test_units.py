from __future__ import annotations

import struct

import pytest

from stateforge import units
from stateforge.enums import MemoryUnitType
from stateforge.exceptions import ValueTypeError
from stateforge.models import MemoryUnit


class TestMessageFactory:
    def test_basic_construction(self, session_id):
        u = units.message(session_id, key="msg:0", value="hello")
        assert isinstance(u, MemoryUnit)
        assert u.session_id == session_id
        assert u.key == "msg:0"
        assert u.value == "hello"
        assert u.type is MemoryUnitType.MESSAGE
        assert u.source == "user"
        assert u.source_ref is None
        assert u.embedding is None
        assert u.metadata == {}
        assert len(u.id) == 36  # uuid4 string form

    def test_custom_source(self, session_id):
        u = units.message(session_id, "k", "v", source="assistant", source_ref="ai")
        assert u.source == "assistant"
        assert u.source_ref == "ai"

    def test_rejects_bytes_value(self, session_id):
        with pytest.raises(ValueTypeError):
            units.message(session_id, "k", b"raw")


class TestKVFactory:
    def test_default_source_is_agent(self, session_id):
        u = units.kv(session_id, "goal", {"task": "summarize"})
        assert u.type is MemoryUnitType.KV
        assert u.source == "agent"

    def test_nested_dict_value(self, session_id):
        u = units.kv(session_id, "state", {"a": [1, 2, {"b": None}]})
        assert u.value == {"a": [1, 2, {"b": None}]}


class TestEmbeddingFactory:
    def test_list_packed_to_float32(self, session_id):
        vec = [0.0, 1.0, -1.5, 3.14]
        u = units.embedding(session_id, "doc", vec, source="encoder")
        assert u.type is MemoryUnitType.EMBEDDING
        assert u.embedding is not None
        assert len(u.embedding) == 4 * 4  # 4 floats × 4 bytes
        roundtrip = list(struct.unpack(f"<{len(vec)}f", u.embedding))
        # float32 precision is finite; compare with tolerance
        for got, want in zip(roundtrip, vec):
            assert abs(got - want) < 1e-5

    def test_bytes_passthrough(self, session_id):
        raw = struct.pack("<3f", 1.0, 2.0, 3.0)
        u = units.embedding(session_id, "doc", raw, source="encoder")
        assert u.embedding == raw

    def test_bad_bytes_length_rejected(self, session_id):
        with pytest.raises(ValueTypeError, match="multiple of 4"):
            units.embedding(session_id, "doc", b"\x00\x01\x02", source="encoder")

    def test_non_numeric_element_rejected(self, session_id):
        with pytest.raises(ValueTypeError, match="not a number"):
            units.embedding(session_id, "doc", [1.0, "oops", 3.0], source="encoder")  # type: ignore[list-item]

    def test_bool_in_vector_rejected(self, session_id):
        with pytest.raises(ValueTypeError, match="not a number"):
            units.embedding(session_id, "doc", [1.0, True, 3.0], source="encoder")  # type: ignore[list-item]


class TestToolResultAndSummary:
    def test_tool_result_requires_source(self, session_id):
        u = units.tool_result(
            session_id, "search:1", {"hits": 3}, source="tool", source_ref="web_search"
        )
        assert u.type is MemoryUnitType.TOOL_RESULT
        assert u.source == "tool"
        assert u.source_ref == "web_search"

    def test_summary(self, session_id):
        u = units.summary(session_id, "sum:1", "the user wants X", source="summarizer")
        assert u.type is MemoryUnitType.SUMMARY


class TestMetadataHandling:
    def test_metadata_defaults_to_empty_dict(self, session_id):
        u = units.message(session_id, "k", "v")
        assert u.metadata == {}

    def test_metadata_is_copied_not_referenced(self, session_id):
        md = {"a": 1}
        u = units.message(session_id, "k", "v", metadata=md)
        md["a"] = 999
        assert u.metadata == {"a": 1}

    def test_metadata_validated(self, session_id):
        with pytest.raises(ValueTypeError):
            units.message(session_id, "k", "v", metadata={"bad": b"bytes"})


class TestImmutability:
    def test_unit_is_frozen(self, session_id):
        u = units.message(session_id, "k", "v")
        with pytest.raises((AttributeError, Exception)):
            u.key = "other"  # type: ignore[misc]

    def test_unique_ids(self, session_id):
        u1 = units.message(session_id, "k", "v")
        u2 = units.message(session_id, "k", "v")
        assert u1.id != u2.id
