from __future__ import annotations

import math
from datetime import datetime
from decimal import Decimal
from uuid import UUID

import pytest

from stateforge.exceptions import ValueTypeError
from stateforge.value import validate_json_safe


class TestJSONSafePrimitives:
    @pytest.mark.parametrize(
        "value",
        [None, True, False, 0, 1, -1, 2**62, 0.0, 1.5, -3.14, "", "hello", "🚀"],
    )
    def test_primitives_accepted(self, value):
        validate_json_safe(value)

    def test_non_finite_float_rejected(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueTypeError, match="non-finite float"):
                validate_json_safe(bad)


class TestJSONSafeContainers:
    def test_empty_list_accepted(self):
        validate_json_safe([])

    def test_empty_dict_accepted(self):
        validate_json_safe({})

    def test_nested_structures_accepted(self):
        validate_json_safe(
            {
                "k1": [1, 2, {"k2": [None, True, "x"]}],
                "k3": {"k4": {"k5": [1.5, -2.5]}},
            }
        )

    def test_list_with_bad_element_rejected_with_path(self):
        with pytest.raises(ValueTypeError, match=r"\[2\]"):
            validate_json_safe([1, 2, b"oops"])

    def test_dict_non_string_key_rejected(self):
        with pytest.raises(ValueTypeError, match="dict key"):
            validate_json_safe({1: "v"})

    def test_dict_nested_bad_value_rejected_with_path(self):
        with pytest.raises(ValueTypeError, match=r"\.outer\.inner"):
            validate_json_safe({"outer": {"inner": object()}})


class TestRejectedTypes:
    def test_bytes_rejected(self):
        with pytest.raises(ValueTypeError, match="bytes"):
            validate_json_safe(b"\x00\x01")

    def test_datetime_rejected(self):
        with pytest.raises(ValueTypeError, match="datetime"):
            validate_json_safe(datetime(2025, 1, 1))

    def test_uuid_rejected(self):
        with pytest.raises(ValueTypeError, match="UUID"):
            validate_json_safe(UUID("11111111-1111-4111-8111-111111111111"))

    def test_decimal_rejected(self):
        with pytest.raises(ValueTypeError, match="Decimal"):
            validate_json_safe(Decimal("1.5"))

    def test_tuple_rejected_with_hint(self):
        with pytest.raises(ValueTypeError, match="tuple"):
            validate_json_safe((1, 2, 3))

    def test_custom_class_rejected(self):
        class Foo:
            pass

        with pytest.raises(ValueTypeError, match="Foo"):
            validate_json_safe(Foo())

    def test_set_rejected(self):
        with pytest.raises(ValueTypeError, match="set"):
            validate_json_safe({1, 2, 3})


class TestBoundary:
    def test_root_path_in_error(self):
        with pytest.raises(ValueTypeError, match="<root>"):
            validate_json_safe(b"bad")

    def test_finite_float_edges(self):
        validate_json_safe(math.pi)
        validate_json_safe(-1e300)
        validate_json_safe(1e-300)
