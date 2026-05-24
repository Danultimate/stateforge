from __future__ import annotations

import math
from typing import Any

from stateforge.exceptions import ValueTypeError


def validate_json_safe(value: Any, *, path: str = "<root>") -> None:
    """Recursively validate that ``value`` is JSON-safe.

    JSON-safe means: str, int, float (finite), bool, None, list of JSON-safe,
    or dict with string keys and JSON-safe values. ``bytes``, ``datetime``,
    ``UUID``, ``Decimal``, and any custom class are rejected.

    Raises:
        ValueTypeError: if the value (or any nested element) is not JSON-safe.
    """
    # bool is a subclass of int — check explicitly so the message is accurate.
    if value is None or isinstance(value, (bool, str)):
        return

    if isinstance(value, int):
        # int (and bool, already handled) are JSON-safe.
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueTypeError(
                f"non-finite float at {path}: {value!r} is not JSON-representable"
            )
        return

    if isinstance(value, list):
        for i, item in enumerate(value):
            validate_json_safe(item, path=f"{path}[{i}]")
        return

    if isinstance(value, tuple):
        raise ValueTypeError(
            f"tuple at {path} is not JSON-safe; convert to list before passing"
        )

    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueTypeError(
                    f"dict key at {path} is not a string: "
                    f"{type(k).__name__} {k!r}"
                )
            validate_json_safe(v, path=f"{path}.{k}")
        return

    raise ValueTypeError(
        f"value at {path} is not JSON-safe: type={type(value).__name__}"
    )
