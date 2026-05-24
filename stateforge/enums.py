from enum import Enum


class MemoryUnitType(str, Enum):
    MESSAGE = "message"
    KV = "kv"
    EMBEDDING = "embedding"
    TOOL_RESULT = "tool_result"
    SUMMARY = "summary"
