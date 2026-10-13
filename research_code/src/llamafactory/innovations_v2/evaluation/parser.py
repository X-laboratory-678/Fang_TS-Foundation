import json
import math
from dataclasses import dataclass
from enum import Enum

try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass


class ParseCode(StrEnum):
    OK = "OK"
    INVALID_JSON = "INVALID_JSON"
    WRONG_LENGTH = "WRONG_LENGTH"
    NON_NUMERIC = "NON_NUMERIC"
    NAN_OR_INF = "NAN_OR_INF"
    EXTRA_TEXT = "EXTRA_TEXT"


@dataclass(frozen=True)
class ParseResult:
    code: ParseCode
    values: list[float] | None = None
    message: str = ""


def parse_forecast(text: str, horizon: int) -> ParseResult:
    stripped = text.strip()
    if not stripped.startswith("[") or not stripped.endswith("]"):
        return ParseResult(ParseCode.EXTRA_TEXT, message="Output must contain only a JSON list.")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return ParseResult(ParseCode.INVALID_JSON, message=str(exc))
    if not isinstance(value, list):
        return ParseResult(ParseCode.INVALID_JSON, message="Top-level JSON value is not a list.")
    if len(value) != horizon:
        return ParseResult(ParseCode.WRONG_LENGTH, message=f"Expected {horizon} values, got {len(value)}.")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return ParseResult(ParseCode.NON_NUMERIC, message="Every forecast element must be numeric.")
    values = [float(item) for item in value]
    if not all(math.isfinite(item) for item in values):
        return ParseResult(ParseCode.NAN_OR_INF, message="Forecast contains NaN or Inf.")
    return ParseResult(ParseCode.OK, values=values)
