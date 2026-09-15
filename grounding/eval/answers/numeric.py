"""Numeric auto-scoring for the answer benchmark (Story 25.3, metric 1).

Numeric items are scored against ``answer.numeric`` before any judge call.
The check is deliberately conservative: it only decides when the answer
leaves no doubt, and otherwise returns ``None`` so the rubric judge grades the
item.

Rules, applied to the answer text with citations removed:

1. Collect the quantities stated with the gold unit (unit strings compared
   case-insensitively after light normalization, no unit conversion). For a
   unitless gold value, every number counts.
2. No such quantity, or any of them given as a range: defer to the judge.
3. Every matching value within tolerance: score 1, unless the item also has
   ``must_include`` facts, which only the judge can check (defer).
4. Values both inside and outside tolerance: defer (a hedge or a worked
   calculation the judge should read).
5. Only out-of-tolerance values: score 0 when the gold has a unit. Without a
   unit, stray numbers (table or clause numbers) are too easy to catch, so
   defer instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from grounding.eval.fixtures import NumericAnswer

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
_NUM = r"[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+−]?\.\d+"
_SCI = (
    r"(?:\s*[eE][-+]?\d+"
    r"|\s*[x×]\s*10\s*(?:\^|\*\*)\s*[-+−]?\d+"
    r"|\s*[x×]\s*10[⁰¹²³⁴-⁹⁻]+)?"
)
_FRACTION = r"(?:\d+\s+)?\d+/\d+(?!\d)"
_UNIT = (
    r"(?:(?P<space>\s?)(?P<unit>%|°\s?[CFK]\b|"
    r"[A-Za-zµμΩ][A-Za-z0-9µμΩ²³·*/^.\-]*))?"
)
_QUANTITY_RE = re.compile(rf"(?<![\w.])(?P<num>{_FRACTION}|(?:{_NUM}){_SCI}){_UNIT}")
_RANGE_JOIN_RE = re.compile(r"^\s*(?:-|\u2013|\u2014|to|and)\s*$", re.I)
# Lowercase unit symbols accepted after a space ("5 mm"). Anything else after
# a space must look like a unit (a capital, digit or symbol, as in "200 GPa"),
# so ordinary words ("190 to 210", "2.5 per") are never read as units.
_LOWER_UNITS = {
    "mm", "cm", "m", "km", "um", "nm", "in", "ft", "yd", "mi", "mil", "mils",
    "lb", "lbs", "lbf", "kip", "kips", "kg", "g", "mg", "t", "s", "ms", "min",
    "h", "hr", "hrs", "psi", "ksi", "psf", "rpm", "hz", "khz", "deg", "rad",
    "mph", "l", "ml", "gal", "cfm", "gpm", "hp", "kw", "w", "j", "kj",
}

_UNIT_ALIASES = {
    "percent": "%", "pct": "%", "inch": "in", "inches": "in", "in.": "in",
    "degc": "°c", "degf": "°f", "celsius": "°c", "fahrenheit": "°f",
    "mpa": "mpa", "n/mm2": "mpa", "n/mm^2": "mpa",
}


@dataclass
class Quantity:
    value: float
    unit: str | None
    raw: str
    is_range: bool = False
    upper: float | None = None


def normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    u = unit.strip().rstrip(".,;:").replace(" ", "")
    u = u.replace("µ", "u").replace("μ", "u").replace("²", "2").replace("³", "3")
    u = u.replace("·", "*").replace("^", "").lower()
    return _UNIT_ALIASES.get(u, u) or None


def _looks_like_unit(unit: str) -> bool:
    bare = unit.rstrip(".,;:")
    if bare.lower() in _LOWER_UNITS or bare.lower() in _UNIT_ALIASES:
        return True
    return bool(re.search(r"[A-Z0-9%°µμΩ²³/·*^]", bare))


def _to_float(raw: str) -> float | None:
    text = raw.strip().replace("−", "-").replace(",", "")
    try:
        if "/" in text:
            parts = text.split()
            whole = float(parts[0]) if len(parts) == 2 else 0.0
            num, den = parts[-1].split("/")
            return whole + float(num) / float(den)
        match = re.match(r"^(.*?)\s*[x×]\s*10\s*(?:\^|\*\*)?\s*(.+)$", text)
        if match:
            exponent = match.group(2).translate(_SUPERSCRIPTS)
            return float(match.group(1)) * 10 ** int(exponent)
        return float(text)
    except (ValueError, ZeroDivisionError):
        return None


def extract_quantities(text: str) -> List[Quantity]:
    """Numbers with the unit that immediately follows them, ranges merged."""
    found = []
    for match in _QUANTITY_RE.finditer(text):
        value = _to_float(match.group("num"))
        if value is None:
            continue
        unit = match.group("unit")
        end = match.end()
        if unit and match.group("space") and not _looks_like_unit(unit):
            unit, end = None, match.end("num")
        found.append((match.start(), match.end("num"), end, value, unit, text[match.start():end]))

    quantities: List[Quantity] = []
    i = 0
    while i < len(found):
        start, num_end, end, value, unit, raw = found[i]
        if i + 1 < len(found) and unit is None:
            n_start, _, n_end, n_value, n_unit, n_raw = found[i + 1]
            if _RANGE_JOIN_RE.match(text[num_end:n_start]):
                quantities.append(
                    Quantity(min(value, n_value), normalize_unit(n_unit),
                             text[start:n_end], is_range=True, upper=max(value, n_value))
                )
                i += 2
                continue
        quantities.append(Quantity(value, normalize_unit(unit), raw.strip().rstrip(".,;:")))
        i += 1
    return quantities


def numeric_check(text: str, numeric: NumericAnswer, *, has_must_include: bool) -> Dict[str, Any]:
    """Apply the module rules; ``verdict`` is 1.0, 0.0, or None (use the judge)."""
    gold_unit = normalize_unit(numeric.unit)
    quantities = extract_quantities(text)
    if gold_unit is None:
        matching = quantities
    else:
        matching = [q for q in quantities if q.unit == gold_unit]

    def _row(q: Quantity) -> Dict[str, Any]:
        return {"raw": q.raw, "value": q.value, "unit": q.unit, "range": q.is_range}

    inside = [q for q in matching if not q.is_range and numeric.accepts(q.value)]
    outside = [q for q in matching if not q.is_range and not numeric.accepts(q.value)]
    result: Dict[str, Any] = {
        "gold_unit": gold_unit,
        "matching": [_row(q) for q in matching],
        "in_tolerance": [q.raw for q in inside],
        "out_of_tolerance": [q.raw for q in outside],
        "verdict": None,
    }
    if not matching:
        result["reason"] = "no value stated with the gold unit"
    elif any(q.is_range for q in matching):
        result["reason"] = "answer states a range"
    elif inside and outside:
        result["reason"] = "answer states values both inside and outside tolerance"
    elif inside:
        if has_must_include:
            result["reason"] = "value within tolerance; must_include facts need the judge"
        else:
            result["verdict"] = 1.0
            result["reason"] = f"{inside[0].raw} is within tolerance"
    elif gold_unit is None:
        result["reason"] = "unitless values out of tolerance; judge decides"
    else:
        result["verdict"] = 0.0
        result["reason"] = f"{outside[0].raw} is outside tolerance"
    return result
