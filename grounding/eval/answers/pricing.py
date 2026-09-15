"""Per-model token pricing and cost arithmetic (Epic 25, metric 5 and D6).

Prices are Anthropic first-party API list prices in US dollars per million
tokens. Source: the ``claude-api`` skill's "Current Models" table (cached
2026-06-24), which mirrors https://platform.claude.com/docs/en/pricing.md.
Cache writes are priced at 1.25x the input rate and cache reads at 0.1x,
except where a model documents its own cache-read rate. Update this table,
and ``PRICING_SOURCE``, whenever the published prices change; every run
records the table it used so old runs stay interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

PRICING_SOURCE = (
    "claude-api skill, Current Models table (cached 2026-06-24); "
    "https://platform.claude.com/docs/en/pricing.md"
)

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.10


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float | None = None  # None -> 0.1x input

    def as_dict(self) -> Dict[str, float]:
        return {
            "input_per_mtok": self.input_per_mtok,
            "output_per_mtok": self.output_per_mtok,
            "cache_write_per_mtok": self.input_per_mtok * CACHE_WRITE_MULTIPLIER,
            "cache_read_per_mtok": self.cache_read(),
        }

    def cache_read(self) -> float:
        if self.cache_read_per_mtok is not None:
            return self.cache_read_per_mtok
        return self.input_per_mtok * CACHE_READ_MULTIPLIER


PRICES: Dict[str, ModelPrice] = {
    "claude-fable-5-1": ModelPrice(10.0, 50.0, cache_read_per_mtok=0.25),
    "claude-fable-5": ModelPrice(10.0, 50.0),
    "claude-opus-5": ModelPrice(5.0, 25.0),
    "claude-opus-4-8": ModelPrice(5.0, 25.0),
    "claude-opus-4-7": ModelPrice(5.0, 25.0),
    "claude-opus-4-6": ModelPrice(5.0, 25.0),
    "claude-sonnet-5": ModelPrice(2.0, 10.0),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0),
}


def price_for(model: str) -> ModelPrice | None:
    """Price for ``model``, accepting a dated snapshot of a listed alias."""
    if model in PRICES:
        return PRICES[model]
    for alias, price in PRICES.items():
        if model.startswith(alias + "-") and model[len(alias) + 1 :].isdigit():
            return price
    return None


def usage_cost(model: str, usage: Mapping[str, Any]) -> float | None:
    """Dollar cost of one ``usage`` block, or None when the model is unpriced."""
    price = price_for(model)
    if price is None:
        return None
    return (
        _tok(usage, "input_tokens") * price.input_per_mtok
        + _tok(usage, "output_tokens") * price.output_per_mtok
        + _tok(usage, "cache_creation_input_tokens")
        * price.input_per_mtok
        * CACHE_WRITE_MULTIPLIER
        + _tok(usage, "cache_read_input_tokens") * price.cache_read()
    ) / 1_000_000


def tokens_cost(model: str, input_tokens: float, output_tokens: float) -> float | None:
    """Cost of uncached input and output token counts (used by estimates)."""
    price = price_for(model)
    if price is None:
        return None
    return (
        input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok
    ) / 1_000_000


def pricing_snapshot(models: list[str]) -> Dict[str, Any]:
    """Record the prices used for ``models`` in a run manifest."""
    return {
        "source": PRICING_SOURCE,
        "models": {
            m: (price_for(m).as_dict() if price_for(m) else None) for m in sorted(set(models))
        },
    }


def _tok(usage: Mapping[str, Any], key: str) -> float:
    value = usage.get(key) or 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
