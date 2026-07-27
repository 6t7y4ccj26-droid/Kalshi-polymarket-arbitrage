"""Conservative Kalshi/Polymarket contract matching and arbitrage analysis.

This module is deliberately detection-only.  A qualifying opportunity buys
complementary outcomes (YES on one venue and NO on the other), walks both
order books, and assumes the two legs must be filled atomically.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Callable, Optional

from kalshi_client.models import KalshiMarket
from polymarket_client.models import Market, OrderBook, OrderBookSide, TokenType

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ContractMatch:
    """Auditable result of comparing two contracts."""

    accepted: bool
    confidence: float
    similarity: float
    category: str
    reasons: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    outcome_alignment: str = "same"


@dataclass(frozen=True)
class MatchDiagnostic:
    """A rejected candidate with enough evidence for safe human review."""

    polymarket_id: str
    polymarket_question: str
    kalshi_ticker: str
    kalshi_title: str
    category: str
    similarity: float
    confidence: float
    polymarket_dates: tuple[str, ...]
    kalshi_dates: tuple[str, ...]
    positive_evidence: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    manual_review_recommended: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MarketPair:
    """A strict, auditable pair of equivalent binary contracts."""

    polymarket_id: str
    kalshi_ticker: str
    polymarket_question: str
    kalshi_title: str
    similarity_score: float
    category: str = ""
    confidence: float = 0.0
    outcome_alignment: str = "same"
    match_reasons: list[str] = field(default_factory=list)
    ambiguity_gap: float = 1.0
    matched_at: datetime = field(default_factory=_utcnow)

    @property
    def pair_id(self) -> str:
        return f"poly:{self.polymarket_id}|kalshi:{self.kalshi_ticker}"


@dataclass(frozen=True)
class ExecutionLeg:
    platform: str
    market_id: str
    outcome: str
    action: str
    contracts: float
    average_price: float
    worst_price: float
    notional: float
    estimated_fee: float


@dataclass
class CrossPlatformOpportunity:
    """An alert-ready paper opportunity; never an execution instruction."""

    opportunity_id: str
    market_pair: MarketPair
    buy_platform: str
    sell_platform: str
    token: str
    buy_price: float
    sell_price: float
    gross_edge: float
    net_edge: float
    edge_pct: float
    suggested_size: float = 0.0
    max_size: float = 0.0
    buy_liquidity: float = 0.0
    sell_liquidity: float = 0.0
    gross_profit: float = 0.0
    estimated_fees: float = 0.0
    estimated_slippage: float = 0.0
    estimated_gas: float = 0.0
    total_cost: float = 0.0
    guaranteed_payout: float = 0.0
    confidence: float = 0.0
    rank_score: float = 0.0
    legs: tuple[ExecutionLeg, ExecutionLeg] | tuple = ()
    execution_policy: str = "FOK_OR_ABORT_BOTH"
    dry_run: bool = True
    detected_at: datetime = field(default_factory=_utcnow)

    def __str__(self) -> str:
        return (
            f"PaperArb {self.market_pair.pair_id}: {self.token} | "
            f"net ${self.net_edge:.2f} ({self.edge_pct:.2%}), "
            f"confidence {self.confidence:.0%}"
        )

    def to_alert_dict(self) -> dict:
        """Stable, JSON-ready output for downstream alert adapters."""
        return {
            "schema_version": "1.0",
            "type": "cross_platform_arbitrage",
            "mode": "paper",
            "auto_execution": False,
            "opportunity_id": self.opportunity_id,
            "detected_at": self.detected_at.isoformat(),
            "pair": {
                "id": self.market_pair.pair_id,
                "polymarket_question": self.market_pair.polymarket_question,
                "kalshi_title": self.market_pair.kalshi_title,
                "category": self.market_pair.category,
                "confidence": round(self.confidence, 6),
                "outcome_alignment": self.market_pair.outcome_alignment,
                "match_reasons": list(self.market_pair.match_reasons),
            },
            "economics": {
                "contracts": round(self.suggested_size, 6),
                "total_cost": round(self.total_cost, 6),
                "guaranteed_payout": round(self.guaranteed_payout, 6),
                "gross_profit": round(self.gross_profit, 6),
                "estimated_fees": round(self.estimated_fees, 6),
                "estimated_slippage": round(self.estimated_slippage, 6),
                "estimated_gas": round(self.estimated_gas, 6),
                "conservative_net_profit": round(self.net_edge, 6),
                "conservative_net_edge_pct": round(self.edge_pct, 6),
                "rank_score": round(self.rank_score, 6),
            },
            "legs": [asdict(leg) for leg in self.legs],
            "execution_guard": {
                "policy": self.execution_policy,
                "partial_fills_accepted": False,
                "instruction": "Fill both complete legs or abort; output is paper-only.",
            },
        }


class MarketMatcher:
    """Reject-first matcher for binary contracts with auditable evidence."""

    NOISE_WORDS = {
        "will",
        "the",
        "a",
        "an",
        "be",
        "to",
        "in",
        "on",
        "by",
        "at",
        "what",
        "who",
        "which",
        "when",
        "is",
        "are",
        "market",
        "prediction",
    }
    CATEGORY_TERMS = {
        "sports": {"mlb", "nfl", "nba", "nhl", "game", "match", "series", "playoffs"},
        "politics": {"election", "president", "senate", "governor", "nominee", "vote"},
        "crypto": {"bitcoin", "btc", "ethereum", "eth", "crypto", "solana"},
        "finance": {"fed", "inflation", "gdp", "recession", "interest", "rate"},
        "weather": {"temperature", "rain", "snow", "hurricane", "weather"},
        "entertainment": {"oscar", "grammy", "emmy", "movie", "album"},
        "tech": {"openai", "gpt", "apple", "google", "microsoft", "nvidia"},
    }
    MONTHS = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    RESOLUTION_TERMS = {
        "official": {"official", "officially", "certified", "certification"},
        "announcement": {"announce", "announced", "announcement"},
        "settlement": {"settle", "settled", "resolution", "resolve"},
        "source": {"ap", "reuters", "noaa", "nws", "mlb", "nba", "nfl", "nhl"},
    }
    THRESHOLD_PATTERN = re.compile(
        r"(?:above|below|over|under|at least|more than|less than|reach(?:es)?)\s*"
        r"\$?\d+(?:\.\d+)?%?",
        re.IGNORECASE,
    )

    def __init__(
        self,
        min_similarity: float = 0.78,
        min_confidence: float = 0.80,
        ambiguity_margin: float = 0.08,
        require_date_evidence: bool = True,
    ):
        self.min_similarity = min_similarity
        self.min_confidence = min_confidence
        self.ambiguity_margin = ambiguity_margin
        self.require_date_evidence = require_date_evidence
        self._matched_pairs: dict[str, MarketPair] = {}
        self._rejection_diagnostics: list[MatchDiagnostic] = []

    def normalize_text(self, text: str) -> str:
        words = re.sub(r"[^\w%$.\s-]", " ", text.lower()).split()
        return " ".join(word for word in words if word not in self.NOISE_WORDS)

    def _categorize_market(self, text: str) -> str:
        words = set(re.findall(r"[a-z]+", text.lower()))
        scores = {
            category: len(words & terms)
            for category, terms in self.CATEGORY_TERMS.items()
        }
        category, score = max(scores.items(), key=lambda item: item[1])
        return category if score else "other"

    def extract_date(
        self, text: str, fallback: Optional[datetime] = None
    ) -> Optional[str]:
        numeric = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
        if numeric:
            return f"{numeric.group(1)}-{int(numeric.group(2)):02d}-{int(numeric.group(3)):02d}"
        numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", text)
        if numeric:
            year = int(numeric.group(3))
            year = year + 2000 if year < 100 else year
            return f"{year:04d}-{int(numeric.group(1)):02d}-{int(numeric.group(2)):02d}"
        for name, month in self.MONTHS.items():
            match = re.search(
                rf"\b{name}\.?\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b",
                text,
                re.IGNORECASE,
            )
            if match:
                year = (
                    int(match.group(2))
                    if match.group(2)
                    else (fallback.year if fallback else _utcnow().year)
                )
                return f"{year:04d}-{month:02d}-{int(match.group(1)):02d}"
        return fallback.date().isoformat() if fallback else None

    def _outcome_signature(self, text: str) -> tuple[str, ...]:
        lowered = text.lower()
        signatures = []
        for phrase in ("win", "wins", "above", "below", "over", "under", "yes", "no"):
            if re.search(rf"\b{phrase}\b", lowered):
                signatures.append(phrase)
        return tuple(signatures)

    def _settlement_signature(self, text: str) -> set[str]:
        words = set(re.findall(r"[a-z]+", text.lower()))
        return {
            group for group, terms in self.RESOLUTION_TERMS.items() if words & terms
        }

    def _thresholds(self, text: str) -> set[str]:
        return {
            re.sub(r"\s+", " ", match.group(0).lower())
            for match in self.THRESHOLD_PATTERN.finditer(text)
        }

    def _candidate_key(
        self,
        text: str,
        metadata_date: Optional[datetime],
    ) -> Optional[tuple[str, str]]:
        """Build a conservative category/date key before fuzzy comparison."""
        category = self._categorize_market(text)
        dates = {
            value
            for value in (
                self.extract_date(text),
                metadata_date.date().isoformat() if metadata_date else None,
            )
            if value
        }
        if category == "other" or len(dates) != 1:
            return None
        return category, next(iter(dates))

    def _market_dates(
        self, text: str, metadata_date: Optional[datetime]
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    value
                    for value in (
                        self.extract_date(text),
                        metadata_date.date().isoformat() if metadata_date else None,
                    )
                    if value
                }
            )
        )

    def _diagnostic(
        self,
        poly: Market,
        kalshi: KalshiMarket,
        result: ContractMatch,
    ) -> MatchDiagnostic:
        poly_text = " ".join(filter(None, (poly.question, poly.description)))
        kalshi_text = " ".join(
            filter(None, (kalshi.title, kalshi.subtitle, kalshi.rules_primary))
        )
        hard_blockers = {
            "category_mismatch_or_unknown",
            "date_mismatch",
            "internally_inconsistent_date",
            "threshold_or_direction_mismatch",
            "outcome_direction_mismatch",
            "settlement_rule_mismatch",
        }
        manual_review = result.similarity >= 0.55 and not hard_blockers.intersection(
            result.rejection_reasons
        )
        return MatchDiagnostic(
            polymarket_id=poly.market_id,
            polymarket_question=poly.question,
            kalshi_ticker=kalshi.ticker,
            kalshi_title=kalshi.title,
            category=result.category,
            similarity=result.similarity,
            confidence=result.confidence,
            polymarket_dates=self._market_dates(poly_text, poly.end_date),
            kalshi_dates=self._market_dates(
                kalshi_text, kalshi.expiration_time or kalshi.close_time
            ),
            positive_evidence=tuple(result.reasons),
            rejection_reasons=tuple(result.rejection_reasons),
            manual_review_recommended=manual_review,
        )

    def calculate_similarity(
        self, polymarket_question: str, kalshi_title: str
    ) -> float:
        first = self.normalize_text(polymarket_question)
        second = self.normalize_text(kalshi_title)
        sequence = SequenceMatcher(None, first, second).ratio()
        first_tokens, second_tokens = set(first.split()), set(second.split())
        union = first_tokens | second_tokens
        jaccard = len(first_tokens & second_tokens) / len(union) if union else 0.0
        return (0.55 * sequence) + (0.45 * jaccard)

    def evaluate_contracts(
        self,
        poly: Market,
        kalshi: KalshiMarket,
    ) -> ContractMatch:
        poly_text = " ".join(filter(None, (poly.question, poly.description)))
        kalshi_text = " ".join(
            filter(
                None,
                (
                    kalshi.title,
                    getattr(kalshi, "subtitle", ""),
                    getattr(kalshi, "rules_primary", ""),
                ),
            )
        )
        similarity = self.calculate_similarity(poly.question, kalshi.title)
        poly_category = self._categorize_market(poly_text)
        kalshi_category = self._categorize_market(kalshi_text)
        reasons: list[str] = []
        rejected: list[str] = []

        if poly_category != kalshi_category or poly_category == "other":
            rejected.append("category_mismatch_or_unknown")
        else:
            reasons.append("category_match")

        if similarity < self.min_similarity:
            rejected.append("insufficient_title_similarity")
        else:
            reasons.append("title_similarity")

        poly_dates = {
            value
            for value in (
                self.extract_date(poly_text),
                poly.end_date.date().isoformat() if poly.end_date else None,
            )
            if value
        }
        kalshi_timestamp = getattr(kalshi, "expiration_time", None) or getattr(
            kalshi, "close_time", None
        )
        kalshi_dates = {
            value
            for value in (
                self.extract_date(kalshi_text),
                kalshi_timestamp.date().isoformat() if kalshi_timestamp else None,
            )
            if value
        }
        if len(poly_dates) > 1 or len(kalshi_dates) > 1:
            rejected.append("internally_inconsistent_date")
        if poly_dates and kalshi_dates:
            if poly_dates != kalshi_dates:
                rejected.append("date_mismatch")
            else:
                reasons.append("date_match")
        elif self.require_date_evidence:
            rejected.append("missing_date_evidence")

        poly_thresholds, kalshi_thresholds = self._thresholds(
            poly_text
        ), self._thresholds(kalshi_text)
        if poly_thresholds or kalshi_thresholds:
            if poly_thresholds != kalshi_thresholds:
                rejected.append("threshold_or_direction_mismatch")
            else:
                reasons.append("threshold_match")

        poly_outcome, kalshi_outcome = self._outcome_signature(
            poly.question
        ), self._outcome_signature(kalshi.title)
        contradictory = (
            ("above" in poly_outcome and "below" in kalshi_outcome)
            or ("below" in poly_outcome and "above" in kalshi_outcome)
            or ("over" in poly_outcome and "under" in kalshi_outcome)
            or ("under" in poly_outcome and "over" in kalshi_outcome)
        )
        if contradictory:
            rejected.append("outcome_direction_mismatch")
        else:
            reasons.append("outcome_alignment")

        poly_resolution = self._settlement_signature(poly.description)
        kalshi_resolution = self._settlement_signature(
            " ".join(
                filter(
                    None,
                    (
                        getattr(kalshi, "subtitle", ""),
                        getattr(kalshi, "rules_primary", ""),
                    ),
                )
            )
        )
        if poly_resolution or kalshi_resolution:
            if (
                not poly_resolution
                or not kalshi_resolution
                or not (poly_resolution & kalshi_resolution)
            ):
                rejected.append("settlement_rule_mismatch")
            else:
                reasons.append("settlement_rule_overlap")
        else:
            rejected.append("missing_settlement_evidence")

        confidence = min(
            1.0,
            similarity * 0.55
            + (0.15 if "date_match" in reasons else 0)
            + (0.15 if "settlement_rule_overlap" in reasons else 0)
            + (0.10 if "outcome_alignment" in reasons else 0)
            + (0.05 if "threshold_match" in reasons else 0),
        )
        if confidence < self.min_confidence:
            rejected.append("confidence_below_threshold")
        return ContractMatch(
            accepted=not rejected,
            confidence=confidence,
            similarity=similarity,
            category=poly_category,
            reasons=reasons,
            rejection_reasons=sorted(set(rejected)),
        )

    async def find_matches(
        self,
        polymarket_markets: list[Market],
        kalshi_markets: list[KalshiMarket],
        on_progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> list[MarketPair]:
        """Find only unique, high-confidence matches; reject close runners-up."""
        candidates: list[tuple[Market, KalshiMarket, ContractMatch]] = []
        diagnostics: list[MatchDiagnostic] = []
        comparisons = 0
        kalshi_index: dict[tuple[str, str], list[KalshiMarket]] = {}
        kalshi_by_ticker: dict[str, KalshiMarket] = {}
        token_index: dict[str, dict[str, set[str]]] = {}
        for kalshi in (market for market in kalshi_markets if market.is_active):
            kalshi_text = " ".join(
                filter(
                    None,
                    (kalshi.title, kalshi.subtitle, kalshi.rules_primary),
                )
            )
            category = self._categorize_market(kalshi_text)
            kalshi_by_ticker[kalshi.ticker] = kalshi
            normalized_tokens = {
                token
                for token in self.normalize_text(kalshi.title).split()
                if len(token) >= 3
            }
            for index_category in {category, "*"}:
                category_tokens = token_index.setdefault(index_category, {})
                for token in normalized_tokens:
                    if token:
                        category_tokens.setdefault(token, set()).add(kalshi.ticker)
            key = self._candidate_key(
                kalshi_text,
                kalshi.expiration_time or kalshi.close_time,
            )
            if key:
                kalshi_index.setdefault(key, []).append(kalshi)

        poly_candidates: list[tuple[Market, list[KalshiMarket], list[KalshiMarket]]] = (
            []
        )
        for poly in (
            market
            for market in polymarket_markets
            if market.active and not market.closed
        ):
            poly_text = " ".join(filter(None, (poly.question, poly.description)))
            key = self._candidate_key(poly_text, poly.end_date)
            exact_candidates = kalshi_index.get(key, []) if key else []

            category = self._categorize_market(poly_text)
            candidate_counts: dict[str, int] = {}
            poly_tokens = set(self.normalize_text(poly.question).split())
            for token in poly_tokens:
                for ticker in token_index.get(category, {}).get(token, ()):
                    candidate_counts[ticker] = candidate_counts.get(ticker, 0) + 1
            # Unknown category metadata is common in Kalshi's bulk feed. Use
            # the global title index only for diagnostics, never acceptance.
            if not candidate_counts:
                for token in poly_tokens:
                    for ticker in token_index.get("*", {}).get(token, ()):
                        candidate_counts[ticker] = candidate_counts.get(ticker, 0) + 1
            exact_tickers = {market.ticker for market in exact_candidates}
            nearest_candidates = [
                kalshi_by_ticker[ticker]
                for ticker, shared_tokens in sorted(
                    candidate_counts.items(),
                    key=lambda item: (item[1], item[0]),
                    reverse=True,
                )
                if shared_tokens >= 2 and ticker not in exact_tickers
            ][:3]
            if exact_candidates or nearest_candidates:
                poly_candidates.append((poly, exact_candidates, nearest_candidates))

        total_comparisons = sum(
            len(exact_candidates) for _, exact_candidates, _ in poly_candidates
        )
        for poly, kalshi_candidates, nearest_candidates in poly_candidates:
            ranked: list[tuple[KalshiMarket, ContractMatch]] = []
            for kalshi in kalshi_candidates:
                result = self.evaluate_contracts(poly, kalshi)
                comparisons += 1
                if result.accepted:
                    ranked.append((kalshi, result))
                else:
                    diagnostics.append(self._diagnostic(poly, kalshi, result))
                if on_progress and comparisons % 500 == 0:
                    on_progress(comparisons, total_comparisons, len(candidates))
            for kalshi in nearest_candidates:
                result = self.evaluate_contracts(poly, kalshi)
                if not result.accepted:
                    diagnostics.append(self._diagnostic(poly, kalshi, result))
            ranked.sort(key=lambda item: item[1].confidence, reverse=True)
            if not ranked:
                continue
            gap = ranked[0][1].confidence - (
                ranked[1][1].confidence if len(ranked) > 1 else 0.0
            )
            if len(ranked) > 1 and gap < self.ambiguity_margin:
                logger.warning(
                    "Rejected ambiguous match for %s (gap %.3f)", poly.market_id, gap
                )
                continue
            kalshi, result = ranked[0]
            candidates.append((poly, kalshi, result))

        # Enforce one-to-one mapping. A Kalshi contract cannot back multiple pairs.
        by_kalshi: dict[str, list[tuple[Market, KalshiMarket, ContractMatch]]] = {}
        for candidate in candidates:
            by_kalshi.setdefault(candidate[1].ticker, []).append(candidate)

        matches: list[MarketPair] = []
        for ticker, group in by_kalshi.items():
            group.sort(key=lambda item: item[2].confidence, reverse=True)
            if (
                len(group) > 1
                and group[0][2].confidence - group[1][2].confidence
                < self.ambiguity_margin
            ):
                logger.warning("Rejected non-unique Kalshi mapping for %s", ticker)
                continue
            poly, kalshi, result = group[0]
            pair = MarketPair(
                polymarket_id=poly.market_id,
                kalshi_ticker=kalshi.ticker,
                polymarket_question=poly.question,
                kalshi_title=kalshi.title,
                similarity_score=result.similarity,
                category=result.category,
                confidence=result.confidence,
                match_reasons=result.reasons,
                ambiguity_gap=(
                    group[0][2].confidence - group[1][2].confidence
                    if len(group) > 1
                    else 1.0
                ),
            )
            matches.append(pair)
            self._matched_pairs[pair.pair_id] = pair
        if on_progress:
            on_progress(comparisons, total_comparisons, len(matches))
        self._rejection_diagnostics = sorted(
            diagnostics,
            key=lambda item: (
                item.manual_review_recommended,
                item.confidence,
                item.similarity,
            ),
            reverse=True,
        )[:50]
        return matches

    def get_cached_pairs(self) -> list[MarketPair]:
        return list(self._matched_pairs.values())

    def get_rejection_diagnostics(self) -> list[dict]:
        return [item.to_dict() for item in self._rejection_diagnostics]


@dataclass
class CrossPlatformArbConfig:
    min_net_edge: float = 0.01
    min_confidence: float = 0.80
    polymarket_taker_fee: float = 0.015
    kalshi_taker_fee: float = 0.01
    slippage_buffer_bps: float = 20.0
    gas_cost_total: float = 0.04
    min_contracts: float = 5.0
    min_net_profit: float = 0.25
    max_contracts: float = 100.0
    max_book_age_seconds: float = 5.0


class CrossPlatformArbEngine:
    """Depth-aware paper scanner for complementary cross-venue contracts."""

    def __init__(
        self,
        min_edge: float = 0.02,
        polymarket_taker_fee: float = 0.015,
        kalshi_taker_fee: float = 0.01,
        gas_cost: float = 0.02,
        config: Optional[CrossPlatformArbConfig] = None,
    ):
        self.config = config or CrossPlatformArbConfig(
            min_net_edge=min_edge,
            polymarket_taker_fee=polymarket_taker_fee,
            kalshi_taker_fee=kalshi_taker_fee,
            gas_cost_total=gas_cost * 2,
        )
        self.matcher = MarketMatcher(min_confidence=self.config.min_confidence)
        self._opportunities: list[CrossPlatformOpportunity] = []
        self._opportunity_count = 0
        self._rejections: dict[str, int] = {}

    def _reject(self, reason: str) -> None:
        self._rejections[reason] = self._rejections.get(reason, 0) + 1

    def _book_is_fresh(self, book: OrderBook, now: datetime) -> bool:
        timestamp = book.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return (
            max(0.0, (now - timestamp).total_seconds())
            <= self.config.max_book_age_seconds
        )

    @staticmethod
    def _sorted_asks(side: OrderBookSide) -> list:
        return sorted(
            (level for level in side.levels if 0 < level.price < 1 and level.size > 0),
            key=lambda level: level.price,
        )

    def _evaluate_direction(
        self,
        pair: MarketPair,
        first_platform: str,
        first_market_id: str,
        first_outcome: str,
        first_side: OrderBookSide,
        second_platform: str,
        second_market_id: str,
        second_outcome: str,
        second_side: OrderBookSide,
    ) -> Optional[CrossPlatformOpportunity]:
        first_levels = self._sorted_asks(first_side)
        second_levels = self._sorted_asks(second_side)
        if not first_levels or not second_levels:
            return None

        first_fee = (
            self.config.polymarket_taker_fee
            if first_platform == "polymarket"
            else self.config.kalshi_taker_fee
        )
        second_fee = (
            self.config.polymarket_taker_fee
            if second_platform == "polymarket"
            else self.config.kalshi_taker_fee
        )
        buffer = self.config.slippage_buffer_bps / 10_000
        i = j = 0
        remaining_first = first_levels[0].size
        remaining_second = second_levels[0].size
        contracts = first_cost = second_cost = fees = slippage = 0.0
        worst_first = worst_second = 0.0

        while (
            i < len(first_levels)
            and j < len(second_levels)
            and contracts < self.config.max_contracts
        ):
            first = first_levels[i]
            second = second_levels[j]
            marginal_fees = (first.price * first_fee) + (second.price * second_fee)
            marginal_slippage = (first.price + second.price) * buffer
            marginal_net = (
                1.0 - first.price - second.price - marginal_fees - marginal_slippage
            )
            if marginal_net < self.config.min_net_edge:
                break
            quantity = min(
                remaining_first,
                remaining_second,
                self.config.max_contracts - contracts,
            )
            if quantity <= 0:
                break
            contracts += quantity
            first_cost += first.price * quantity
            second_cost += second.price * quantity
            fees += marginal_fees * quantity
            slippage += marginal_slippage * quantity
            worst_first, worst_second = first.price, second.price
            remaining_first -= quantity
            remaining_second -= quantity
            if remaining_first <= 1e-12:
                i += 1
                if i < len(first_levels):
                    remaining_first = first_levels[i].size
            if remaining_second <= 1e-12:
                j += 1
                if j < len(second_levels):
                    remaining_second = second_levels[j].size

        if contracts < self.config.min_contracts:
            return None
        total_cost = first_cost + second_cost
        payout = contracts
        gross_profit = payout - total_cost
        net_profit = gross_profit - fees - slippage - self.config.gas_cost_total
        net_edge_pct = net_profit / total_cost if total_cost else 0.0
        if (
            net_profit < self.config.min_net_profit
            or net_edge_pct < self.config.min_net_edge
        ):
            return None

        self._opportunity_count += 1
        first_leg = ExecutionLeg(
            platform=first_platform,
            market_id=first_market_id,
            outcome=first_outcome,
            action="BUY",
            contracts=contracts,
            average_price=first_cost / contracts,
            worst_price=worst_first,
            notional=first_cost,
            estimated_fee=first_cost * first_fee,
        )
        second_leg = ExecutionLeg(
            platform=second_platform,
            market_id=second_market_id,
            outcome=second_outcome,
            action="BUY",
            contracts=contracts,
            average_price=second_cost / contracts,
            worst_price=worst_second,
            notional=second_cost,
            estimated_fee=second_cost * second_fee,
        )
        confidence = min(pair.confidence, 1.0)
        rank_score = (
            net_edge_pct * confidence * min(1.0, contracts / self.config.max_contracts)
        )
        return CrossPlatformOpportunity(
            opportunity_id=f"xplat_{self._opportunity_count}",
            market_pair=pair,
            buy_platform=first_platform,
            sell_platform=second_platform,
            token=f"{first_outcome}/{second_outcome}",
            buy_price=first_leg.average_price,
            sell_price=second_leg.average_price,
            gross_edge=gross_profit,
            net_edge=net_profit,
            edge_pct=net_edge_pct,
            suggested_size=contracts,
            max_size=contracts,
            buy_liquidity=sum(level.size for level in first_levels),
            sell_liquidity=sum(level.size for level in second_levels),
            gross_profit=gross_profit,
            estimated_fees=fees,
            estimated_slippage=slippage,
            estimated_gas=self.config.gas_cost_total,
            total_cost=total_cost,
            guaranteed_payout=payout,
            confidence=confidence,
            rank_score=rank_score,
            legs=(first_leg, second_leg),
        )

    def check_arbitrage(
        self,
        market_pair: MarketPair,
        polymarket_ob: OrderBook,
        kalshi_ob: OrderBook,
    ) -> Optional[CrossPlatformOpportunity]:
        if market_pair.confidence < self.config.min_confidence:
            self._reject("low_match_confidence")
            return None
        now = _utcnow()
        if not self._book_is_fresh(polymarket_ob, now) or not self._book_is_fresh(
            kalshi_ob, now
        ):
            self._reject("stale_order_book")
            return None

        directions = (
            (
                "polymarket",
                polymarket_ob.market_id,
                "YES",
                polymarket_ob.yes.asks,
                "kalshi",
                kalshi_ob.market_id,
                "NO",
                kalshi_ob.no.asks,
            ),
            (
                "kalshi",
                kalshi_ob.market_id,
                "YES",
                kalshi_ob.yes.asks,
                "polymarket",
                polymarket_ob.market_id,
                "NO",
                polymarket_ob.no.asks,
            ),
        )
        opportunities = [
            opportunity
            for args in directions
            if (opportunity := self._evaluate_direction(market_pair, *args)) is not None
        ]
        if not opportunities:
            self._reject("no_executable_depth")
            return None
        best = max(opportunities, key=lambda item: (item.rank_score, item.net_edge))
        self._opportunities.append(best)
        logger.info("PAPER opportunity: %s", best)
        return best

    def rank_opportunities(
        self, opportunities: Optional[list[CrossPlatformOpportunity]] = None
    ) -> list[CrossPlatformOpportunity]:
        return sorted(
            opportunities if opportunities is not None else self._opportunities,
            key=lambda item: (item.rank_score, item.net_edge),
            reverse=True,
        )

    def get_recent_opportunities(
        self, limit: int = 50
    ) -> list[CrossPlatformOpportunity]:
        return self.rank_opportunities(self._opportunities[-limit:])

    def get_stats(self) -> dict:
        return {
            "total_opportunities": len(self._opportunities),
            "matched_pairs": len(self.matcher.get_cached_pairs()),
            "avg_edge": (
                sum(item.edge_pct for item in self._opportunities)
                / len(self._opportunities)
                if self._opportunities
                else 0
            ),
            "rejections": dict(self._rejections),
            "mode": "paper",
            "auto_execution": False,
        }
