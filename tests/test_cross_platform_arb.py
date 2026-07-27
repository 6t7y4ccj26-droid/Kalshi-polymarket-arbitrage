from datetime import datetime, timedelta, timezone

import pytest

from core.cross_platform_arb import (
    CrossPlatformArbConfig,
    CrossPlatformArbEngine,
    MarketMatcher,
    MarketPair,
)
from kalshi_client.models import KalshiMarket
from polymarket_client.models import (
    Market,
    OrderBook,
    OrderBookSide,
    PriceLevel,
    TokenOrderBook,
    TokenType,
)


def market_pair(confidence: float = 0.94) -> MarketPair:
    return MarketPair(
        polymarket_id="poly-fed",
        kalshi_ticker="KXFED-26SEP16-H0",
        polymarket_question="Will the Fed hold rates on September 16, 2026?",
        kalshi_title="Fed holds rates on September 16, 2026",
        similarity_score=0.91,
        category="finance",
        confidence=confidence,
        match_reasons=["date_match", "settlement_rule_overlap", "outcome_alignment"],
    )


def book(
    market_id: str,
    *,
    yes_asks: list[tuple[float, float]],
    no_asks: list[tuple[float, float]],
    age_seconds: float = 0,
) -> OrderBook:
    return OrderBook(
        market_id=market_id,
        yes=TokenOrderBook(
            TokenType.YES,
            asks=OrderBookSide([PriceLevel(*level) for level in yes_asks]),
        ),
        no=TokenOrderBook(
            TokenType.NO,
            asks=OrderBookSide([PriceLevel(*level) for level in no_asks]),
        ),
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
    )


def engine(**overrides) -> CrossPlatformArbEngine:
    values = {
        "min_net_edge": 0.01,
        "min_confidence": 0.80,
        "polymarket_taker_fee": 0.01,
        "kalshi_taker_fee": 0.01,
        "slippage_buffer_bps": 10,
        "gas_cost_total": 0.02,
        "min_contracts": 5,
        "min_net_profit": 0.25,
        "max_contracts": 100,
        "max_book_age_seconds": 5,
    }
    values.update(overrides)
    return CrossPlatformArbEngine(config=CrossPlatformArbConfig(**values))


class TestStrictContractMatching:
    def poly(self, **overrides) -> Market:
        values = dict(
            market_id="poly-fed",
            condition_id="condition",
            question="Will the Fed hold rates on September 16, 2026?",
            description="Resolves using the official Federal Reserve announcement.",
            end_date=datetime(2026, 9, 16, tzinfo=timezone.utc),
        )
        values.update(overrides)
        return Market(**values)

    def kalshi(self, **overrides) -> KalshiMarket:
        values = dict(
            ticker="KXFED-26SEP16-H0",
            event_ticker="KXFED-26SEP16",
            series_ticker="KXFED",
            title="Will the Fed hold rates on September 16, 2026?",
            subtitle="Resolves using the official Federal Reserve announcement.",
            expiration_time=datetime(2026, 9, 16, tzinfo=timezone.utc),
        )
        values.update(overrides)
        return KalshiMarket(**values)

    def test_accepts_equivalent_contract_with_audit_reasons(self):
        result = MarketMatcher().evaluate_contracts(self.poly(), self.kalshi())

        assert result.accepted
        assert result.confidence >= 0.80
        assert {"date_match", "settlement_rule_overlap", "outcome_alignment"} <= set(
            result.reasons
        )

    @pytest.mark.parametrize(
        ("poly_changes", "kalshi_changes", "reason"),
        [
            (
                {},
                {"expiration_time": datetime(2026, 9, 17, tzinfo=timezone.utc)},
                "date_mismatch",
            ),
            (
                {"question": "Will inflation be above 3% on September 16, 2026?"},
                {
                    "title": "Will inflation be below 3% on September 16, 2026?",
                    "subtitle": "Resolves using the official CPI announcement.",
                },
                "outcome_direction_mismatch",
            ),
            (
                {
                    "description": "Resolves using the official Federal Reserve announcement."
                },
                {"subtitle": "Resolves using an unofficial survey."},
                "settlement_rule_mismatch",
            ),
        ],
    )
    def test_rejects_contract_mismatches(self, poly_changes, kalshi_changes, reason):
        result = MarketMatcher().evaluate_contracts(
            self.poly(**poly_changes),
            self.kalshi(**kalshi_changes),
        )

        assert not result.accepted
        assert reason in result.rejection_reasons

    @pytest.mark.asyncio
    async def test_prefilters_candidates_by_category_and_exact_date(self):
        matcher = MarketMatcher()
        matching = self.kalshi()
        irrelevant = [
            self.kalshi(
                ticker=f"KXFED-OTHER-{day}",
                title=f"Will the Fed hold rates on September {day}, 2026?",
                expiration_time=datetime(2026, 9, day, tzinfo=timezone.utc),
            )
            for day in range(1, 16)
        ]
        original_evaluate = matcher.evaluate_contracts
        evaluated = []

        def tracking_evaluate(poly, kalshi):
            evaluated.append(kalshi.ticker)
            return original_evaluate(poly, kalshi)

        matcher.evaluate_contracts = tracking_evaluate
        matches = await matcher.find_matches([self.poly()], irrelevant + [matching])

        assert len(matches) == 1
        assert matching.ticker in evaluated
        assert len(evaluated) <= 4

    @pytest.mark.asyncio
    async def test_rejection_diagnostics_explain_hard_blockers(self):
        matcher = MarketMatcher()
        await matcher.find_matches(
            [self.poly()],
            [self.kalshi(expiration_time=datetime(2026, 9, 17, tzinfo=timezone.utc))],
        )

        diagnostics = matcher.get_rejection_diagnostics()

        assert len(diagnostics) == 1
        assert "date_mismatch" in diagnostics[0]["rejection_reasons"]
        assert diagnostics[0]["polymarket_dates"] == ("2026-09-16",)
        assert set(diagnostics[0]["kalshi_dates"]) == {
            "2026-09-16",
            "2026-09-17",
        }
        assert diagnostics[0]["manual_review_recommended"] is False

    @pytest.mark.asyncio
    async def test_missing_evidence_can_be_flagged_for_manual_review(self):
        matcher = MarketMatcher()
        poly = self.poly(description="")
        kalshi = self.kalshi(subtitle="", rules_primary="")

        await matcher.find_matches([poly], [kalshi])
        diagnostic = matcher.get_rejection_diagnostics()[0]

        assert "missing_settlement_evidence" in diagnostic["rejection_reasons"]
        assert diagnostic["manual_review_recommended"] is True

    @pytest.mark.asyncio
    async def test_unknown_category_candidate_is_explained_but_never_accepted(self):
        matcher = MarketMatcher()
        poly = self.poly(
            question="Seattle Mariners vs Texas Rangers",
            description=(
                "MLB game scheduled July 27, 2026. "
                "Official final statistics determine resolution."
            ),
            end_date=datetime(2026, 7, 27, tzinfo=timezone.utc),
        )
        kalshi = self.kalshi(
            title="yes Seattle Mariners, no Texas Rangers",
            subtitle="",
            rules_primary="",
            expiration_time=datetime(2026, 7, 30, tzinfo=timezone.utc),
        )

        matches = await matcher.find_matches([poly], [kalshi])
        diagnostic = matcher.get_rejection_diagnostics()[0]

        assert matches == []
        assert "category_mismatch_or_unknown" in diagnostic["rejection_reasons"]
        assert diagnostic["manual_review_recommended"] is False


class TestDepthAwareProfitability:
    def test_walks_depth_and_stops_before_unprofitable_levels(self):
        poly = book(
            "poly-fed",
            yes_asks=[(0.40, 10), (0.47, 20)],
            no_asks=[(0.70, 100)],
        )
        kalshi = book(
            "kalshi:fed",
            yes_asks=[(0.80, 100)],
            no_asks=[(0.48, 10), (0.56, 20)],
        )

        opportunity = engine().check_arbitrage(market_pair(), poly, kalshi)

        assert opportunity is not None
        assert opportunity.suggested_size == pytest.approx(10)
        assert opportunity.total_cost == pytest.approx(8.8)
        assert opportunity.guaranteed_payout == pytest.approx(10)
        assert opportunity.net_edge < opportunity.gross_profit
        assert opportunity.legs[0].average_price == pytest.approx(0.40)
        assert opportunity.legs[1].average_price == pytest.approx(0.48)

    def test_rejects_top_of_book_that_cannot_meet_minimum_size(self):
        poly = book(
            "poly-fed",
            yes_asks=[(0.35, 2), (0.55, 100)],
            no_asks=[(0.90, 100)],
        )
        kalshi = book(
            "kalshi:fed",
            yes_asks=[(0.90, 100)],
            no_asks=[(0.45, 2), (0.55, 100)],
        )

        assert (
            engine(min_contracts=5).check_arbitrage(market_pair(), poly, kalshi) is None
        )

    def test_rejects_stale_books_and_low_confidence_pairs(self):
        poly = book(
            "poly-fed", yes_asks=[(0.40, 20)], no_asks=[(0.70, 20)], age_seconds=6
        )
        kalshi = book("kalshi:fed", yes_asks=[(0.80, 20)], no_asks=[(0.48, 20)])

        scanner = engine(max_book_age_seconds=5)
        assert scanner.check_arbitrage(market_pair(), poly, kalshi) is None
        assert scanner.get_stats()["rejections"]["stale_order_book"] == 1

        fresh_poly = book("poly-fed", yes_asks=[(0.40, 20)], no_asks=[(0.70, 20)])
        assert (
            scanner.check_arbitrage(market_pair(confidence=0.70), fresh_poly, kalshi)
            is None
        )
        assert scanner.get_stats()["rejections"]["low_match_confidence"] == 1

    def test_alert_output_is_structured_and_paper_only(self):
        poly = book("poly-fed", yes_asks=[(0.40, 20)], no_asks=[(0.70, 20)])
        kalshi = book("kalshi:fed", yes_asks=[(0.80, 20)], no_asks=[(0.48, 20)])

        opportunity = engine().check_arbitrage(market_pair(), poly, kalshi)
        alert = opportunity.to_alert_dict()

        assert alert["schema_version"] == "1.0"
        assert alert["mode"] == "paper"
        assert alert["auto_execution"] is False
        assert len(alert["legs"]) == 2
        assert {leg["outcome"] for leg in alert["legs"]} == {"YES", "NO"}
        assert alert["execution_guard"]["partial_fills_accepted"] is False
        assert alert["economics"]["conservative_net_profit"] > 0

    def test_ranking_uses_edge_confidence_and_executable_size(self):
        poly = book("poly-fed", yes_asks=[(0.40, 100)], no_asks=[(0.70, 100)])
        shallow_kalshi = book(
            "kalshi:shallow", yes_asks=[(0.80, 100)], no_asks=[(0.45, 5)]
        )
        deep_kalshi = book("kalshi:deep", yes_asks=[(0.80, 100)], no_asks=[(0.48, 100)])
        scanner = engine()
        shallow = scanner.check_arbitrage(market_pair(), poly, shallow_kalshi)
        deep = scanner.check_arbitrage(market_pair(), poly, deep_kalshi)

        ranked = scanner.rank_opportunities([shallow, deep])
        assert ranked[0] is deep
        assert deep.rank_score > shallow.rank_score
