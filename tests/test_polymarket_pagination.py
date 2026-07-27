from unittest.mock import AsyncMock

import httpx
import pytest

from polymarket_client.api import PolymarketClient


def gamma_market(index: int) -> dict:
    return {
        "id": str(index),
        "conditionId": f"condition-{index}",
        "question": f"Market {index}",
        "clobTokenIds": f'["yes-{index}", "no-{index}"]',
        "active": True,
        "closed": False,
    }


def pagination_error() -> httpx.HTTPStatusError:
    request = httpx.Request(
        "GET", "https://gamma-api.polymarket.com/markets?offset=100"
    )
    response = httpx.Response(422, request=request)
    return httpx.HTTPStatusError(
        "Unprocessable Entity", request=request, response=response
    )


@pytest.mark.asyncio
async def test_list_markets_keeps_complete_pages_when_gamma_rejects_terminal_offset():
    client = PolymarketClient(dry_run=True)
    first_page = [gamma_market(index) for index in range(100)]
    client._request = AsyncMock(side_effect=[first_page, pagination_error()])

    markets = await client.list_markets({"active": True})

    assert len(markets) == 100
    assert client._request.await_count == 2
    assert len(client._markets_cache) == 100


@pytest.mark.asyncio
async def test_list_markets_does_not_hide_invalid_initial_request():
    client = PolymarketClient(dry_run=True)
    client._request = AsyncMock(side_effect=pagination_error())

    with pytest.raises(httpx.HTTPStatusError):
        await client.list_markets({"active": True})
