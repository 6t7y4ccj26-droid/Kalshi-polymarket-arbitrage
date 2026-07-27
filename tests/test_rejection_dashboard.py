from dashboard.server import DashboardState, get_embedded_html


def test_dashboard_state_exposes_rejection_diagnostics():
    state = DashboardState()
    diagnostic = {
        "polymarket_question": "Will example happen?",
        "kalshi_title": "Example happens",
        "rejection_reasons": ["missing_settlement_evidence"],
        "manual_review_recommended": True,
    }
    state.cross_platform["rejection_diagnostics"] = [diagnostic]

    serialized = state.to_dict()

    assert serialized["cross_platform"]["rejection_diagnostics"] == [diagnostic]


def test_embedded_dashboard_renders_diagnostics_safely():
    html = get_embedded_html()

    assert 'id="diagnosticsGrid"' in html
    assert "function updateRejectionDiagnostics" in html
    assert "escapeHtml(truncate(item.polymarket_question" in html
    assert "A diagnostic is not an arbitrage alert." in html
