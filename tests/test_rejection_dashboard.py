from fastapi.testclient import TestClient

from dashboard.server import (
    DashboardState,
    create_app,
    dashboard_state,
    get_embedded_html,
)


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


def test_cloud_health_check_is_paper_only():
    previous_running = dashboard_state.is_running
    previous_mode = dashboard_state.mode
    dashboard_state.is_running = True
    dashboard_state.mode = "dry_run"
    try:
        response = TestClient(create_app()).get("/api/health")
    finally:
        dashboard_state.is_running = previous_running
        dashboard_state.mode = previous_mode

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "mode": "dry_run",
        "auto_execution_enabled": False,
    }
