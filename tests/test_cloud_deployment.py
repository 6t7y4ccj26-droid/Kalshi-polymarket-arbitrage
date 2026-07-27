from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_container_is_hardwired_to_dry_run():
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "run_with_dashboard.py --dry-run" in dockerfile
    assert "--live" not in dockerfile
    assert "${PORT:-10000}" in dockerfile
    assert "USER scanner" in dockerfile


def test_render_blueprint_is_single_instance_paid_paper_service():
    blueprint = yaml.safe_load((ROOT / "render.yaml").read_text())
    service = blueprint["services"][0]

    assert service["type"] == "web"
    assert service["runtime"] == "docker"
    assert service["plan"] == "starter"
    assert service["branch"] == "main"
    assert service["numInstances"] == 1
    assert service["healthCheckPath"] == "/api/health"
    assert service["disk"] == {
        "name": "scanner-logs",
        "mountPath": "/app/logs",
        "sizeGB": 1,
    }
    assert service["envVars"] == [{"key": "PAPER_SCANNER_ONLY", "value": "true"}]


def test_free_workflow_is_scheduled_paper_only_and_bounded():
    workflow = (ROOT / ".github/workflows/free-paper-scanner.yml").read_text()

    assert 'cron: "17 */6 * * *"' in workflow
    assert "workflow_dispatch:" in workflow
    assert "timeout-minutes: 350" in workflow
    assert "run_with_dashboard.py --dry-run" in workflow
    assert "--live" not in workflow
    assert "python -m pytest -q" in workflow
    assert "issues: write" in workflow
    assert 'grep -c "PAPER opportunity:"' in workflow
    assert "gh issue create" in workflow
    assert "No trades were placed." in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "retention-days: 7" in workflow
