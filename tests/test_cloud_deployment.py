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
    assert service["envVars"] == [{"key": "PAPER_SCANNER_ONLY", "value": "true"}]
