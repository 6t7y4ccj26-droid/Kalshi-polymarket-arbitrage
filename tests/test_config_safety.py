from argparse import Namespace
from dataclasses import replace

import pytest

import main
import run_with_dashboard
from utils.config_loader import (
    ConfigError,
    get_default_config,
    load_config,
    _validate_config,
)


def test_checked_in_configuration_is_paper_only():
    config = load_config("config.yaml")

    assert config.is_dry_run
    assert config.mode.auto_execution_enabled is False


def test_live_trading_is_rejected_for_this_milestone():
    config = get_default_config()
    config.mode = replace(config.mode, trading_mode="live")

    with pytest.raises(ConfigError, match="must remain 'dry_run'"):
        _validate_config(config)


def test_auto_execution_is_rejected_for_this_milestone():
    config = get_default_config()
    config.mode = replace(config.mode, auto_execution_enabled=True)

    with pytest.raises(ConfigError, match="auto_execution_enabled must remain false"):
        _validate_config(config)


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", [main.main_async, run_with_dashboard.main_async])
async def test_live_cli_override_cannot_bypass_paper_mode(entrypoint):
    args = Namespace(
        config="config.yaml",
        live=True,
        dry_run=False,
        backtest=False,
        backtest_duration=1,
        port=8888,
    )

    with pytest.raises(SystemExit) as error:
        await entrypoint(args)

    assert error.value.code == 2
