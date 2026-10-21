# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Pytest E2E router-replay perf benchmark driven by a JSON case file."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from helpers.replay_run import validate_completed_run

_SUITE_HELPERS = Path(__file__).resolve().parent / "helpers"


def _load_agentaware_helper(stem: str):
    mod_name = f"_agentaware_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = _SUITE_HELPERS / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agentaware helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_case_config = _load_agentaware_helper("case_config")
_replay_runner = _load_agentaware_helper("replay_runner")
RouterReplayE2EConfig = _case_config.RouterReplayE2EConfig
run_router_replay = _replay_runner.run_router_replay


@pytest.mark.e2e_perf
def test_router_replay_completes(pytestconfig: pytest.Config) -> None:
    """Run one router-replay case selected by ``--test-config-file``."""

    test_config_file = pytestconfig.getoption("--test-config-file")
    if not test_config_file:
        pytest.skip("--test-config-file is required for router-replay perf benchmarks")

    config = RouterReplayE2EConfig.from_case_file(
        Path(test_config_file),
        task_num=pytestconfig.getoption("--task-num"),
        max_concurrency=pytestconfig.getoption("--max-concurrency"),
    )
    run_dir = run_router_replay(config)
    summary = validate_completed_run(run_dir)
    print(
        f"[agentaware:{config.router_arm}] completed: "
        f"tasks={summary['tasks']['completed']} requests={summary['requests']['requests']} "
        f"dir={run_dir}",
        flush=True,
    )
