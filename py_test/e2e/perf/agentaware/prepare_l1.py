#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Router-replay L1 prerequisite: start serve, calibrate, write program JSON, stop serve.

Run from the repository root:

    python py_test/e2e/perf/agentaware/prepare_l1.py \\
        --serve-config-file py_test/e2e/perf/agentaware/configs/l1-serve.json

Stable artifacts (default ``results/agentaware/l1/``):

- ``program-scheduling.json`` — ``cases/program_scheduling/``
- ``program-scheduling-binding-only.json`` — ``cases/binding_only/``

L2 pytest cases only reference these paths; they do not re-run calibration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_e2e_paths_module(root: Path):
    import importlib.util

    paths_file = root / "py_test" / "e2e" / "helpers" / "paths.py"
    spec = importlib.util.spec_from_file_location("_e2e_paths", paths_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load E2E paths module from {paths_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_import_paths() -> Path:
    suite_dir = Path(__file__).resolve().parent
    root = suite_dir.parents[4]
    for path in (root, suite_dir):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)

    paths_mod = _load_e2e_paths_module(root)
    paths_mod.ensure_agent_infer_tools_on_path()
    return paths_mod.repo_root()


def _print_next_steps(output_root: Path) -> None:
    root = output_root.expanduser().resolve()
    program = root / "program-scheduling.json"
    binding = root / "program-scheduling-binding-only.json"
    suite = Path(__file__).resolve().parent
    print(
        "[prepare_l1] L1 complete — run perf cases (each starts its own serve + router):",
        flush=True,
    )
    print(f"  binding_only arm       → {binding}", flush=True)
    print(f"  program_scheduling arm → {program}", flush=True)
    print("Example:", flush=True)
    print(
        "  pytest -s -v py_test/e2e/perf/agentaware/run_benchmark.py \\",
        flush=True,
    )
    print(
        f"    --test-config-file {suite / 'cases/binding_only/glm52_24x12_router_replay.json'}",
        flush=True,
    )


def main() -> None:
    root = _ensure_import_paths()
    os.chdir(root)
    from helpers.l1_prepare import bootstrap, parser as l1_parser

    cli = l1_parser()
    cli.prog = Path(__file__).name
    cli.description = (
        "Start vLLM from a serve config, run prefill/decode calibration, write both program JSON files, "
        "then stop serve."
    )
    args = cli.parse_args()
    bootstrap(args)
    _print_next_steps(Path(args.output_root))


if __name__ == "__main__":
    main()
