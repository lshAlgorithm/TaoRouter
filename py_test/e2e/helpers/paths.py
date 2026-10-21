# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AgentInfer project
"""Repository-root resolution for E2E perf (run pytest from the vllm-router repo root)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Built with `cargo build --release` in this repository.
DEFAULT_ROUTER_BINARY = Path("target/release/vllm-router")


def repo_root() -> Path:
    """vllm-router repository root (``py_test/e2e/helpers`` → four levels up)."""

    return Path(__file__).resolve().parents[3]


def ensure_repo_on_sys_path() -> Path:
    """Insert repo root on ``sys.path`` so optional scripts can import ``py_test`` helpers."""

    root = repo_root()
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root


def ensure_agent_infer_tools_on_path() -> Path | None:
    """When ``AGENT_INFER_ROOT`` is set, add AgentInfer repo root for ``tools.calibration`` (L1)."""

    raw = os.environ.get("AGENT_INFER_ROOT", "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"AGENT_INFER_ROOT is not a directory: {root}")
    tools = root / "tools"
    if not tools.is_dir():
        raise FileNotFoundError(f"AGENT_INFER_ROOT has no tools/ directory: {root}")
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    return root
