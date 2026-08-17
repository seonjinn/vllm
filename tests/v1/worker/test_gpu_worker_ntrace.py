# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

from vllm.config import ProfilerConfig
from vllm.v1.worker.gpu_worker import Worker


def test_profiler_config_accepts_ntrace() -> None:
    config = ProfilerConfig(profiler="ntrace")

    assert config.profiler == "ntrace"


def test_ntrace_profile_uses_rollout_boundaries() -> None:
    worker = Worker.__new__(Worker)
    worker.profiler_config = SimpleNamespace(profiler="ntrace")
    worker._ntrace_rollout_controller = Mock()

    worker.profile(is_start=True, profile_prefix="decode")
    worker.profile(is_start=False)

    assert worker._ntrace_rollout_controller.method_calls == [
        call.begin_rollout(step_id="decode"),
        call.finish_rollout(),
        call.assert_trace_completed(),
    ]


def test_ntrace_wraps_cuda_graph_capture() -> None:
    events: list[str] = []
    controller = Mock()
    controller.begin_engine_initialization.side_effect = lambda **_: (
        events.append("begin") or "token"
    )
    controller.end_engine_initialization.side_effect = lambda token: events.append(
        f"end:{token}"
    )
    model_runner = Mock()
    model_runner.capture_model.side_effect = lambda: events.append("capture") or 123

    worker = Worker.__new__(Worker)
    worker._ntrace_rollout_controller = controller
    worker.model_runner = model_runner

    assert worker._capture_model_with_ntrace() == 123
    assert events == ["begin", "capture", "end:token"]
