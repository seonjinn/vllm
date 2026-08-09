# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from threading import Lock

from vllm.config.compilation import CUDAGraphMode

_RUNTIME_MODES = (
    CUDAGraphMode.NONE,
    CUDAGraphMode.PIECEWISE,
    CUDAGraphMode.FULL,
)


@dataclass
class CUDAGraphModeCounts:
    dispatches: int = 0
    requests: int = 0
    tokens: int = 0

    def add(self, *, num_requests: int, num_tokens: int) -> None:
        self.dispatches += 1
        self.requests += num_requests
        self.tokens += num_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "dispatches": self.dispatches,
            "requests": self.requests,
            "tokens": self.tokens,
        }


class CUDAGraphDispatchMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counts = self._empty_counts()

    @staticmethod
    def _empty_counts() -> dict[CUDAGraphMode, CUDAGraphModeCounts]:
        return {mode: CUDAGraphModeCounts() for mode in _RUNTIME_MODES}

    @staticmethod
    def _validate_count(value: int, *, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    def observe(
        self,
        mode: CUDAGraphMode,
        *,
        num_requests: int,
        num_tokens: int,
    ) -> None:
        if not isinstance(mode, CUDAGraphMode) or not mode.is_valid_runtime_mode():
            raise ValueError(f"Expected a concrete CUDA graph runtime mode, got {mode}")
        self._validate_count(num_requests, name="num_requests")
        self._validate_count(num_tokens, name="num_tokens")
        with self._lock:
            self._counts[mode].add(
                num_requests=num_requests,
                num_tokens=num_tokens,
            )

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return self._json_snapshot()

    def _json_snapshot(self) -> dict[str, object]:
        return {mode.name: self._counts[mode].to_dict() for mode in _RUNTIME_MODES}

    def reset(self) -> None:
        with self._lock:
            self._counts = self._empty_counts()
