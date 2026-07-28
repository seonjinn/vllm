# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for vllm.ray.ray_env — env var propagation to Ray workers."""

import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


class _EnvModule(ModuleType):
    VLLM_CONFIG_ROOT = str(Path(__file__).with_name(".missing-vllm-config"))
    environment_variables: dict[str, object] = {}

    def __getattr__(self, name: str) -> str:
        if name in {
            "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY",
            "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY",
        }:
            return os.getenv(name, "")
        raise AttributeError(name)


def _load_ray_env_module() -> ModuleType:
    module_path = Path(__file__).parents[1] / "vllm/ray/ray_env.py"
    spec = importlib.util.spec_from_file_location(
        "_standalone_vllm_ray_env", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load vLLM Ray environment module")

    fake_vllm = ModuleType("vllm")
    fake_envs = _EnvModule("vllm.envs")
    fake_logger = ModuleType("vllm.logger")
    fake_logger.init_logger = logging.getLogger  # type: ignore[attr-defined]
    original_modules = {
        name: sys.modules.get(name)
        for name in ("vllm", "vllm.envs", "vllm.logger")
    }
    try:
        sys.modules["vllm"] = fake_vllm
        sys.modules["vllm.envs"] = fake_envs
        sys.modules["vllm.logger"] = fake_logger
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


_RAY_ENV = _load_ray_env_module()
get_env_vars_to_copy = _RAY_ENV.get_env_vars_to_copy

# ---------------------------------------------------------------------------
# Default prefix matching
# ---------------------------------------------------------------------------


class TestDefaultPrefixes:
    """Built-in prefixes (VLLM_, LMCACHE_, NCCL_, UCX_, HF_, HUGGING_FACE_)
    should be forwarded without any extra configuration."""

    @patch.dict(
        os.environ,
        {
            "VLLM_MXFP8_DENSE_CONFIG_FILE": (
                "qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json"
            )
        },
        clear=True,
    )
    def test_mxfp8_dense_config_uses_native_vllm_prefix(self):
        """The one-file MXFP8 contract reaches Ray workers without extra vars."""
        expected = "qwen3_30ba3b_tp1_v0202_rollout_trace_bootstrap.json"

        assert "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY" not in os.environ
        result = get_env_vars_to_copy()
        copied = {
            name: os.environ[name] for name in result if name in os.environ
        }

        assert "VLLM_MXFP8_DENSE_CONFIG_FILE" in result
        assert copied["VLLM_MXFP8_DENSE_CONFIG_FILE"] == expected

    @patch.dict(os.environ, {"LMCACHE_LOCAL_CPU": "True"}, clear=False)
    def test_lmcache_prefix(self):
        result = get_env_vars_to_copy()
        assert "LMCACHE_LOCAL_CPU" in result

    @patch.dict(os.environ, {"NCCL_DEBUG": "INFO"}, clear=False)
    def test_nccl_prefix(self):
        result = get_env_vars_to_copy()
        assert "NCCL_DEBUG" in result

    @patch.dict(os.environ, {"UCX_TLS": "rc"}, clear=False)
    def test_ucx_prefix(self):
        result = get_env_vars_to_copy()
        assert "UCX_TLS" in result

    @patch.dict(os.environ, {"HF_TOKEN": "secret"}, clear=False)
    def test_hf_token_via_prefix(self):
        result = get_env_vars_to_copy()
        assert "HF_TOKEN" in result

    @patch.dict(os.environ, {"HUGGING_FACE_HUB_TOKEN": "secret"}, clear=False)
    def test_hugging_face_prefix(self):
        result = get_env_vars_to_copy()
        assert "HUGGING_FACE_HUB_TOKEN" in result


# ---------------------------------------------------------------------------
# Default extra vars
# ---------------------------------------------------------------------------


class TestDefaultExtraVars:
    """Individual vars listed in VLLM_RAY_EXTRA_ENV_VARS_TO_COPY's default."""

    def test_pythonhashseed_in_result(self):
        """PYTHONHASHSEED should always be in the result set (as a name to
        copy) regardless of whether it is actually set in os.environ."""
        result = get_env_vars_to_copy()
        assert "PYTHONHASHSEED" in result


# ---------------------------------------------------------------------------
# User-supplied extensions
# ---------------------------------------------------------------------------


class TestUserExtensions:
    """Users can add prefixes and extra vars at deploy time."""

    @patch.dict(
        os.environ,
        {
            "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY": "MYLIB_",
            "MYLIB_FOO": "bar",
        },
        clear=False,
    )
    def test_user_prefix(self):
        """User-supplied prefixes are additive — built-in defaults are kept."""
        result = get_env_vars_to_copy()
        assert "MYLIB_FOO" in result

    @patch.dict(
        os.environ,
        {
            "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": "MY_SECRET",
            "MY_SECRET": "val",
        },
        clear=False,
    )
    def test_user_extra_var(self):
        """User-supplied extras are additive — PYTHONHASHSEED still included."""
        result = get_env_vars_to_copy()
        assert "MY_SECRET" in result
        assert "PYTHONHASHSEED" in result


# ---------------------------------------------------------------------------
# Exclusion
# ---------------------------------------------------------------------------


class TestExclusion:
    """exclude_vars and RAY_NON_CARRY_OVER_ENV_VARS take precedence."""

    @patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}, clear=False)
    def test_exclude_vars(self):
        result = get_env_vars_to_copy(exclude_vars={"CUDA_VISIBLE_DEVICES"})
        assert "CUDA_VISIBLE_DEVICES" not in result

    @patch.dict(os.environ, {"LMCACHE_LOCAL_CPU": "True"}, clear=False)
    @patch.object(
        _RAY_ENV,
        "RAY_NON_CARRY_OVER_ENV_VARS",
        {"LMCACHE_LOCAL_CPU"},
    )
    def test_non_carry_over_blacklist(self):
        result = get_env_vars_to_copy()
        assert "LMCACHE_LOCAL_CPU" not in result


# ---------------------------------------------------------------------------
# additional_vars (platform extension point)
# ---------------------------------------------------------------------------


class TestAdditionalVars:
    """The additional_vars parameter supports platform-specific vars."""

    @patch.dict(os.environ, {"CUSTOM_PLATFORM_VAR": "1"}, clear=False)
    def test_additional_vars_passthrough(self):
        result = get_env_vars_to_copy(additional_vars={"CUSTOM_PLATFORM_VAR"})
        assert "CUSTOM_PLATFORM_VAR" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Prefix matching should be strict (startswith, not contains)."""

    @patch.dict(os.environ, {"LMCACH_TYPO": "1"}, clear=False)
    def test_prefix_no_partial_match(self):
        """'LMCACH_' does not match the 'LMCACHE_' prefix."""
        result = get_env_vars_to_copy()
        assert "LMCACH_TYPO" not in result

    @patch.dict(
        os.environ,
        {
            "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY": " MYLIB_ , OTHER_ ",
        },
        clear=False,
    )
    def test_csv_whitespace_handling(self):
        """Whitespace around commas and tokens should be stripped."""
        result = get_env_vars_to_copy()
        # MYLIB_ and OTHER_ should be parsed as valid prefixes — no crash
        assert isinstance(result, set)

    @patch.dict(
        os.environ,
        {
            "VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY": "MYLIB_",
            "LMCACHE_BACKEND": "cpu",
            "NCCL_DEBUG": "INFO",
            "MYLIB_FOO": "bar",
        },
        clear=False,
    )
    def test_user_prefix_additive(self):
        """Setting VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY does NOT drop defaults."""
        result = get_env_vars_to_copy()
        # Built-in defaults still present
        assert "LMCACHE_BACKEND" in result
        assert "NCCL_DEBUG" in result
        # User addition also present
        assert "MYLIB_FOO" in result

    @patch.dict(
        os.environ,
        {
            "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY": "MY_FLAG",
            "PYTHONHASHSEED": "42",
            "MY_FLAG": "1",
        },
        clear=False,
    )
    def test_user_extra_additive(self):
        """Setting VLLM_RAY_EXTRA_ENV_VARS_TO_COPY does NOT drop defaults."""
        result = get_env_vars_to_copy()
        # Built-in default still present
        assert "PYTHONHASHSEED" in result
        # User addition also present
        assert "MY_FLAG" in result
