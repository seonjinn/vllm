# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from runtime_bootstrap import configure_runtime
from shape_tactic_runtime import install_from_environment

configure_runtime()
install_from_environment()
