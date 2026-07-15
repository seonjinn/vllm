# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

from vllm.benchmarks.dynamic_sd import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.utils.argparse_utils import FlexibleArgumentParser


class BenchmarkDynamicSDSubcommand(BenchmarkSubcommandBase):
    """The ``dynamic-sd`` subcommand for ``vllm bench``."""

    name = "dynamic-sd"
    help = "Profile and select Dynamic SD schedules."

    @classmethod
    def add_cli_args(cls, parser: FlexibleArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)
