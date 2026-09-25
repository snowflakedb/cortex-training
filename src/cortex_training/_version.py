# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Installed package version and HTTP user agent."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version


try:
    __version__ = version("cortex-training")
except PackageNotFoundError:  # Source tree imported without an installation.
    __version__ = "0.0.3"

USER_AGENT = f"cortex-training/{__version__}"
