# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cortex Training client v2, built on the Arctic Platform client.

Install with ``pip install "cortex-training[client-v2]"`` (Python >= 3.12).
"""

from cortex_training.client.core import Hardware
from cortex_training.client.core import InferenceConfig
from cortex_training.client.core import JobType
from cortex_training.client.core import SubJobConfig
from cortex_training.client.core import TrainingConfig
from cortex_training.client_v2.core import CortexTrainingClient
from cortex_training.model_deployment import ModelDeploymentSpec

__all__ = [
    "CortexTrainingClient",
    "Hardware",
    "InferenceConfig",
    "JobType",
    "ModelDeploymentSpec",
    "SubJobConfig",
    "TrainingConfig",
]
