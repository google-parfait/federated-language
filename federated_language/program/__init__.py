# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Libraries for creating federated programs."""

# pylint: disable=g-importing-member
from federated_language.program.data_source import FederatedDataSource
from federated_language.program.data_source import FederatedDataSourceIterator
from federated_language.program.federated_context import check_in_federated_context
from federated_language.program.federated_context import ComputationArg
from federated_language.program.federated_context import contains_only_server_placed_data
from federated_language.program.federated_context import FederatedContext
from federated_language.program.logging_release_manager import LoggingReleaseManager
from federated_language.program.memory_release_manager import MemoryReleaseManager
from federated_language.program.native_platform import NativeFederatedContext
from federated_language.program.native_platform import NativeValueReference
from federated_language.program.program_state_manager import ProgramStateExistsError
from federated_language.program.program_state_manager import ProgramStateManager
from federated_language.program.program_state_manager import ProgramStateNotFoundError
from federated_language.program.program_state_manager import ProgramStateStructure
from federated_language.program.program_state_manager import ProgramStateValue
from federated_language.program.release_manager import DelayedReleaseManager
from federated_language.program.release_manager import FilteringReleaseManager
from federated_language.program.release_manager import GroupingReleaseManager
from federated_language.program.release_manager import Key
from federated_language.program.release_manager import NotFilterableError
from federated_language.program.release_manager import PeriodicReleaseManager
from federated_language.program.release_manager import ReleasableStructure
from federated_language.program.release_manager import ReleasableValue
from federated_language.program.release_manager import ReleasedValueNotFoundError
from federated_language.program.release_manager import ReleaseManager
from federated_language.program.value_reference import MaterializableStructure
from federated_language.program.value_reference import MaterializableTypeSignature
from federated_language.program.value_reference import MaterializableValue
from federated_language.program.value_reference import MaterializableValueReference
from federated_language.program.value_reference import materialize_value
from federated_language.program.value_reference import MaterializedStructure
from federated_language.program.value_reference import MaterializedValue
# pylint: enable=g-importing-member
