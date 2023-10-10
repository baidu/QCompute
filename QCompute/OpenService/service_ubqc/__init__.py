#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Universal blind quantum computation service
"""
FileErrorCode = 2

from datetime import datetime
from typing import Dict

from QCompute.Define import MeasureFormat
from QCompute.OpenService.service_ubqc.client.client import PlatformClient
from QCompute.OpenSimulator import QImplement, QResult
from QCompute.QPlatform.Processor.PostProcessor import formatMeasure
from QCompute.OpenService import ModuleErrorCode
from QCompute.QPlatform import Error

__all__ = ["Backend"]


class Backend(QImplement):
    """
    Universal blind quantum computation backend
    """

    def commit(self) -> None:
        """
        Commit the circuit to universal blind quantum computation backend

        .. code-block:: python

            env = QEnv()
            env.backend(BackendName.ServiceUbqc)
        """
        # Collect the result to simulator for subsequent invoking
        self.result = QResult()
        self.result.startTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'

        ret: Dict[str, int] = {}
        width = 0

        client = PlatformClient(self.shots, self.program)

        if client.failed:
            raise Error.ArgumentError('Computing Failed!', ModuleErrorCode, FileErrorCode, 1)
        else:
            self.result.endTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'
            self.result.counts = formatMeasure(client.ret, width, MeasureFormat.Bin)
            self.result.shots = self.shots
            self.result.output = self.result.toJson(True)