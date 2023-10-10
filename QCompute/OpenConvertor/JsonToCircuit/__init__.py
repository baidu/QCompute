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
Convert the json to circuit
"""
FileErrorCode = 11

from google.protobuf.json_format import Parse

from QCompute.OpenConvertor import ConvertorImplement
from QCompute.QProtobuf import PBProgram


class JsonToCircuit(ConvertorImplement):
    """
    Json to circuit
    """

    def convert(self, jsonStr: str) -> 'PBProgram':
        """
        Convert the json to circuit.

        Example:

        program = JsonToCircuit().convert(jsonStr)

        :param jsonStr: json str
        :return: Protobuf format of the circuit
        """

        return Parse(jsonStr, PBProgram())