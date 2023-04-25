#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
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
The abstract `StabilizerCode` class for the quantum error correction module.
The implementations of various stabilizer codes must inherit this abstract class.
"""
import abc
from typing import Any, List

from qcompute_qep.utils.types import QProgram


class StabilizerCode(abc.ABC):
    r"""The stabilizer error correction code abstract class.
    """
    def __init__(self, stabilizers: List[str], **kwargs: Any):
        r"""init function of the `StabilizerCode` class.

        :param stabilizers: List[str], a list of stabilizer generators, each is a Pauli string
        """
        self._stabilizers = stabilizers

    @property
    def n(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @property
    def k(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @property
    def r(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @property
    def syndrome_dict(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @property
    def n_k_d(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @property
    def stabilizers(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @property
    def check_matrix(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @property
    def standard_form(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @property
    def logical_xs(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @property
    def logical_zs(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @property
    def name(self):
        r"""The name of the stabilizer code.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sanity_check(self, **kwargs):
        r"""
        """
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, qp: QProgram, **kwargs):
        r"""Append the encoding circuit to the quantum program.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def correct(self, qp: QProgram, **kwargs):
        r"""Append the correction circuit to the quantum program.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, qp: QProgram, **kwargs):
        r"""Append the decoding circuit to the quantum program.
        """
        raise NotImplementedError
