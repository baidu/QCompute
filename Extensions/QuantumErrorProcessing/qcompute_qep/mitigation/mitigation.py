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
Abstract interface ``Mitigator`` for the Quantum Gate Error Mitigation methods.
Concrete quantum gate error mitigation methods must inherit this abstract class and implement the ``mitigate`` method.
"""

import abc
from typing import Any


class Mitigator(abc.ABC):
    r"""The Abstract Error Mitigation Class.

    Each inherited class must implement the ``mitigate`` method.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.mitigate(*args, **kwargs)

    @abc.abstractmethod
    def mitigate(self, *args: Any, **kwargs: Any) -> Any:
        r"""The abstract mitigate function.

        Each inherited class must implement this method.
        """
        raise NotImplementedError
