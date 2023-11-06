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

r"""
Module for protocol and request queue in resource management.
"""

from abc import ABC
from typing import List
from Extensions.QuantumNetwork.qcompute_qnet.protocols.protocol import Protocol

__all__ = ["RMP", "RequestQueue"]


class RMP(Protocol, ABC):
    r"""Abstract protocol class for the resource management protocol.

    Args:
        name (str): name of the protocol
    """

    def __init__(self, name: str):
        r"""Constructor of the resource management protocol.

        Args:
             name (str): name of the resource management protocol
        """
        super().__init__(name)


class RequestQueue:
    r"""Class for the request queue in resource management protocol.

    Attributes:
        max_volume (int): maximum number of requests that the request queue can accommodate
        current_volume (int): current number of requests in the request queue
        requests (List[ClassicalMessage]): the list of the requests
        node (Node): the node that holds the request queue
    """

    def __init__(self, node: "Node", max_volume: int):
        r"""Constructor for RequestQueue class.

        Args:
            node (Node): the node that holds the request queue
            max_volume (int): maximum number of requests that the request queue can accommodate
        """
        self.node = node
        self.max_volume = max_volume
        self.current_volume = 0
        self.requests = []

    def push(self, request: "ClassicalMessage") -> bool:
        r"""Push a request into the request queue.

        Args:
            request (ClassicalMessage): the request to be pushed

        Returns:
            bool: whether the request is pushed into the request queue successfully
        """
        if self.current_volume != self.max_volume:
            self.requests.insert(0, request)
            self.current_volume += 1
            return True
        else:
            return False

    def pop(self, strategy: str = "_fcfs") -> "ClassicalMessage":
        r"""Pop a request from the request queue according to some certain strategies.
        The first-come-first-served algorithm is adopted as the default strategy.

        Args:
            strategy (str): the strategy for popping the request

        Returns:
            ClassicalMessage: the request to execute
        """
        assert self.current_volume != 0, f"There's no request in the request queue of {self.node.name}"

        # Pop a request according to a certain strategy
        func = getattr(self, strategy)
        self.current_volume -= 1
        return func()

    def _fcfs(self) -> "ClassicalMessage":
        r"""First-come-first-served strategy.

        Returns:
            ClassicalMessage: popped request selected by the first-come-first-served strategy
        """
        return self.requests.pop()
