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
Module for utility functions in discrete-event simulation.
"""

import math
from typing import List
from io import StringIO

__all__ = [
    "show_heap",
    "is_heap"
]


def show_heap(heap: List["Event"]) -> None:
    r"""Print the heap.

    Args:
        heap (List[Event]): heap to show
    """

    width = 180
    output = StringIO()
    last_row = -1
    for i, n in enumerate(heap):
        if i:
            row = int(math.floor(math.log(i + 1, 2)))
        else:
            row = 0
        if row != last_row:
            output.write('\n')
        columns = 2 ** row
        col_width = int(math.floor((width * 1.0) / columns))
        output.write(f"ev {i}: '{n.handler.owner.name}.{n.handler.method}' (t = {n.time}, p = {n.priority})"
                     .center(col_width, " "))
        last_row = row
    print('-' * width + "\nEvent list heap:" + output.getvalue() + '\n' + '-' * width)


def is_heap(list_: list) -> bool:
    r"""Check if the list is a heap.

    Args:
        list_ (list): list to be checked

    Returns:
        bool: whether the list is a heap
    """
    bool_ = []
    for i in range((len(list_) - 3) // 2 + 1):
        bool1 = (list_[i] < list_[2 * i + 1]) or (list_[i] == list_[2 * i + 1])
        bool2 = (list_[i] < list_[2 * i + 2]) or (list_[i] == list_[2 * i + 2])
        bool_.append(bool1 and bool2)
    if len(list_) % 2:
        return all(bool_)
    else:
        return all(bool_) and (list_[(len(list_) - 1) // 2] < list_[-1] or list_[(len(list_) - 1) // 2] == list_[-1])
