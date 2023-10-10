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
Quantum Parameter Expression
"""
FileErrorCode = 5

import copy
import enum
from enum import IntEnum
from typing import Union, TYPE_CHECKING, Type, List

if TYPE_CHECKING:
    from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterStorage


@enum.unique
class MathOpEnum(IntEnum):
    # unitary operators
    # - obj
    NEG = 0
    # + obj
    POS = enum.auto()
    # abs(obj)
    ABS = enum.auto()

    # following operators support corresponding
    # reversed (__rxxx__) and in-place (__ixxx__) operators
    # obj + other
    ADD = enum.auto()
    # obj - other
    SUB = enum.auto()
    # obj * other
    MUL = enum.auto()
    # obj / other
    TRUEDIV = enum.auto()
    # obj // other
    FLOORDIV = enum.auto()
    # obj % other
    MOD = enum.auto()
    # obj ** other
    POW = enum.auto()

    # Trigonometric Function
    SIN = enum.auto()
    COS = enum.auto()
    TAN = enum.auto()


class ProcedureParameterExpression:
    """
    The storage for procedure parameter expression
    """

    def __init__(self, value: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']) -> None:
        """
        The quantum param object needs to know its index.

        :param index: the quantum register index
        """
        self.expressionList: List[Union[int, float, 'ProcedureParameterStorage', MathOpEnum]] = None
        valueType = validValue(value)
        if valueType == int or valueType == float:
            self.expressionList = [value]
        elif valueType.__name__ == 'ProcedureParameterStorage':
            self.expressionList = copy.deepcopy(value.expressionList)
        elif valueType.__name__ == 'ProcedureParameterExpression':
            self.expressionList = copy.deepcopy(value.expressionList)

    def __neg__(self):
        ret = copy.deepcopy(self)
        ret.expressionList.append(MathOpEnum.NEG)
        return ret

    def __pos__(self):
        ret = copy.deepcopy(self)
        # ret.expressionList.append(MathOpEnum.POS)
        return ret

    def __abs__(self):
        ret = copy.deepcopy(self)
        ret.expressionList.append(MathOpEnum.ABS)
        return ret

    def __add__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.append(other)
        else:  # ProcedureParameterExpression
            ret.expressionList.extend(other.expressionList)
        ret.expressionList.append(MathOpEnum.ADD)
        return ret

    def __radd__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.insert(0, other)
        else:  # ProcedureParameterExpression
            temp = ret.expressionList
            ret.expressionList = copy.deepcopy(other)
            ret.expressionList.extend(temp)
        ret.expressionList.append(MathOpEnum.ADD)
        return ret

    def __sub__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.append(other)
        else:  # ProcedureParameterExpression
            ret.expressionList.extend(other.expressionList)
        ret.expressionList.append(MathOpEnum.SUB)
        return ret

    def __rsub__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.insert(0, other)
        else:  # ProcedureParameterExpression
            temp = ret.expressionList
            ret.expressionList = copy.deepcopy(other)
            ret.expressionList.extend(temp)
        ret.expressionList.append(MathOpEnum.SUB)
        return ret

    def __mul__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.append(other)
        else:  # ProcedureParameterExpression
            ret.expressionList.extend(other.expressionList)
        ret.expressionList.append(MathOpEnum.MUL)
        return ret

    def __rmul__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.insert(0, other)
        else:  # ProcedureParameterExpression
            temp = ret.expressionList
            ret.expressionList = copy.deepcopy(other)
            ret.expressionList.extend(temp)
        ret.expressionList.append(MathOpEnum.MUL)
        return ret

    def __truediv__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.append(other)
        else:  # ProcedureParameterExpression
            ret.expressionList.extend(other.expressionList)
        ret.expressionList.append(MathOpEnum.TRUEDIV)
        return ret

    def __rtruediv__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.insert(0, other)
        else:  # ProcedureParameterExpression
            temp = ret.expressionList
            ret.expressionList = copy.deepcopy(other)
            ret.expressionList.extend(temp)
        ret.expressionList.append(MathOpEnum.TRUEDIV)
        return ret

    def __floordiv__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.append(other)
        else:  # ProcedureParameterExpression
            ret.expressionList.extend(other.expressionList)
        ret.expressionList.append(MathOpEnum.FLOORDIV)
        return ret

    def __rfloordiv__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.insert(0, other)
        else:  # ProcedureParameterExpression
            temp = ret.expressionList
            ret.expressionList = copy.deepcopy(other)
            ret.expressionList.extend(temp)
        ret.expressionList.append(MathOpEnum.FLOORDIV)
        return ret

    def __mod__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.append(other)
        else:  # ProcedureParameterExpression
            ret.expressionList.extend(other.expressionList)
        ret.expressionList.append(MathOpEnum.MOD)
        return ret

    def __rmod__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.insert(0, other)
        else:  # ProcedureParameterExpression
            temp = ret.expressionList
            ret.expressionList = copy.deepcopy(other)
            ret.expressionList.extend(temp)
        ret.expressionList.append(MathOpEnum.MOD)
        return ret

    def __pow__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.append(other)
        else:  # ProcedureParameterExpression
            ret.expressionList.extend(other.expressionList)
        ret.expressionList.append(MathOpEnum.POW)
        return ret

    def __rpow__(self, other: Union[int, float, 'ProcedureParameterStorage', 'ProcedureParameterExpression']):
        valueType = validValue(other)
        ret = copy.deepcopy(self)
        if valueType == int or valueType == float:
            ret.expressionList.insert(0, other)
        else:  # ProcedureParameterExpression
            temp = ret.expressionList
            ret.expressionList = copy.deepcopy(other)
            ret.expressionList.extend(temp)
        ret.expressionList.append(MathOpEnum.POW)
        return ret

    def sin(self):
        ret = copy.deepcopy(self)
        ret.expressionList.append(MathOpEnum.SIN)
        return ret

    def cos(self):
        ret = copy.deepcopy(self)
        ret.expressionList.append(MathOpEnum.COS)
        return ret

    def tan(self):
        ret = copy.deepcopy(self)
        ret.expressionList.append(MathOpEnum.TAN)
        return ret


def validValue(value) -> Type:
    valueType = type(value)
    assert valueType == int or valueType == float \
           or valueType.__name__ == 'ProcedureParameterStorage' \
           or valueType.__name__ == 'ProcedureParameterExpression'
    return valueType