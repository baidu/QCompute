# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ParameterExpression.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x19ParameterExpression.proto\"u\n\x13ParameterExpression\x12\x14\n\nargumentId\x18\x01 \x01(\x05H\x00\x12\x17\n\rargumentValue\x18\x02 \x01(\x01H\x00\x12!\n\x08operator\x18\x03 \x01(\x0e\x32\r.MathOperatorH\x00\x42\x0c\n\nexpression\"4\n\x0e\x45xpressionList\x12\"\n\x04list\x18\x01 \x03(\x0b\x32\x14.ParameterExpression*\x8c\x01\n\x0cMathOperator\x12\x07\n\x03NEG\x10\x00\x12\x07\n\x03POS\x10\x01\x12\x07\n\x03\x41\x42S\x10\x02\x12\x07\n\x03\x41\x44\x44\x10\x03\x12\x07\n\x03SUB\x10\x04\x12\x07\n\x03MUL\x10\x05\x12\x0b\n\x07TRUEDIV\x10\x06\x12\x0c\n\x08\x46LOORDIV\x10\x07\x12\x07\n\x03MOD\x10\x08\x12\x07\n\x03POW\x10\t\x12\x07\n\x03SIN\x10\n\x12\x07\n\x03\x43OS\x10\x0b\x12\x07\n\x03TAN\x10\x0c\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'ParameterExpression_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _MATHOPERATOR._serialized_start=203
  _MATHOPERATOR._serialized_end=343
  _PARAMETEREXPRESSION._serialized_start=29
  _PARAMETEREXPRESSION._serialized_end=146
  _EXPRESSIONLIST._serialized_start=148
  _EXPRESSIONLIST._serialized_end=200
# @@protoc_insertion_point(module_scope)
