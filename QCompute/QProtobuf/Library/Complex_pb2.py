# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: Complex.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='Complex.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\rComplex.proto\"3\n\x07\x43omplex\x12\x0c\n\x04real\x18\x01 \x01(\x01\x12\x11\n\x04imag\x18\x02 \x01(\x01H\x00\x88\x01\x01\x42\x07\n\x05_imag\"0\n\x06Matrix\x12\r\n\x05shape\x18\x01 \x03(\r\x12\x17\n\x05\x61rray\x18\x02 \x03(\x0b\x32\x08.Complexb\x06proto3'
)




_COMPLEX = _descriptor.Descriptor(
  name='Complex',
  full_name='Complex',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='real', full_name='Complex.real', index=0,
      number=1, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='imag', full_name='Complex.imag', index=1,
      number=2, type=1, cpp_type=5, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='_imag', full_name='Complex._imag',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=17,
  serialized_end=68,
)


_MATRIX = _descriptor.Descriptor(
  name='Matrix',
  full_name='Matrix',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='Matrix.shape', index=0,
      number=1, type=13, cpp_type=3, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='array', full_name='Matrix.array', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=70,
  serialized_end=118,
)

_COMPLEX.oneofs_by_name['_imag'].fields.append(
  _COMPLEX.fields_by_name['imag'])
_COMPLEX.fields_by_name['imag'].containing_oneof = _COMPLEX.oneofs_by_name['_imag']
_MATRIX.fields_by_name['array'].message_type = _COMPLEX
DESCRIPTOR.message_types_by_name['Complex'] = _COMPLEX
DESCRIPTOR.message_types_by_name['Matrix'] = _MATRIX
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Complex = _reflection.GeneratedProtocolMessageType('Complex', (_message.Message,), {
  'DESCRIPTOR' : _COMPLEX,
  '__module__' : 'Complex_pb2'
  # @@protoc_insertion_point(class_scope:Complex)
  })
_sym_db.RegisterMessage(Complex)

Matrix = _reflection.GeneratedProtocolMessageType('Matrix', (_message.Message,), {
  'DESCRIPTOR' : _MATRIX,
  '__module__' : 'Complex_pb2'
  # @@protoc_insertion_point(class_scope:Matrix)
  })
_sym_db.RegisterMessage(Matrix)


# @@protoc_insertion_point(module_scope)