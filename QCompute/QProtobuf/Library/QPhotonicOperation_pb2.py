# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: QPhotonicOperation.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18QPhotonicOperation.proto\"\xe8\x02\n\x17PhotonicGaussianMeasure\x12+\n\x04type\x18\x01 \x01(\x0e\x32\x1d.PhotonicGaussianMeasure.Type\x12\x10\n\x08\x63RegList\x18\x02 \x03(\r\x12?\n\nheterodyne\x18\x03 \x03(\x0b\x32+.PhotonicGaussianMeasure.HeterodyneArgument\x12\x41\n\x0bphotonCount\x18\x04 \x01(\x0b\x32,.PhotonicGaussianMeasure.PhotonCountArgument\x1a,\n\x12HeterodyneArgument\x12\x0b\n\x03phi\x18\x01 \x01(\x01\x12\t\n\x01r\x18\x02 \x01(\x01\x1a%\n\x13PhotonCountArgument\x12\x0e\n\x06\x63utoff\x18\x01 \x01(\r\"5\n\x04Type\x12\x0c\n\x08Homodyne\x10\x00\x12\x0e\n\nHeterodyne\x10\x01\x12\x0f\n\x0bPhotonCount\x10\x02\"7\n\x13PhotonicFockMeasure\x12\x10\n\x08\x63RegList\x18\x01 \x03(\r\x12\x0e\n\x06\x63utoff\x18\x02 \x01(\r*\x8b\x02\n\x14PhotonicGaussianGate\x12\x16\n\x12PhotonicGaussianDX\x10\x00\x12\x16\n\x12PhotonicGaussianDP\x10\x01\x12\x17\n\x13PhotonicGaussianPHA\x10\x02\x12\x16\n\x12PhotonicGaussianBS\x10\x03\x12\x16\n\x12PhotonicGaussianCZ\x10\x04\x12\x16\n\x12PhotonicGaussianCX\x10\x05\x12\x17\n\x13PhotonicGaussianDIS\x10\x06\x12\x17\n\x13PhotonicGaussianSQU\x10\x07\x12\x18\n\x14PhotonicGaussianTSQU\x10\x08\x12\x16\n\x12PhotonicGaussianMZ\x10\t*c\n\x10PhotonicFockGate\x12\x12\n\x0ePhotonicFockAP\x10\x00\x12\x13\n\x0fPhotonicFockPHA\x10\x01\x12\x12\n\x0ePhotonicFockBS\x10\x02\x12\x12\n\x0ePhotonicFockMZ\x10\x03\x62\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'QPhotonicOperation_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _PHOTONICGAUSSIANGATE._serialized_start=449
  _PHOTONICGAUSSIANGATE._serialized_end=716
  _PHOTONICFOCKGATE._serialized_start=718
  _PHOTONICFOCKGATE._serialized_end=817
  _PHOTONICGAUSSIANMEASURE._serialized_start=29
  _PHOTONICGAUSSIANMEASURE._serialized_end=389
  _PHOTONICGAUSSIANMEASURE_HETERODYNEARGUMENT._serialized_start=251
  _PHOTONICGAUSSIANMEASURE_HETERODYNEARGUMENT._serialized_end=295
  _PHOTONICGAUSSIANMEASURE_PHOTONCOUNTARGUMENT._serialized_start=297
  _PHOTONICGAUSSIANMEASURE_PHOTONCOUNTARGUMENT._serialized_end=334
  _PHOTONICGAUSSIANMEASURE_TYPE._serialized_start=336
  _PHOTONICGAUSSIANMEASURE_TYPE._serialized_end=389
  _PHOTONICFOCKMEASURE._serialized_start=391
  _PHOTONICFOCKMEASURE._serialized_end=446
# @@protoc_insertion_point(module_scope)
