# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: generativesampler.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17generativesampler.proto\x12\x1bnvidia.cheminformatics.grpc\x1a\x1bgoogle/protobuf/empty.proto\"\x99\x02\n\x0eGenerativeSpec\x12;\n\x05model\x18\x01 \x01(\x0e\x32,.nvidia.cheminformatics.grpc.GenerativeModel\x12\x0e\n\x06smiles\x18\x02 \x03(\t\x12\x13\n\x06radius\x18\x03 \x01(\x02H\x00\x88\x01\x01\x12\x19\n\x0cnumRequested\x18\x04 \x01(\x05H\x01\x88\x01\x01\x12\x14\n\x07padding\x18\x05 \x01(\x05H\x02\x88\x01\x01\x12\x18\n\x0b\x66orceUnique\x18\x06 \x01(\x08H\x03\x88\x01\x01\x12\x15\n\x08sanitize\x18\x07 \x01(\x08H\x04\x88\x01\x01\x42\t\n\x07_radiusB\x0f\n\r_numRequestedB\n\n\x08_paddingB\x0e\n\x0c_forceUniqueB\x0b\n\t_sanitize\"e\n\nSmilesList\x12\x17\n\x0fgeneratedSmiles\x18\x01 \x03(\t\x12>\n\nembeddings\x18\x02 \x03(\x0b\x32*.nvidia.cheminformatics.grpc.EmbeddingList\"Q\n\rEmbeddingList\x12\x11\n\tembedding\x18\x01 \x03(\x02\x12\x0b\n\x03\x64im\x18\x02 \x03(\x05\x12\x10\n\x08pad_mask\x18\x03 \x03(\x08\x12\x0e\n\x06tokens\x18\x04 \x03(\x02\"\x1a\n\x07Version\x12\x0f\n\x07version\x18\x01 \x01(\t*:\n\x0fGenerativeModel\x12\x08\n\x04\x43\x44\x44\x44\x10\x00\x12\x0f\n\x0bMegaMolBART\x10\x01\x12\x0c\n\x07MolBART\x10\x90N2\x8c\x04\n\x11GenerativeSampler\x12n\n\x11SmilesToEmbedding\x12+.nvidia.cheminformatics.grpc.GenerativeSpec\x1a*.nvidia.cheminformatics.grpc.EmbeddingList\"\x00\x12j\n\x11\x45mbeddingToSmiles\x12*.nvidia.cheminformatics.grpc.EmbeddingList\x1a\'.nvidia.cheminformatics.grpc.SmilesList\"\x00\x12\x66\n\x0c\x46indSimilars\x12+.nvidia.cheminformatics.grpc.GenerativeSpec\x1a\'.nvidia.cheminformatics.grpc.SmilesList\"\x00\x12\x65\n\x0bInterpolate\x12+.nvidia.cheminformatics.grpc.GenerativeSpec\x1a\'.nvidia.cheminformatics.grpc.SmilesList\"\x00\x12L\n\nGetVersion\x12\x16.google.protobuf.Empty\x1a$.nvidia.cheminformatics.grpc.Version\"\x00\x62\x06proto3')

_GENERATIVEMODEL = DESCRIPTOR.enum_types_by_name['GenerativeModel']
GenerativeModel = enum_type_wrapper.EnumTypeWrapper(_GENERATIVEMODEL)
CDDD = 0
MegaMolBART = 1
MolBART = 10000


_GENERATIVESPEC = DESCRIPTOR.message_types_by_name['GenerativeSpec']
_SMILESLIST = DESCRIPTOR.message_types_by_name['SmilesList']
_EMBEDDINGLIST = DESCRIPTOR.message_types_by_name['EmbeddingList']
_VERSION = DESCRIPTOR.message_types_by_name['Version']
GenerativeSpec = _reflection.GeneratedProtocolMessageType('GenerativeSpec', (_message.Message,), {
  'DESCRIPTOR' : _GENERATIVESPEC,
  '__module__' : 'generativesampler_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.cheminformatics.grpc.GenerativeSpec)
  })
_sym_db.RegisterMessage(GenerativeSpec)

SmilesList = _reflection.GeneratedProtocolMessageType('SmilesList', (_message.Message,), {
  'DESCRIPTOR' : _SMILESLIST,
  '__module__' : 'generativesampler_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.cheminformatics.grpc.SmilesList)
  })
_sym_db.RegisterMessage(SmilesList)

EmbeddingList = _reflection.GeneratedProtocolMessageType('EmbeddingList', (_message.Message,), {
  'DESCRIPTOR' : _EMBEDDINGLIST,
  '__module__' : 'generativesampler_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.cheminformatics.grpc.EmbeddingList)
  })
_sym_db.RegisterMessage(EmbeddingList)

Version = _reflection.GeneratedProtocolMessageType('Version', (_message.Message,), {
  'DESCRIPTOR' : _VERSION,
  '__module__' : 'generativesampler_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.cheminformatics.grpc.Version)
  })
_sym_db.RegisterMessage(Version)

_GENERATIVESAMPLER = DESCRIPTOR.services_by_name['GenerativeSampler']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _GENERATIVEMODEL._serialized_start=583
  _GENERATIVEMODEL._serialized_end=641
  _GENERATIVESPEC._serialized_start=86
  _GENERATIVESPEC._serialized_end=367
  _SMILESLIST._serialized_start=369
  _SMILESLIST._serialized_end=470
  _EMBEDDINGLIST._serialized_start=472
  _EMBEDDINGLIST._serialized_end=553
  _VERSION._serialized_start=555
  _VERSION._serialized_end=581
  _GENERATIVESAMPLER._serialized_start=644
  _GENERATIVESAMPLER._serialized_end=1168
# @@protoc_insertion_point(module_scope)
