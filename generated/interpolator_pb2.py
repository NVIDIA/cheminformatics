# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: interpolator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='interpolator.proto',
  package='nvidia.cheminformatics.grpc',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x12interpolator.proto\x12\x1bnvidia.cheminformatics.grpc\"2\n\x11InterpolationSpec\x12\r\n\x05model\x18\x01 \x01(\t\x12\x0e\n\x06smiles\x18\x02 \x03(\t\"%\n\nSmilesList\x12\x17\n\x0fgeneratedSmiles\x18\x01 \x03(\t2x\n\x0cInterpolator\x12h\n\x0bInterpolate\x12..nvidia.cheminformatics.grpc.InterpolationSpec\x1a\'.nvidia.cheminformatics.grpc.SmilesList\"\x00\x62\x06proto3'
)




_INTERPOLATIONSPEC = _descriptor.Descriptor(
  name='InterpolationSpec',
  full_name='nvidia.cheminformatics.grpc.InterpolationSpec',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='nvidia.cheminformatics.grpc.InterpolationSpec.model', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='smiles', full_name='nvidia.cheminformatics.grpc.InterpolationSpec.smiles', index=1,
      number=2, type=9, cpp_type=9, label=3,
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
  serialized_start=51,
  serialized_end=101,
)


_SMILESLIST = _descriptor.Descriptor(
  name='SmilesList',
  full_name='nvidia.cheminformatics.grpc.SmilesList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='generatedSmiles', full_name='nvidia.cheminformatics.grpc.SmilesList.generatedSmiles', index=0,
      number=1, type=9, cpp_type=9, label=3,
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
  serialized_start=103,
  serialized_end=140,
)

DESCRIPTOR.message_types_by_name['InterpolationSpec'] = _INTERPOLATIONSPEC
DESCRIPTOR.message_types_by_name['SmilesList'] = _SMILESLIST
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InterpolationSpec = _reflection.GeneratedProtocolMessageType('InterpolationSpec', (_message.Message,), {
  'DESCRIPTOR' : _INTERPOLATIONSPEC,
  '__module__' : 'interpolator_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.cheminformatics.grpc.InterpolationSpec)
  })
_sym_db.RegisterMessage(InterpolationSpec)

SmilesList = _reflection.GeneratedProtocolMessageType('SmilesList', (_message.Message,), {
  'DESCRIPTOR' : _SMILESLIST,
  '__module__' : 'interpolator_pb2'
  # @@protoc_insertion_point(class_scope:nvidia.cheminformatics.grpc.SmilesList)
  })
_sym_db.RegisterMessage(SmilesList)



_INTERPOLATOR = _descriptor.ServiceDescriptor(
  name='Interpolator',
  full_name='nvidia.cheminformatics.grpc.Interpolator',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=142,
  serialized_end=262,
  methods=[
  _descriptor.MethodDescriptor(
    name='Interpolate',
    full_name='nvidia.cheminformatics.grpc.Interpolator.Interpolate',
    index=0,
    containing_service=None,
    input_type=_INTERPOLATIONSPEC,
    output_type=_SMILESLIST,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_INTERPOLATOR)

DESCRIPTOR.services_by_name['Interpolator'] = _INTERPOLATOR

# @@protoc_insertion_point(module_scope)
