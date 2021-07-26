# What Is Mega-MolBART gRPC service?
[Mega-MolBART](https://ngc.nvidia.com/models/nvidia/clara:megamolbart) is a model trained on SMILES string and this container deploys Mega-MolBART model for infrencing. The container bring up a gRPC service available in port 500051.

# Getting Started
**Mega-MolBART gRPC service** is currently used with [Cheminformatics Demo](https://ngc.nvidia.com/containers/nvidia/clara:cheminformatics_demo) application (also available as a container.) Please refer [Cheminformatics](https://ngc.nvidia.com/resources/nvidia/clara:cheminformatics) resource for bringing up the complete suite.

Please following these steps to start **Mega-MolBART gRPC service** in stand-alone mode.

- Download [Mega-MolBART](https://ngc.nvidia.com/models/nvidia/clara:megamolbart) model or execute the following command to do so

 ```
 ngc registry model download-version "nvidia/clara/megamolbart:0.1"
 ```

- Start an instance of the Docker image using the following command:

 ```
 # E.g. for Docker version 19.03 or later
 docker run \
 --gpus all \
 --rm \
 -v $(pwd)/megamolbart_v0.1/:/models/megamolbart \
 nvcr.io/nvidia/clara/megamolbart:0.1

 # E.g. for Docker version 19.02 or later
 docker run \
 --runtime nvidia \
 --rm \
 -v $(pwd)/megamolbart_v0.1/:/models/megamolbart \
 nvcr.io/nvidia/clara/megamolbart:0.1
 ```

# Interface
**Protobuf** definition([generativesampler.proto](https://ngc.nvidia.com/resources/nvidia/clara:cheminformatics/files?version=0.1#)) of this service is available in [Cheminformatics](https://ngc.nvidia.com/resources/nvidia/clara:cheminformatics) resources.

Please use the following command to generate language specific stubs.


```
python -m grpc_tools.protoc -I./grpc/ \
 --<>_out=generated \
 --experimental_allow_proto3_optional \
 --grpc_python_out=generated \
 ./grpc/generativesampler.proto
```


The service has three functions:
- **SmilesToEmbedding**: Returns the latent space embedding of an input SMILES string.
- **FindSimilars**: Generates SMILES using the model by probing the vicinity of input SMILES's position in latent space.
- **Interpolate**: Generates SMILES using the model by probing the latent space between two input SMILES.
- **GetIteration**: Returns the checkpoint iteration version of the deployed model.


# Current usage
- [**Cheminformatics**](https://ngc.nvidia.com/resources/nvidia/clara:cheminformatics)