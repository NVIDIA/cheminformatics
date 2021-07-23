Overview
-------------------------------------------------------------------------------
**Cheminformatics** resource is a collection of scripts and configuration to bringing up Cheminformatices application. The application itself requires two images available at NGC:
 - [Cheminformatics](https://ngc.nvidia.com/containers/nv-drug-discovery-dev:cheminformatics_demo)
 - [Mega-MolBART gRPC Service](https://ngc.nvidia.com/containers/nv-drug-discovery-dev:megamolbart)

Please refer documentation at [Cheminformatics](https://ngc.nvidia.com/containers/nv-drug-discovery-dev:cheminformatics_demo) image for features.



Setup
-------------------------------------------------------------------------------

## Pre-reqs
 - Linux OS
 - Pascal, Volta, Turing, or an NVIDIA Ampere architecture-based GPU.
 - Nvidia Driver
 - Docker
 - docker-compose


## Steps
  - Please download the file and extract or execute the following command to get the resource.
    ```
    ngc registry resource download-version "nv-drug-discovery-dev/cheminformatics:0.1"
    ```

  - Execute the following commands to start the tool.
    ```
    chmod +x launch
    ./launch
    # .env file will be created on first execution.
    # Please enter a valid ngc api key in .env (REGISTRY_ACCESS_TOKEN) and re-run the command.
    # Refer NGC -> Setup -> Generate API KEY
    ```

Initial setup will be slow because the tool will download ChEMBLE Database(v27) and the models used in the application.


Please use .env file to change SUBNET and IP's used by the container. docker-compose is used to orchestrate the containers.


## Quick Start Guide
See the [tutorial](https://github.com/NVIDIA/cheminformatics/blob/master/tutorial/Tutorial.md) for an example walkthrough.
