# Cheminformatics Clustering
---
### Introduction
A central task in drug discovery is searching, screening, and organizing large chemical databases.
Here, we implement clustering on molecular similarity.
We compute Morgan circular fingerprints, cluster using k-means, and perform dimensionality reduction using UMAP.
Distributed GPU-accelerated algorithms enable real-time interactive exploration of chemical space.

![screenshot](screenshot.jpg "Screenshot of cheminformatics dashboard")

### Preparing your environment (optional)
A launch script, `launch.sh`, is provided to perform all tasks. 

#### Customization

The environment can be customized for control of the container, to create your own repo, or to store the data in a custom location. However, if this is not needed, 
skip to [Getting Started](#getting-started) to use the defaults.

To customize your local environment, edit the appropriate section of `launch.sh` or provide a `~/.cheminf_local_environment` file with the following information below.  
To generate a template for `.cheminf_local_environment`, just run `./launch.sh` with no arguments.  
If `.cheminf_local_environment` does not exist, then a template will be written for you.

```
CONT=gitlab-master.nvidia.com:5005/<path_to_your_repo>
REGISTRY=gitlab-master.nvidia.com:5005
REGISTRY_USER='$oauthtoken'
REGISTRY_ACCESS_TOKEN=$(cat <path_to_your_access_token>)
PROJECT_PATH=/path/to/local/repo/dir
JUPYTER_PORT=8888
DATA_PATH=/path/to/scratch/space
```

### Getting Started
Once your environment is setup, the following commands should be all you need.

Build your container:

```
./launch.sh build
```

Download the ChEMBL database:
```
./launch.sh dbSetup
```

Launch the dash interactive ChEMBL exploration tool:

```
./launch.sh dash
```

Navigate a browser to:

```
https://0.0.0.0:5000
```

### Acceleration

The latest benchmarks reside in the [benchmarks](benchmarks) directory. The files are available in [markdown](benchmarks/benchmark.md) and [Excel](benchmarks/benchmark.xlsx). A [plot](benchmarks/benchmark.png) is also available.



