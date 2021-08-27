# gpuci-build-environment
Chemformatics build environment for gpuCI

## How to extend gpuCI Dockerfiles

* Use the same build args supplied to gpuCI Dockerfiles in custom image
* Inherit from base gpuCI Docker image

```
gpuci/miniconda-cuda:${CUDA_VER}-${IMAGE_TYPE}-${LINUX_VERSION}
```
