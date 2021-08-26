# gpuci-build-environment
Clara Genomics build environment for gpuCI

## How to extend gpuCI Dockerfiles

* Use the same build args supplied to gpuCI Dockerfiles in custom image
* Inherit from base gpuCI Docker image

```
gpuci/rapidsai-base:cuda${CUDA_VERSION}-${LINUX_VERSION}-gcc${CC_VERSION}-py${PYTHON_VERSION}
```
