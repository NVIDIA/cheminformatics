# Getting started

## Option 1 - Using existing container
This is the easiest method to execute benchmark tests using packaged models.

### Step 1 - Start container
Depending on the model to be benchmarked start container using `launch.sh`
```
./launch.sh dev
```

### Step 2 - Start benchmark test
To start a benchmark task
```
python3 -m chembench.data --config-dir /workspace/cheminformatics/benchmark/config/
```

### TIP - To start a run in a container in daemon mode please execute the following command

```
# For benchmarking MegaMolBART
./launch.sh dev 2 "python3 -m cuchembench --config-dir /workspace/benchmark/scripts/"
```
<hr>
<br>

## Option 2 - Setup a clean environment(container/baremetal)
This is recommended for benchmarking any unsupported generative model using this module. This section explains setting up prerequisites for the benchmark module alone. Additional steps will be required to set up inference capabilities.

### Step 1 - Create Conda environment
Please use the [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-in-silent-mode) at ./conda/env.yml

```
conda env create -f ./conda/env.yml
```

### Step 2 - Install benchmark module
```
pip install .
```

### Step 3 - Install prerequisites for inferencing the model
Please install the software prerequisites and implement a class with following structure for inferencing the model

```python

class SomeModelInferenceWrapper():

    def __init__(self) -> None:

    def is_ready(self, timeout: int = 10) -> bool:

    def smiles_to_embedding(self, smiles: str, padding: int,
                            scaled_radius=None, num_requested: int = 10,
                            sanitize=True):

    def embedding_to_smiles(self, embedding, dim: int, pad_mask):

    def find_similars_smiles(self, smiles: str, num_requested: int = 10,
                             scaled_radius=1, force_unique=False,
                             sanitize=True):

    def interpolate_smiles(self, smiles: List, num_points: int = 10,
                           scaled_radius=None, force_unique=False,
                           sanitize=True):
```

<hr>
<br>
<br>

# Configuration
please refer to `config` directory