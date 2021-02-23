# Setup

To create a conda virtual environment for use outside of the container, do:

```
conda env create -n cheminformatics -f cuchem_rapids_0.17.yml
```
Define `DATA_MOUNT_PATH` in `.cheminf_local_environment`. For example:
```
DATA_MOUNT_PATH=/tmp/db
```
Start dash:
```
python3 startdash.py analyze
```
