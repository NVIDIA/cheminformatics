# Getting started
- Start MolBART container
  ```
  ./launch.sh dev 2
  ```

- To start a benchmark task
  ```
  python3 -m cuchembm --config-dir /workspace/benchmark/scripts/
  ```

# Configuration
Configuration file for manipulating benchmark run is located at benchmark/scripts/


# Results
Benchmark test results can be found at /data/benchmark_output folder.


# Create env
conda env update -n base --file env.yml
export PYTHONPATH=$PYTHONPATH:/workspace/common/generated
export PYTHONPATH=$PYTHONPATH:/workspace/megamolbart
export PYTHONPATH=$PYTHONPATH:/workspace/common/
