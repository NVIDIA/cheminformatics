# Benchmarking -- MegaMolBART

This directory contains benchmarks of MegaMolBART. Only three metrics were completed -- Validity, Uniqueness, and Nearest Neighbor Correlation

# Statistics

Date: 2021/06/08  
Dataset: benchmark_ChEMBL_random_sampled_drugs  
Model Attention Heads: 8  
Model Layers: 4  
Model Hidden Size: 256  
Model Parameters: 10M  
Trained: Draco-rno, 4 nodes x 8 GPUs, data parallel only, ~24 hours  
Iteration(s) Benchmarked: 10000, 50000, 350000, 610000  
