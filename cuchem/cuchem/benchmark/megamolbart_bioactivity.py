#!/usr/bin/env python3

from cuchem.wf.generative import MegatronMolBART
from cuchem.benchmark.datasets.bioactivity import (ExCAPEBioactivity, ExCAPEFingerprints)

from cuchem.benchmark.data import BioActivityEmbeddingData
from cuchem.benchmark.metrics.embeddings import Modelability
from cuml.ensemble.randomforestregressor import RandomForestRegressor
from copy import deepcopy

#### Dataset lists

smiles_dataset = [ExCAPEBioactivity()]
fingerprint_dataset = [ExCAPEFingerprints()]

#### Model

inferrer = MegatronMolBART()
estimator = RandomForestRegressor(accuracy_metric='mse', random_state=0)
param_dict = {'n_estimators': [10, 15]}
model_dict = {'random forest': [estimator, param_dict]}

#### Prep datasets
smiles_dataset = smiles_dataset[0] # Keep code as similar to other version as possible
fingerprint_dataset = fingerprint_dataset[0]

#### Metrics

emb_cache = BioActivityEmbeddingData()
metric = Modelability(inferrer, emb_cache, smiles_dataset)

smiles_dataset.load()
fingerprint_dataset.load()

groups = zip(smiles_dataset.data.groupby(level=0), smiles_dataset.properties.groupby(level=0), fingerprint_dataset.data.groupby(level=0))

#### Calculate metrics

smiles_ = deepcopy(smiles_dataset)
fingerp_ = deepcopy(fingerprint_dataset)

for (label, sm_), (_, prop_), (_, fp_) in groups:
    smiles_.data = sm_ # This is a hack to reduce dataset size for testing
    smiles_.properties = prop_
    fingerp_.data = fp_

    result = metric.calculate(smiles_dataset=smiles_, # Filtered dataset was pruned on the drive
                              fingerprint_dataset=fingerp_,
                              properties_dataset=smiles_.properties,
                              estimator=estimator,
                              param_dict=param_dict,
                              top_k=1,
                              num_samples=1,
                              radius=1)
    print(label, result)

