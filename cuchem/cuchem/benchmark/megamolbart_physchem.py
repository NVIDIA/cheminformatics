#!/usr/bin/env python3

from cuchem.wf.generative import MegatronMolBART
from cuchem.benchmark.datasets.molecules import (ChEMBLApprovedDrugsPhyschem,
                                                 MoleculeNetESOLPhyschem,
                                                 MoleculeNetFreeSolvPhyschem,
                                                 MoleculeNetLipophilicityPhyschem )

from cuchem.benchmark.datasets.fingerprints import (ChEMBLApprovedDrugsFingerprints,
                                                    MoleculeNetESOLFingerprints,
                                                    MoleculeNetFreeSolvFingerprints,
                                                    MoleculeNetLipophilicityFingerprints )

from cuchem.benchmark.data import PhysChemEmbeddingData
from cuchem.benchmark.metrics.embeddings import Modelability
from cuml.ensemble.randomforestregressor import RandomForestRegressor

#### Dataset lists

physchem_dataset_list = [MoleculeNetESOLPhyschem(), MoleculeNetFreeSolvPhyschem(), MoleculeNetLipophilicityPhyschem()]
physchem_fingerprint_list = [MoleculeNetESOLFingerprints(), MoleculeNetFreeSolvFingerprints(), MoleculeNetLipophilicityFingerprints()]

#### Model

inferrer = MegatronMolBART()
estimator = RandomForestRegressor(accuracy_metric='mse', random_state=0)
param_dict = {'n_estimators': [10, 15]}
model_dict = {'random forest': [estimator, param_dict]}

#### Metrics

emb = PhysChemEmbeddingData()
metric = Modelability(inferrer, emb)

#### Prep datasets

smiles_dataset = physchem_dataset_list
fingerprint_dataset = physchem_fingerprint_list

for x in smiles_dataset:
    x.load()
for x in fingerprint_dataset:
    x.load()

groups = zip([x.name for x in smiles_dataset],
             smiles_dataset,
             fingerprint_dataset)

#### Calculate metrics

n_data = 200
for (label, smiles_, fp_) in groups:

    smiles_.data = smiles_.data.iloc[:n_data] # TODO for testing
    smiles_.properties = smiles_.properties.iloc[:n_data]  # TODO for testing
    fp_.data = fp_.data.iloc[:n_data]  # TODO for testing

    print(label, smiles_.data.head(n=1), smiles_.properties.head(n=1), fp_.data.head(n=1))
    # result = metric.calculate(smiles_dataset=smiles_,
    #                           fingerprint_dataset=fp_,
    #                           properties_dataset=smiles_.properties,
    #                           estimator=estimator,
    #                           param_dict=param_dict,
    #                           top_k=1,
    #                           num_samples=1,
    #                           radius=1)
    # print(label, result)

print('Done')
