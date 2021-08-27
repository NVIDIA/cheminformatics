#!/usr/bin/env python3

import pandas as pd
import torch

from megamolbart.inference import MegaMolBART


if __name__ == '__main__':

    num_interp = 3
    smiles1 = 'CC(=O)Nc1ccc(O)cc1'
    smiles2 = 'CC(=O)Oc1ccccc1C(=O)O'

    with torch.no_grad():
        wf = MegaMolBART()

        # Test each of the major functions
        mols_list_1, _ = wf.interpolate_molecules(smiles1, smiles2, num_interp, wf.tokenizer)
        assert len(mols_list_1) == num_interp
        assert isinstance(mols_list_1, list)
        assert isinstance(mols_list_1[0], str)

        mols_list_2, _, _ = wf.find_similars_smiles_list(smiles1, num_requested=num_interp)
        assert len(mols_list_2) == num_interp + 1
        assert isinstance(mols_list_2, list)
        assert isinstance(mols_list_2[0], str)

        mols_df_1 = wf.find_similars_smiles(smiles2, num_interp)
        assert len(mols_df_1) == num_interp + 1
        assert isinstance(mols_df_1, pd.DataFrame)
        assert isinstance(mols_df_1.loc[1, 'SMILES'], str)

        mols_df_2 = wf.interpolate_smiles([smiles1, smiles2], num_interp)
        assert len(mols_df_2) == num_interp + 2
        assert isinstance(mols_df_2, pd.DataFrame)
        assert isinstance(mols_df_2.loc[1, 'SMILES'], str)

