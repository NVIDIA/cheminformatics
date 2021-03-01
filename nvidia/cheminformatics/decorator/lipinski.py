import cudf
from numpy.core.numeric import NaN
import pandas
from typing import Union

from rdkit import Chem

from nvidia.cheminformatics.decorator import BaseMolPropertyDecorator


class LipinskiRuleOfFiveDecorator(BaseMolPropertyDecorator):

    MAX_LOGP = 3
    MAX_MOL_WT = 300
    MAX_H_DONORS = 6
    MAX_H_ACCEPTORS = 6
    MAX_ROTATABLE_BONDS = 3
    MAX_QED = 3

    def decorate(self,
                df:Union[cudf.DataFrame, pandas.DataFrame],
                smile_cols:int=0):

        mol_wt = []
        mol_logp = []
        hdonors = []
        hacceptors = []
        rotatable_bonds = []
        qeds = []

        for idx in range(df.shape[0]):

            smiles = df.iat[idx, smile_cols]
            m = Chem.MolFromSmiles(smiles)

            try:
                logp = Chem.Descriptors.MolLogP(m)
                mol_logp.append({'value': logp,
                                 'level': 'info' if logp < LipinskiRuleOfFiveDecorator.MAX_LOGP else 'warning'})
            except Exception as ex:
                mol_logp.append({'value': NaN, 'level': 'info'})

            try:
                wt = Chem.Descriptors.MolWt(m)
                mol_wt.append({'value': wt,
                               'level': 'info' if wt < LipinskiRuleOfFiveDecorator.MAX_MOL_WT else 'warning'})
            except Exception as ex:
                mol_wt.append({'value': NaN, 'level': 'info'})

            try:
                hdonor = Chem.Lipinski.NumHDonors(m)
                hdonors.append({'value': hdonor,
                                'level': 'info' if hdonor < LipinskiRuleOfFiveDecorator.MAX_H_DONORS else 'warning'})
            except Exception as ex:
                hdonors.append({'value': NaN, 'level': 'info'})

            try:
                hacceptor = Chem.Lipinski.NumHAcceptors(m)
                hacceptors.append({'value': hacceptor,
                                   'level': 'info' if hacceptor < LipinskiRuleOfFiveDecorator.MAX_H_DONORS else 'warning'})
            except Exception as ex:
                hacceptors.append({'value': NaN, 'level': 'info'})

            try:
                rotatable_bond = Chem.Lipinski.NumRotatableBonds(m)
                rotatable_bonds.append({'value': rotatable_bond,
                                        'level': 'info' if rotatable_bond < LipinskiRuleOfFiveDecorator.MAX_ROTATABLE_BONDS else 'warning'})
            except Exception as ex:
                rotatable_bonds.append({'value': NaN, 'level': 'info'})

            try:
                qed = Chem.Chem.QED.qed(m)
                qeds.append({'value': qed,
                             'level': 'info' if qed < LipinskiRuleOfFiveDecorator.MAX_QED else 'warning'})
            except Exception as ex:
                qeds.append({'value': NaN, 'level': 'info'})

        df['mol_wt'] = mol_wt
        df['logp'] = mol_logp
        df['hdonors'] = hdonors
        df['hacceptors'] = hacceptors
        df['rotatable_bonds'] = rotatable_bonds
        df['qeds'] = qeds

        return df
