import logging
from typing import Union

import cudf
import pandas
from numpy.core.numeric import NaN
from cuchem.decorator import BaseMolPropertyDecorator
from rdkit import Chem
from rdkit.Chem import QED, Descriptors, Lipinski

logger = logging.getLogger(__name__)


class LipinskiRuleOfFiveDecorator(BaseMolPropertyDecorator):
    MAX_LOGP = 3
    MAX_MOL_WT = 300
    MAX_H_DONORS = 6
    MAX_H_ACCEPTORS = 6
    MAX_ROTATABLE_BONDS = 3
    MAX_QED = 3

    def decorate(self,
                 df: Union[cudf.DataFrame, pandas.DataFrame],
                 smile_cols: int = 0):

        mol_wt = []
        mol_logp = []
        hdonors = []
        hacceptors = []
        rotatable_bonds = []
        qeds = []
        invalid = []

        for idx in range(df.shape[0]):

            smiles = df.iat[idx, smile_cols]
            m = Chem.MolFromSmiles(smiles)

            if m is None:
                logger.info(f'{idx}: Could not make a Mol from {smiles}')
                invalid.append(True)
                mol_logp.append({'value': '-', 'level': 'info'})
                mol_wt.append({'value': '-', 'level': 'info'})
                hdonors.append({'value': '-', 'level': 'info'})
                hacceptors.append({'value': '-', 'level': 'info'})
                rotatable_bonds.append({'value': '-', 'level': 'info'})
                qeds.append({'value': '-', 'level': 'info'})
                continue
            else: 
                invalid.append(False)
            try:
                logp = Descriptors.MolLogP(m)
                mol_logp.append({'value': round(logp, 2),
                                 'level': 'info' if logp < LipinskiRuleOfFiveDecorator.MAX_LOGP else 'error'})
            except Exception as ex:
                logger.exception(ex)
                mol_logp.append({'value': '-', 'level': 'info'})

            try:
                wt = Descriptors.MolWt(m)
                mol_wt.append({'value': round(wt, 2),
                               'level': 'info' if wt < LipinskiRuleOfFiveDecorator.MAX_MOL_WT else 'error'})
            except Exception as ex:
                logger.exception(ex)
                mol_wt.append({'value': '-', 'level': 'info'})

            try:
                hdonor = Lipinski.NumHDonors(m)
                hdonors.append({'value': hdonor,
                                'level': 'info' if hdonor < LipinskiRuleOfFiveDecorator.MAX_H_DONORS else 'error'})
            except Exception as ex:
                logger.exception(ex)
                hdonors.append({'value': '-', 'level': 'info'})

            try:
                hacceptor = Lipinski.NumHAcceptors(m)
                hacceptors.append(
                    {'value': hacceptor,
                     'level': 'info' if hacceptor < LipinskiRuleOfFiveDecorator.MAX_H_DONORS else 'error'})
            except Exception as ex:
                logger.exception(ex)
                hacceptors.append({'value': '-', 'level': 'info'})

            try:
                rotatable_bond = Lipinski.NumRotatableBonds(m)
                rotatable_bonds.append(
                    {'value': rotatable_bond,
                     'level': 'info' if rotatable_bond < LipinskiRuleOfFiveDecorator.MAX_ROTATABLE_BONDS else 'error'})
            except Exception as ex:
                logger.exception(ex)
                rotatable_bonds.append({'value': '-', 'level': 'info'})

            try:
                qed = QED.qed(m)
                qeds.append({'value': round(qed, 4),
                             'level': 'info' if qed < LipinskiRuleOfFiveDecorator.MAX_QED else 'error'})
            except Exception as ex:
                logger.exception(ex)
                qeds.append({'value': '-', 'level': 'info'})

        df['Molecular Weight'] = mol_wt
        df['LogP'] = mol_logp
        df['H-Bond Donors'] = hdonors
        df['H-Bond Acceptors'] = hacceptors
        df['Rotatable Bonds'] = rotatable_bonds
        df['QED'] = qeds
        # TODO: this may be redundant as chemvisualize seems to be handling such invalid molecules
        df['invalid'] = invalid

        return df
