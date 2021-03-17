import cudf
import pandas
import base64
import logging
from typing import Union

from rdkit import Chem
from rdkit.Chem import Draw

from nvidia.cheminformatics.decorator import BaseMolPropertyDecorator

logger = logging.getLogger(__name__)


class MolecularStructureDecorator(BaseMolPropertyDecorator):

    def decorate(self,
                 df:Union[cudf.DataFrame, pandas.DataFrame],
                 smile_cols:int=0):

        mol_struct = []
        for idx in range(df.shape[0]):

            smiles = df.iat[idx, smile_cols]
            try:
                m = Chem.MolFromSmiles(smiles)
                drawer = Draw.rdMolDraw2D.MolDraw2DCairo(500, 125)
                drawer.SetFontSize(1.0)
                drawer.DrawMolecule(m)
                drawer.FinishDrawing()

                img_binary = "data:image/png;base64," + \
                    base64.b64encode(drawer.GetDrawingText()).decode("utf-8")

                mol_struct.append({'value': img_binary, 'level': 'info'})
            except Exception as ex:
                logger.exception(ex)
                mol_struct.append(
                    {'value': 'Error interpreing SMILES using RDKIT',
                     'level': 'error'})

        df['Chemical Structure'] = mol_struct

        return df
