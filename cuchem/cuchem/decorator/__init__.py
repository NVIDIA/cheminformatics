from typing import Union

import cudf
import pandas


class BaseMolPropertyDecorator(object):

    def decorate(self,
                 df: Union[cudf.DataFrame, pandas.DataFrame],
                 smiles_cols: int = 0):
        NotImplemented


from .lipinski import LipinskiRuleOfFiveDecorator as LipinskiRuleOfFiveDecorator
from .mol_structure import MolecularStructureDecorator as MolecularStructureDecorator
