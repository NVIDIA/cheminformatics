import cudf
import pandas
from typing import Union


class BaseMolPropertyDecorator(object):

    def decorate(self,
                 df:Union[cudf.DataFrame, pandas.DataFrame],
                 smile_cols:int=0):
        NotImplemented


from .lipinski import LipinskiRuleOfFiveDecorator as LipinskiRuleOfFiveDecorator
from .mol_structure import MolecularStructureDecorator as MolecularStructureDecorator
