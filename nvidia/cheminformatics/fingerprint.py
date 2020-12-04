import logging
from abc import ABC
from enum import Enum
from rdkit import Chem
from rdkit.Chem import AllChem


logger = logging.getLogger(__name__)

class TransformationDefaults(Enum):
    morgan_fingerprint = {'radius':2, 'nBits':512}
    MorganFingerprint = {'radius':2, 'nBits':512}


class BaseTransformation(ABC):
    def __init__(self, **kwargs):
        self.name = None
        self.kwargs = None
        self.func = None

    def transform(self, data):
        return NotImplemented

    def transform_many(self, data):
        return map(self.transform, data)

    def __len__(self):
        return NotImplemented
    

class MorganFingerprint(BaseTransformation):

    def __init__(self, **kwargs):
        self.name = __class__.__name__.split('.')[-1]
        self.kwargs = TransformationDefaults[self.name].value
        self.kwargs.update(kwargs)
        self.func = AllChem.GetMorganFingerprintAsBitVect

    def transform(self, data):
        m = Chem.MolFromSmiles(data)
        fp = self.func(m, **self.kwargs)
        return ', '.join(list(fp.ToBitString()))

    def __len__(self):
        return self.kwargs['nBits']
        

### DEPRECATED ###
def morgan_fingerprint(smiles, radius=2, nBits=512):
    m = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)

    return ', '.join(list(fp.ToBitString()))


def get_transformation_function_kwargs(transformation_function=morgan_fingerprint, enum_class=TransformationDefaults, **transformation_kwargs):
    """Get arguments for transformation function"""

    if not isinstance(transformation_function, str):
       transformation_function = transformation_function.__name__

    tf_default_dict = enum_class[transformation_function].value
    for key in tf_default_dict:
        transformation_kwargs[key] = tf_default_dict[key]

    # TODO functools.partial does not seem compatible with Dask. Is there a workaround?
    # return partial(transformation_function, **transformation_kwargs)
    return transformation_kwargs

