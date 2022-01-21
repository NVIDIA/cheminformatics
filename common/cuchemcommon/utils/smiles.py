import numba
from numba.types import bool_, string, int64

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


@numba.jit(bool_[:, :](string[:], int64, int64), forceobj=True, parallel=True)
def calculate_morgan_fingerprint(smiles, radius, nbits):
    '''
    Calculate Morgan fingerprint for a series of SMILES strings.

    Parameters
    ----------
    smiles : array_like
        The array of SMILES strings to be processed.
    radius : int
        The radius of the Morgan fingerprint.
    nbits : int
        The number of bits in the Morgan fingerprint.
    '''
    fingerprints = np.empty((smiles.shape[0], nbits), dtype=np.bool_)
    for i in range(smiles.shape[0]):
        mol = Chem.MolFromSmiles(smiles[i])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
            fp = np.fromstring(fp.ToBitString(), 'u1') - ord('0')
        else:
            fp = np.zeros(nbits, dtype=bool)

        fingerprints[i] = fp
    return fingerprints


def validate_smiles(smiles: str,
                    canonicalize=False,
                    return_fingerprint=False,
                    radius=2,
                    nbits=512):
        '''
        Validate SMILES string.

        Parameters
        ----------
        smiles : str
            SMILES string to validate.
        canonicalize : bool, optional
            If True, canonicalize SMILES string.
        return_fingerprint : bool, optional
            If True, return Morgan fingerprint of SMILES string.
        radius : int, optional
            The radius of the Morgan fingerprint.
        nbits : int, optional
            The number of bits in the Morgan fingerprint.
        '''
        valid_smiles = smiles
        is_valid = True

        mol = Chem.MolFromSmiles(smiles)
        fp = None
        if mol:
            if canonicalize:
                valid_smiles = Chem.MolToSmiles(mol, canonical=True)
            else:
                valid_smiles = smiles

            if return_fingerprint:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
                fp = np.fromstring(fp.ToBitString(), 'u1') - ord('0')
        else:
            is_valid = False
            if return_fingerprint:
                fp = np.zeros(nbits, dtype=bool)

        if fp is not None:
            return valid_smiles, is_valid, fp
        else:
            return valid_smiles, is_valid

