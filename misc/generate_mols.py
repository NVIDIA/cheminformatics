import logging
import pandas
from rdkit import Chem

from nvidia.cheminformatics.wf.generative import MolBART, Cddd
from tests.utils import _create_context
from nvidia.cheminformatics.decorator import LipinskiRuleOfFiveDecorator, MolecularStructureDecorator, lipinski
import dask.dataframe as dd
import multiprocessing

logger = logging.getLogger(__name__)

MAX_LOGP = 3
MAX_MOL_WT = 300
MAX_H_DONORS = 6
MAX_H_ACCEPTORS = 6
MAX_ROTATABLE_BONDS = 3
MAX_QED = 3


def score_molecule(smiles):
    lipinski_score = 0
    qed = LipinskiRuleOfFiveDecorator.MAX_QED + 1

    try:
        m = Chem.MolFromSmiles(smiles)
        logp = Chem.Descriptors.MolLogP(m)
        lipinski_score += 1 if logp < LipinskiRuleOfFiveDecorator.MAX_LOGP else 0

        wt = Chem.Descriptors.MolWt(m)
        lipinski_score += 1 if wt < LipinskiRuleOfFiveDecorator.MAX_MOL_WT else 0

        hdonor = Chem.Lipinski.NumHDonors(m)
        lipinski_score += 1 if hdonor < LipinskiRuleOfFiveDecorator.MAX_H_DONORS else 0

        hacceptor = Chem.Lipinski.NumHAcceptors(m)
        lipinski_score += 1 if hacceptor < LipinskiRuleOfFiveDecorator.MAX_H_DONORS else 0

        rotatable_bond = Chem.Lipinski.NumRotatableBonds(m)
        lipinski_score += 1 if rotatable_bond < LipinskiRuleOfFiveDecorator.MAX_ROTATABLE_BONDS else 0

        qed = Chem.QED.qed(m)
    except Exception as ex:
        lipinski_score = 0

    return lipinski_score, qed


def generate():
    wf = MolBART()
    num_to_add = 21

    def _generate(data):
        smiles = data['canonical_smiles']
        lipinski_score, qed = score_molecule(smiles)

        num_to_generate = 40
        lipinski_scores = []
        qed_scores = []
        valid_list = []

        try:
            if lipinski_score >= 3 and qed <= LipinskiRuleOfFiveDecorator.MAX_QED:
                generated_list = wf.find_similars_smiles_list(smiles,
                                                            num_requested=num_to_generate,
                                                            radius=0.0001)
                for new_smiles in generated_list:
                    lipinski_score, qed = score_molecule(new_smiles)

                    if lipinski_score >= 3 and qed <= LipinskiRuleOfFiveDecorator.MAX_QED:
                        valid_list.append(new_smiles)
                        lipinski_scores.append(lipinski_score)
                        qed_scores.append(qed)

                    if len(valid_list) >= num_to_add:
                        break
        except Exception as ex:
            pass

        valid_list += [''] * ((num_to_add) - len(valid_list))
        lipinski_scores += [0] * (num_to_add - len(lipinski_scores))
        qed_scores += [0] * (num_to_add - len(qed_scores))

        return valid_list + lipinski_scores + qed_scores


    data = pandas.read_csv('/workspace/tests/data/benchmark_approved_drugs.csv')

    prop_meta = dict(zip([ i for i in range(num_to_add)],
                        [pandas.Series([], dtype='object') for i in range(num_to_add)]))
    prop_meta.update(dict(zip([ num_to_add + i for i in range(num_to_add)],
                        [pandas.Series([], dtype='int8') for i in range(num_to_add)])))
    prop_meta.update(dict(zip([ (2 * num_to_add) + i for i in range(num_to_add)],
                        [pandas.Series([], dtype='float64') for i in range(num_to_add)])))
    meta_df = pandas.DataFrame(prop_meta)

    context = _create_context()
    ddf = dd.from_pandas(data, npartitions = 4 * multiprocessing.cpu_count())
    ddf = ddf.map_partitions(
        lambda dframe: dframe.apply(_generate, result_type='expand', axis=1),
        meta=meta_df)
    ddf = ddf.compute(scheduler='processes')
    ddf.to_csv("/workspace/similar_mols.csv")


if __name__ == '__main__':
    generate()
