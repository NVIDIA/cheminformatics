import logging
from nvidia.cheminformatics.wf.generative import MolBART, Cddd

from tests.utils import _create_context

from nvidia.cheminformatics.decorator import LipinskiRuleOfFiveDecorator, MolecularStructureDecorator


logger = logging.getLogger(__name__)


def interpolation(wf):
    """
    Verify fetching data from chemblDB when the input is a pandas df.
    """
    num_points = 10

    smiles = ['CHEMBL10454', 'CHEMBL10469']
    genreated_df = wf.interpolate_from_id(smiles, num_points=num_points)

    logger.info(genreated_df.head(10))

    genreated_df = MolecularStructureDecorator().decorate(genreated_df)
    genreated_df = LipinskiRuleOfFiveDecorator().decorate(genreated_df)

    assert genreated_df.shape[0] > 2


def test_molbart_interpolation():
    context = _create_context()
    wf = MolBART()
    interpolation(wf)


def test_cddd_interpolation():
    context = _create_context()
    wf = Cddd()
    interpolation(wf)


def test_cddd_similar_smiles():
    context = _create_context()
    wf = Cddd()
    num_to_generate = 3

    generated_smiles = wf.find_similars_smiles('CC(=O)Nc1ccc(O)cc1',
                                               num_requested=num_to_generate,
                                               radius=0.75)
    logger.info(generated_smiles)

    assert len(generated_smiles) == num_to_generate + 1


def test_molbart_similar_smiles():
    context = _create_context()
    wf = MolBART()
    num_to_generate = 3

    generated_smiles = wf.find_similars_smiles('CC(=O)Nc1ccc(O)cc1',
                                               num_requested=num_to_generate,
                                               radius=0.0001)
    logger.info('# of generated SMILES %s', len(generated_smiles))
