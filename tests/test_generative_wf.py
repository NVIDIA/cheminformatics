import logging
from nvidia.cheminformatics.wf.generative import MolBART, Cddd

from tests.utils import _create_context

from nvidia.cheminformatics.decorator import LipinskiRuleOfFiveDecorator, MolecularStructureDecorator


logger = logging.getLogger(__name__)


def interpolation(wf, num_points = 20, force_unique=False):
    """
    Verify fetching data from chemblDB when the input is a pandas df.
    """

    smiles = ['CHEMBL6328', 'CHEMBL415286']
    # smiles = ['CHEMBL10454', 'CHEMBL10469']
    genreated_df = wf.interpolate_from_id(smiles,
                                          num_points=num_points,
                                          force_unique=force_unique)

    genreated_df = MolecularStructureDecorator().decorate(genreated_df)
    genreated_df = LipinskiRuleOfFiveDecorator().decorate(genreated_df)
    logger.info(genreated_df.shape)
    return genreated_df


def test_cddd_interpolation():
    num_points = 20

    context = _create_context()
    wf = Cddd()
    interp = interpolation(wf,
                           num_points=num_points,
                           force_unique=True)
    logger.info(interp)
    logger.info(interp.columns)

    assert interp.shape[0] == num_points + 2


def test_cddd_similar_smiles():
    context = _create_context()
    wf = Cddd()
    num_to_generate = 3

    generated_smiles = wf.find_similars_smiles_from_id(['CHEMBL6273'],
                                                        num_requested=num_to_generate,
                                                        radius=0.75,
                                                        force_unique=True)
    logger.info(generated_smiles)

    assert len(generated_smiles) == num_to_generate + 1


def test_molbart_interpolation():
    context = _create_context()
    wf = MolBART()
    interp_list = interpolation(wf)


def test_molbart_similar_smiles():
    context = _create_context()
    wf = MolBART()
    num_to_generate = 3

    generated_smiles = wf.find_similars_smiles_from_id(['CHEMBL6273'],
                                                       num_requested=num_to_generate,
                                                       radius=0.0001,
                                                       force_unique=True)

    logger.info('# of generated SMILES %s', len(generated_smiles))
    logger.info(generated_smiles)
    assert generated_smiles.shape[0] == num_to_generate + 1
