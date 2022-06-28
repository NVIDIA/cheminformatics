import logging
import numpy as np
import pickle
import sqlite3
import torch

from contextlib import closing
from collections import defaultdict

from chembench.metrics import BaseMetric
from chembench.utils.smiles import calc_similarity

logger = logging.getLogger(__name__)

__all__ = ['Validity',
           'Unique',
           'Novelty',
           'NonIdenticality',
           'EffectiveNovelty',
           'ScaffoldUnique',
           'ScaffoldNonIdenticalSimilarity',
           'ScaffoldNovelty',
           'EffectiveScaffoldNovelty',
           'Entropy']


class Validity(BaseMetric):
    name = 'validity'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)

    def compute_metrics(self, num_samples, radius):
        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            valid_result = conn.execute('''
                SELECT ss.is_valid, count(*)
                FROM smiles s, smiles_samples ss
                WHERE s.id = ss.input_id
                    AND s.scaled_radius = ?
                    AND ss.is_generated = 1
                    AND s.processed = 1
                    AND s.dataset_type = ?
                GROUP BY ss.is_valid;
                ''',
                [radius, 'SAMPLE'])

            valid_molecules = 0
            total_molecules = 0
            for rec in valid_result.fetchall():
                total_molecules += rec[1]
                if rec[0] == 1:
                    valid_molecules += rec[1]
        self.total_molecules = total_molecules//num_samples
        return valid_molecules, total_molecules


class Unique(BaseMetric):
    name = 'unique'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)

    def compute_metrics(self, num_samples, radius):
        validity = Validity(self.name, self.metric_spec, self.cfg)
        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            unique_result = conn.execute(f'''
                SELECT sum(ratio)
                FROM (
                    SELECT CAST(count(DISTINCT ss.smiles) as float) / CAST(count(ss.smiles) as float) ratio
                    FROM smiles s, smiles_samples ss
                    WHERE s.id = ss.input_id
                        AND s.scaled_radius = ?
                        AND ss.is_valid = 1
                        AND ss.is_generated = 1
                        AND s.processed = 1
                        AND s.dataset_type = ?
                    GROUP BY s.id
                )''',
                [radius, 'SAMPLE'])

            rec = unique_result.fetchone()
        _, self.total_molecules = validity.compute_metrics(num_samples, radius)
        self.total_molecules = self.total_molecules//num_samples
        return rec[0], self.total_molecules


class ScaffoldUnique(BaseMetric):
    name = 'scaffold_unique'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)

    def compute_metrics(self, num_samples, radius):
        validity = Validity(self.name, self.metric_spec, self.cfg)
        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            unique_result = conn.execute(f'''
                SELECT sum(ratio)
                FROM (
                    SELECT CAST(count(DISTINCT ss.scaffold) as float) / CAST(count(ss.scaffold) as float) ratio
                FROM smiles s, smiles_samples ss
                WHERE s.id = ss.input_id
                    AND s.scaled_radius = ?
                    AND ss.is_valid = 1
                    AND ss.is_generated = 1
                    AND s.processed = 1
                    AND s.dataset_type = ?
                GROUP BY s.id
                )''',
                [radius, 'SAMPLE'])

            rec = unique_result.fetchone()
        _, self.total_molecules = validity.compute_metrics(num_samples, radius)
        self.total_molecules = self.total_molecules//num_samples
        return rec[0], self.total_molecules


class Novelty(BaseMetric):
    name = 'novelty'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)
        self.training_data = cfg.model.training_data

    def compute_metrics(self, num_samples, radius):
        validity = Validity(self.name, self.metric_spec, self.cfg)
        valid_molecules, self.total_molecules = validity.compute_metrics(num_samples, radius)
        self.total_molecules = self.total_molecules//num_samples

        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            conn.execute('ATTACH ? AS training_db', [self.training_data])
            res = conn.execute('''
                SELECT count(distinct ss.smiles)
                FROM main.smiles s, main.smiles_samples ss, training_db.train_data td
                WHERE ss.smiles = td.smiles
                    AND s.id = ss.input_id
                    AND s.scaled_radius = ?
                    AND ss.is_valid = 1
                    AND ss.is_generated = 1
                    AND s.processed = 1
                    AND s.dataset_type = ?
                ''',
                [radius, 'SAMPLE'])
            rec = res.fetchone()
            novel_molecules = valid_molecules - rec[0]

        return novel_molecules, valid_molecules


class NonIdenticality(BaseMetric):
    name = 'non_identicality'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)
        self.training_data = cfg.model.training_data

    def compute_metrics(self, num_samples, radius):
        validity = Validity(self.name, self.metric_spec, self.cfg)
        _, self.total_molecules = validity.compute_metrics(num_samples, radius)
        self.total_molecules = self.total_molecules//num_samples

        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            res = conn.execute('''
                SELECT sum(ratio)
                FROM (
                    SELECT CAST(identical_smiles.cnt as float) / CAST(count(ss.smiles) as float) ratio
                FROM smiles s, smiles_samples ss,
                (SELECT s2.id, count(*) cnt
                FROM smiles s2, smiles_samples ss2
                WHERE s2.id = ss2.input_id
                    AND s2.smiles = ss2.smiles
                    AND s2.scaled_radius = ?
                    AND ss2.is_valid = 1
                    AND ss2.is_generated = 1
                    AND s2.processed = 1
                    AND s2.dataset_type = 'SAMPLE'
                GROUP BY s2.id) as identical_smiles
                WHERE s.id = ss.input_id
                    AND identical_smiles.id = s.id
                    AND s.scaled_radius = ?
                    AND ss.is_valid = 1
                    AND ss.is_generated = 1
                    AND s.processed = 1
                    AND s.dataset_type = 'SAMPLE'
                GROUP BY s.id
                )''',
                [radius, radius])
            rec = res.fetchone()
        # logger.info(f'{rec}, {rec[0]}, {self.total_molecules}, {(self.total_molecules - rec[0])/self.total_molecules}, {(rec[0])/self.total_molecules}')
        identical = rec[0] if rec[0] is not None else 0
        return self.total_molecules - identical, self.total_molecules


#TODO: duplicate this for entire molecules
class ScaffoldNonIdenticalSimilarity(BaseMetric):
    name = 'scaffold_non_identical_similarity'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)
        self.training_data = cfg.model.training_data
        self.nbits = cfg.metrics.scaffold_non_identical_similarity.nbits

    def compute_metrics(self, num_samples, radius):
        validity = Validity(self.name, self.metric_spec, self.cfg)
        _, self.total_molecules = validity.compute_metrics(num_samples, radius)
        self.total_molecules = self.total_molecules//num_samples

        # What is the average tanimoto similarity between the scaffolds of the input molecule and the non identical generated molecules
        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            res = conn.execute('''
                SELECT ss.input_id, input_smiles.scaffold as input_scaffold, ss.scaffold
                FROM smiles s, smiles_samples ss,
                (SELECT ss.input_id, ss.smiles, ss.scaffold
                FROM smiles s, smiles_samples ss
                WHERE s.id = ss.input_id
                    AND ss.is_generated = 0
                ) as input_smiles
                WHERE s.id = ss.input_id
                    AND ss.is_generated = 1
                    AND s.smiles <> ss.smiles
                    AND input_smiles.input_id = ss.input_id
                    AND s.scaled_radius = ?
                    AND ss.is_valid = 1
                    AND ss.is_generated = 1
                    AND s.processed = 1
                    AND s.dataset_type = 'SAMPLE'
                ORDER BY s.id
                ''',
                [radius])
            recs = res.fetchall()
        scaffolds = defaultdict(list)
        for rec in recs:
            idx, start_scaffold, generated_scaffold = rec
            scaffolds[idx].append((start_scaffold, generated_scaffold))
        sims = []
        for input_id, v in scaffolds.items():
            sims.append(calc_similarity(v, nbits=self.nbits))
        return np.mean(sims), 1


class EffectiveNovelty(BaseMetric):
    name = 'effective_novelty'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)
        self.training_data = cfg.model.training_data

    def compute_metrics(self, num_samples, radius):
        validity = Validity(self.name, self.metric_spec, self.cfg)
        _, self.total_molecules = validity.compute_metrics(num_samples, radius)
        self.total_molecules = self.total_molecules//num_samples

        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            conn.execute('ATTACH ? AS training_db', [self.training_data])
            res = conn.execute('''
                SELECT SUM(CAST(a.smiles_cnt - case when b.recs is null then 0 else b.recs end as float) / ?)
                FROM (SELECT s.id as id, count(distinct ss.smiles) smiles_cnt
                        FROM main.smiles s, main.smiles_samples ss
                        WHERE s.id = ss.input_id
                            AND s.smiles <> ss.smiles
                            AND ss.is_valid = 1
                            AND ss.is_generated = 1
                            AND s.processed = 1
                            AND s.scaled_radius = ?
                            AND s.dataset_type = ?
                        GROUP BY s.id) as a
                    LEFT OUTER JOIN
                    (SELECT s.id, count(distinct ss.smiles) recs
                    FROM main.smiles s, main.smiles_samples ss, training_db.train_data td
                    WHERE ss.smiles == td.smiles
                        AND s.id = ss.input_id
                        AND ss.is_valid = 1
                        AND ss.is_generated = 1
                        AND s.processed = 1
                        AND s.scaled_radius = ?
                        AND s.dataset_type = ?
                    GROUP BY s.id) as b
                ON a.id = b.id
                ''',
                [num_samples, radius, 'SAMPLE', radius, 'SAMPLE'])
            rec = res.fetchone()
        recs = 0 if rec[0] is None else rec[0]
        return recs, self.total_molecules


class ScaffoldNovelty(BaseMetric):
    name = 'scaffold_novelty'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)
        self.training_data = cfg.model.training_data

    def compute_metrics(self, num_samples, radius):
        validity = Validity(self.name, self.metric_spec, self.cfg)
        valid_molecules, self.total_molecules = validity.compute_metrics(num_samples, radius)
        #TODO: do we need valid_scaffold when just the number is needed and its 1:1 --> guessing yes
        self.total_molecules = self.total_molecules//num_samples
        # comma after FROM (SELECT count(distinct td.scaffold) scff_cnt
        #             FROM training_db.train_data td
        #             ) as train_smiles , train_smiles.scff_cnt
        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            conn.execute('ATTACH ? AS training_db', [self.training_data])
            res = conn.execute('''
                SELECT count(distinct ss.scaffold), train_smiles.scff_cnt
                FROM main.smiles s, main.smiles_samples ss, training_db.scaffolds td,
                (SELECT count(distinct ss.scaffold) scff_cnt
                    FROM main.smiles_samples ss
                    WHERE ss.is_generated = 1
                    ) as train_smiles
                WHERE ss.scaffold = td.scaffold
                    AND s.id = ss.input_id
                    AND s.scaled_radius = ?
                    AND ss.is_valid = 1
                    AND ss.is_generated = 1
                    AND s.processed = 1
                    AND s.dataset_type = ?
                ''',
                [radius, 'SAMPLE'])
            rec = res.fetchone()
            non_novel_scaffolds = rec[0]
            td_scaffolds = rec[1]

        return td_scaffolds - non_novel_scaffolds, td_scaffolds


class EffectiveScaffoldNovelty(BaseMetric):
    name = 'effective_scaffold_novelty'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)
        self.training_data = cfg.model.training_data

    def compute_metrics(self, num_samples, radius):
        validity = Validity(self.name, self.metric_spec, self.cfg)
        _, self.total_molecules = validity.compute_metrics(num_samples, radius)
        self.total_molecules = self.total_molecules//num_samples
        # total molecules is 1:1 with total scaffolds
        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            conn.execute('ATTACH ? AS training_db', [self.training_data])
            res = conn.execute('''
                SELECT SUM(CAST(a.smiles_cnt - case when b.recs is null then 0 else b.recs end as float) / ?)
                FROM (SELECT s.id as id, count(distinct ss.scaffold) smiles_cnt
                        FROM main.smiles s, main.smiles_samples ss,
                            (SELECT s2.id, ss2.input_id, ss2.smiles, ss2.scaffold
                            FROM main.smiles s2, main.smiles_samples ss2
                            WHERE s2.id = ss2.input_id
                                AND ss2.is_generated = 0
                            GROUP BY s2.id) as input_smiles
                        WHERE s.id = ss.input_id
                            AND s.smiles <> ss.smiles
                            AND input_smiles.scaffold <> ss.scaffold
                            AND input_smiles.input_id = ss.input_id
                            AND ss.is_valid = 1
                            AND ss.is_generated = 1
                            AND s.processed = 1
                            AND s.scaled_radius = ?
                            AND s.dataset_type = ?
                        GROUP BY s.id) as a
                    LEFT OUTER JOIN
                    (SELECT s.id, count(distinct ss.scaffold) recs
                    FROM main.smiles s, main.smiles_samples ss, training_db.scaffolds td
                    WHERE ss.scaffold == td.scaffold
                        AND s.id = ss.input_id
                        AND ss.is_valid = 1
                        AND ss.is_generated = 1
                        AND s.processed = 1
                        AND s.scaled_radius = ?
                        AND s.dataset_type = ?
                    GROUP BY s.id) as b
                ON a.id = b.id
                ''',
                [num_samples, radius, 'SAMPLE',  radius, 'SAMPLE'])
            rec = res.fetchone()
        return rec[0], self.total_molecules


class Entropy(BaseMetric):
    name = 'entropy'

    def __init__(self, metric_name, metric_spec, cfg):
        super().__init__(metric_name, metric_spec, cfg)

    def compute_metrics(self, num_samples, radius):
        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            result = conn.execute('''
                SELECT ss.embedding, ss.embedding_dim
                FROM smiles s, smiles_samples ss
                WHERE s.id = ss.input_id
                    AND s.scaled_radius = ?
                    AND ss.is_generated = 0
                    AND s.processed = 1
                    AND s.dataset_type = ?
                GROUP BY ss.input_id;
                ''',
                [radius, 'SAMPLE'])

            latent_space = []
            for rec in result.fetchall():
                dim = pickle.loads(rec[1])
                latent_vector = torch.FloatTensor(list(pickle.loads(rec[0])))
                latent_vector = torch.reshape(latent_vector, dim)
                # logger.info(f'[ENTROPY Shape] {latent_vector.shape}')
                latent_vector = latent_vector.mean(axis=0).tolist()
                # logger.info(f'[ENTROPY Length] {len(latent_vector)}')
                latent_space.append(latent_vector)
        # # pip install . on NPEET which is from github.com/gregversteeg/NPEET
        from npeet import entropy_estimators as ee
        entropy = ee.entropy(latent_space)
        x = np.asarray(latent_space)
        logger.info(f'[ENTROPY] {entropy}')
        logger.info(f'[Normalized ENTROPY] {ee.entropy(x / (x.std(0) + 1e-10))}')
        return entropy, 1
