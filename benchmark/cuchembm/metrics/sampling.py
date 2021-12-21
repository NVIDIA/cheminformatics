#!/usr/bin/env python3

import logging
import numpy as np
import sqlite3

from contextlib import closing


logger = logging.getLogger(__name__)

__all__ = ['Validity', 'Unique', 'Novelty']


class BaseSampleMetric():
    name = None

    """Base class for metrics based on sampling for a single SMILES string"""
    def __init__(self,
                 inferrer,
                 cfg):
        self.inferrer = inferrer
        self.cfg = cfg
        self.total_molecules = 0

    def __len__(self):
        return self.total_molecules

    def _calculate_metric(self, metric_array, num_array):
        return np.nanmean(metric_array / num_array)

    def variations(self):
        return NotImplemented

    def compute_metrics(self, num_samples, radius):
        return NotImplemented


    def calculate(self, radius, num_samples, **kwargs):
        metric_array, num_array = self.compute_metrics(num_samples, radius)
        metric = self._calculate_metric(metric_array, num_array)

        return {'name': self.name,
                'value': metric,
                'radius': radius,
                'num_samples': num_samples}


class Validity(BaseSampleMetric):
    name = 'validity'

    def __init__(self, inferrer, cfg):
        super().__init__(inferrer, cfg)
        self.name = Validity.name

    def variations(self, cfg, **kwargs):
        radius_list = list(cfg.metric.validity.radius)
        radius_list = [float(x) for x in radius_list]
        return {'radius': radius_list}

    def compute_metrics(self, num_samples, radius):
        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            valid_result = conn.execute('''
                SELECT ss.is_valid, count(*)
                FROM smiles s, smiles_samples ss
                WHERE s.id = ss.input_id
                    AND s.model_name = ?
                    AND s.scaled_radius = ?
                    AND s.force_unique = ?
                    AND s.sanitize = ?
                    AND ss.is_generated = 1
                    AND s.processed = 1
                    AND s.dataset_type = ?
                GROUP BY ss.is_valid;
                ''',
                [self.inferrer.__class__.__name__, radius, 0, 1, 'SAMPLE'])

            valid_molecules = 0
            total_molecules = 0
            for rec in valid_result.fetchall():
                total_molecules += rec[1]
                if rec[0] == 1:
                    valid_molecules += rec[1]
        self.total_molecules = total_molecules//num_samples
        return valid_molecules, total_molecules


class Unique(BaseSampleMetric):
    name = 'unique'

    def __init__(self, inferrer, cfg):
        super().__init__(inferrer, cfg)
        self.name = Unique.name

    def variations(self, cfg, **kwargs):
        radius_list = list(cfg.metric.unique.radius)
        radius_list = [float(x) for x in radius_list]
        return {'radius': radius_list}

    def compute_metrics(self, num_samples, radius):
        validity = Validity(self.inferrer, self.cfg)
        with closing(sqlite3.connect(self.cfg.sampling.db,
                                     uri=True,
                                     check_same_thread=False)) as conn:
            unique_result = conn.execute(f'''
                SELECT sum(ratio)
                FROM (
                    SELECT CAST(count(DISTINCT ss.smiles) as float) / CAST(count(ss.smiles) as float) ratio
                FROM smiles s, smiles_samples ss
                WHERE s.id = ss.input_id
                    AND s.model_name = ?
                    AND s.scaled_radius = ?
                    AND s.force_unique = ?
                    AND s.sanitize = ?
                    AND ss.is_valid = 1
                    AND ss.is_generated = 1
                    AND s.processed = 1
                    AND s.dataset_type = ?
                GROUP BY s.id
                )''',
                [self.inferrer.__class__.__name__, radius, 0, 1, 'SAMPLE'])

            rec = unique_result.fetchone()
        _, self.total_molecules = validity.compute_metrics(num_samples, radius)
        self.total_molecules = self.total_molecules//num_samples
        return rec[0], self.total_molecules


class Novelty(BaseSampleMetric):
    name = 'novelty'

    def __init__(self, inferrer, cfg):
        super().__init__(inferrer, cfg)
        self.name = Novelty.name
        self.training_data = cfg.model.training_data

    def variations(self, cfg, **kwargs):
        radius_list = list(cfg.metric.novelty.radius)
        radius_list = [float(x) for x in radius_list]
        return {'radius': radius_list}

    def compute_metrics(self, num_samples, radius):
        validity = Validity(self.inferrer, self.cfg)
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
                    AND s.model_name = ?
                    AND s.scaled_radius = ?
                    AND s.force_unique = ?
                    AND s.sanitize = ?
                    AND ss.is_valid = 1
                    AND ss.is_generated = 1
                    AND s.processed = 1
                    AND s.dataset_type = ?
                ''',
                [self.inferrer.__class__.__name__, radius, 0, 1, 'SAMPLE'])
            rec = res.fetchone()
            novel_molecules = valid_molecules - rec[0]

        return novel_molecules, valid_molecules
