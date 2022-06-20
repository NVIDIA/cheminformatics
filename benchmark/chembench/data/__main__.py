import os
import logging
import hydra

from pydoc import locate
from defusedxml import NotSupportedError
from chembench.data.cache import DatasetCacheGenerator


log = logging.getLogger(__name__)


@hydra.main(config_path=".",
            config_name="benchmark_metrics")
def main(cfg):
    os.makedirs(cfg.output.path, exist_ok=True)

    inferrer = locate(cfg.model.name)()
    ds_generator = DatasetCacheGenerator(inferrer,
                                         db_file=cfg.sampling.db,
                                         batch_size=cfg.model.batch_size)

    for metric in  cfg.metrics:
        datasets = cfg.metrics[metric].datasets
        num_requested = cfg.sampling.sample_size

        if not cfg.metrics[metric].enabled:
            continue

        for dataset in datasets:
            if hasattr(dataset, 'file'):
                ds_generator.initialize_db(dataset,
                                           num_requested=num_requested)
            else:
                raise NotSupportedError(f'Only {dataset} with file accepted')

    ds_generator.sample()

if __name__ == '__main__':
    main()
