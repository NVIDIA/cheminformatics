import os
import logging
from defusedxml import NotSupportedError
import hydra

from chembench.data.cache import DatasetCacheGenerator


log = logging.getLogger('model benchmarking')


@hydra.main(config_path=".",
            config_name="benchmark_metrics",
            version_base=None)
def main(cfg):
    os.makedirs(cfg.output.path, exist_ok=True)

    # TODO: Replace None with an inference wrapper
    ds_generator = DatasetCacheGenerator(None, db_file=cfg.sampling.db,)

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


if __name__ == '__main__':
    main()
