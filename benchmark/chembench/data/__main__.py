import os
import logging
import hydra
import pandas as pd

from datetime import datetime
from pydoc import locate
from defusedxml import NotSupportedError
from chembench.data.cache import DatasetCacheGenerator

log = logging.getLogger(__name__)


def convert_runtime(time_):
    return time_.seconds + (time_.microseconds / 1.0e6)


def save_metric_results(mode_name, metric_list, output_dir, return_predictions):
    """Save CSV for metrics"""

    metric_df = pd.concat(metric_list, axis=1).T
    file_path = os.path.join(output_dir, f'{mode_name}')

    if return_predictions:
        pickle_file = file_path + '.pkl'
        log.info(f'Writing predictions to {pickle_file}...')

        if os.path.exists(pickle_file):
            pickle_df = pd.read_pickle(pickle_file)
            pickle_df = pd.concat([pickle_df, metric_df], axis=0)
        else:
            pickle_df = metric_df

        pickle_df.to_pickle(pickle_file)

    if 'predictions' in metric_df.columns:
        metric_df.drop('predictions', inplace=True, axis=1)

    log.info(metric_df)
    csv_file_path = file_path + '.csv'
    write_header = False if os.path.exists(csv_file_path) else True
    metric_df.to_csv(csv_file_path, index=False, mode='a', header=write_header)


@hydra.main(config_path=".",
            config_name="benchmark_metrics")
def main(cfg):
    os.makedirs(cfg.output.path, exist_ok=True)

    inferrer = locate(cfg.model.name)()
    ds_generator = DatasetCacheGenerator(inferrer,
                                         db_file=cfg.sampling.db,
                                         batch_size=cfg.model.batch_size)

    radius = cfg.sampling.radius
    for metric in  cfg.metrics:
        datasets = cfg.metrics[metric].datasets
        num_requested = cfg.sampling.sample_size

        if not cfg.metrics[metric].enabled:
            continue

        for dataset in datasets:
            if hasattr(dataset, 'file'):
                ds_generator.initialize_db(dataset,
                                           radius,
                                           num_requested=num_requested)
            else:
                raise NotSupportedError(f'Only {dataset} with file accepted')
    ds_generator.sample()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for metric_name in  cfg.metrics:
        for radii in radius:
            metric = cfg.metrics[metric_name]
            if not metric.enabled:
                continue

            impl = locate(metric.impl)(cfg)
            kwargs = {'radius', radii,
                      'num_sample', cfg.sampling.sample_size}
            start_time = datetime.now()
            result = impl.calculate(radii, cfg.sampling.sample_size)
            run_time = convert_runtime(datetime.now() - start_time)

            result['inferrer'] = cfg.model.name
            result['iteration'] = 0
            result['run_time'] = run_time
            result['timestamp'] = timestamp
            result['data_size'] = len(impl)

            # Updates to irregularly used arguments
            key_list = ['model', 'gene', 'remove_invalid', 'n_splits']
            for key in key_list:
                if key in kwargs:
                    result[key] = kwargs[key]

            return_predictions = impl.is_prediction()
            save_metric_results(f'{cfg.model.name}_{cfg.exp_name}',
                                [pd.Series(result)],
                                cfg.output.path,
                                return_predictions=return_predictions)

            # metric.cleanup()

if __name__ == '__main__':
    main()
