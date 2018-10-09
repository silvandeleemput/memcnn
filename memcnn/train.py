import argparse
import os
import logging
import torch

from memcnn.config import Config
from memcnn.experiment.manager import ExperimentManager
from memcnn.experiment.factory import load_experiment_config, experiment_config_parser

import memcnn.utils.log


logger = logging.getLogger('train')


def run_experiment(experiment_tags, data_dir, results_dir, start_fresh=False, use_cuda=False, workers=None,
                   experiments_file=None, *args, **kwargs):
    if not os.path.exists(data_dir):
        raise RuntimeError('Cannot find data_dir directory: {}'.format(data_dir))

    if not os.path.exists(results_dir):
        raise RuntimeError('Cannot find results_dir directory: {}'.format(results_dir))

    cfg = load_experiment_config(experiments_file, experiment_tags)
    logger.info(cfg)

    model, optimizer, trainer, trainer_params = experiment_config_parser(cfg, workers=workers, data_dir=data_dir)

    experiment_dir = os.path.join(results_dir, '_'.join(experiment_tags))
    manager = ExperimentManager(experiment_dir, model, optimizer)
    if start_fresh:
        logger.info('Starting fresh option enabled. Clearing all previous results...')
        manager.delete_dirs()
    manager.make_dirs()

    if use_cuda:
        manager.model = manager.model.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    last_iter = manager.get_last_model_iteration()
    if last_iter > 0:
        logger.info('Continue experiment from iteration: {}'.format(last_iter))
        manager.load_train_state(last_iter)

    trainer_params.update(kwargs)

    trainer(manager, start_iter=last_iter, use_cuda=use_cuda, *args, **trainer_params)


def main():
    # setup logging
    memcnn.utils.log.setup(True)

    # specify defaults for arguments
    use_cuda = torch.cuda.is_available()
    workers = 16
    data_dir = Config()['data_dir']
    results_dir = Config()['results_dir']
    experiments_file = os.path.join(os.path.dirname(__file__), 'config', 'experiments.json')
    start_fresh = False

    # parse arguments
    parser = argparse.ArgumentParser(description='Run memcnn experiments.')
    parser.add_argument('experiment_tags', type=str, nargs='+',
                        help='Experiment tags to run and combine from the experiment config file')
    parser.add_argument('--workers', dest='workers', type=int, default=workers,
                        help='Number of workers for data loading (Default: {})'.format(workers))
    parser.add_argument('--results-dir', dest='results_dir', type=str, default=results_dir,
                        help='Directory for storing results (Default: {})'.format(results_dir))
    parser.add_argument('--data-dir', dest='data_dir', type=str, default=data_dir,
                        help='Directory for input data (Default: {})'.format(data_dir))
    parser.add_argument('--experiments-file', dest='experiments_file', type=str, default=experiments_file,
                        help='Experiments file (Default: {})'.format(experiments_file))
    parser.add_argument('--fresh', dest='start_fresh', action='store_true', default=start_fresh,
                        help='Start with fresh experiment, clears all previous results (Default: {})'
                        .format(start_fresh))
    parser.add_argument('--no-cuda', dest='use_cuda', action='store_false', default=use_cuda,
                        help='Always disables GPU use (Default: use when available)')
    args = parser.parse_args()

    if not use_cuda:
        logger.warning('CUDA is not available in the current configuration!!!')

    if not args.use_cuda:
        logger.warning('CUDA is disabled!!!')

    # run experiment given arguments
    run_experiment(
        args.experiment_tags,
        args.data_dir,
        args.results_dir,
        start_fresh=args.start_fresh,
        experiments_file=args.experiments_file,
        use_cuda=args.use_cuda, workers=args.workers)


if __name__ == '__main__':
    main()
