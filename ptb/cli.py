"""Command line interface (CLI)."""

from argparse import Namespace
from collections import OrderedDict
from itertools import product
from pathlib import Path

import click
import torch

from . import models
from .__version__ import __version__
from .attacks import compute_robustness
from .train import train_classifier

__all__ = ['main', 'basic', 'experiment']


@click.group()
@click.version_option(__version__, '-v', '--version')
def main():
    """Interval Bound Propagation (NeurIPS 2019)."""
    return


@main.command()
@click.option(
    '-v/-t',
    '--validate/--train',
    'evaluate_only',
    is_flag=True,
    default=True,
    show_default=True,
    help='Whether to run in validation or training mode.')
@click.option(
    '--dataset',
    '-d',
    type=click.STRING,
    default='MNIST',
    help='Which dataset to use.')
@click.option(
    '--model',
    '-m',
    type=click.STRING,
    default='small_cnn',
    help='Which model architecture to use.')
@click.option(
    '-p/-s',
    '--pretrained/--scratch',
    'pretrained',
    is_flag=True,
    default=False,
    show_default=True,
    help='Whether to load a pretrained model.')
@click.option(
    '--learning-rate',
    '-lr',
    type=click.FloatRange(min=0),
    default=1e-1,
    help='Learning rate.')
@click.option(
    '--momentum',
    '-mm',
    type=click.FloatRange(min=0),
    default=0.9,
    help='SGD momentum.')
@click.option(
    '--weight-decay',
    '-w',
    type=click.FloatRange(min=0),
    default=1e-4,
    help='SGD weight decay.')
@click.option(
    '--factor',
    '-f',
    type=click.FloatRange(min=0),
    default=1e-4,
    help='The trade-off coefficient.')
@click.option(
    '--temperature',
    '-t',
    type=click.FloatRange(min=0),
    default=1e-4,
    help='Softmax temperature.')
@click.option(
    '--number-of-epochs',
    '-n',
    'epochs',
    type=click.IntRange(min=0),
    default=90,
    help='The maximum number of epochs.')
@click.option(
    '--batch-size',
    '-b',
    type=click.IntRange(min=0),
    default=256,
    help='Mini-batch size.')
@click.option(
    '--jobs',
    '-j',
    type=click.IntRange(min=0),
    default=4,
    help='Number of threads for data loading when using cuda.')
@click.option(
    '--checkpoint',
    '-c',
    type=click.Path(path_type=str),
    default='checkpoint.pth',
    help='A checkpoint file to save the best model.')
@click.option(
    '--resume',
    '-r',
    type=click.Path(path_type=str),
    default='',
    help='A checkpoint file to resume from.')
@click.option(
    '--log-dir',
    '-l',
    type=click.Path(path_type=str),
    default='logs',
    help='A tensorboard logs directory.')
@click.option(
    '--seed',
    '-sd',
    type=click.IntRange(),
    default=None,
    help='Seed the random number generators (slow!).')
@click.option(
    '--epsilon',
    '-e',
    type=click.FloatRange(),
    default=0,
    help='Epsilon used for training with interval bounds.')
def basic(*args, **kwargs):
    """Start basic neural network training."""
    train_classifier(*args, **kwargs)


@main.command()
@click.option(
    '-r/-s',
    '--run/--show',
    'run',
    is_flag=True,
    default=False,
    show_default=True,
    help='Whether to run or show the experiment(s).')
@click.option(
    '--index',
    '-i',
    type=click.IntRange(0),
    default=None,
    help='Which experiment.')
@click.pass_context
def experiment(ctx, run, index):
    """Run one of the experiments."""
    combinations = OrderedDict(
        dataset=['MNIST', 'CIFAR10'],
        model=['small_cnn', 'medium_cnn', 'large_cnn'],
        epsilon=[2 / 255, 8 / 255, 16 / 255, 0.1, 0.2, 0.3, 0.4],
        learning_rate=[1e-2, 1e-3],
        factor=[1e-2, 1e-1, 1, 1e1, 1e2],
        temperature=[1, 5],
    )

    def gen():
        for c in product(*combinations.values()):
            c = Namespace(**dict(zip(combinations.keys(), c)))
            if c.dataset == 'MNIST' and c.epsilon < 0.1:
                continue
            if c.dataset == 'CIFAR10' and c.epsilon > 0.1:
                continue
            yield c

    for i, c in enumerate(gen()):
        if index is not None and i != index:
            continue
        parameters = f'{c.learning_rate}-{c.factor}-{c.temperature}'
        directory = Path(f'{c.dataset}-{c.model}-{c.epsilon}/{parameters}')
        checkpoint_file = directory / 'checkpoint.pth'
        command = (f'ptb basic --train -d {c.dataset} -m {c.model}'
                   f' -e {c.epsilon} -f {c.factor} -t {c.temperature}'
                   f' -lr {c.learning_rate} -l {directory}'
                   f' -c {checkpoint_file}')
        print(i, command)
        if run:
            ctx.invoke(
                basic,
                evaluate_only=False,
                dataset=c.dataset,
                model=c.model,
                epsilon=c.epsilon,
                factor=c.factor,
                temperature=c.temperature,
                learning_rate=c.learning_rate,
                log_dir=directory,
                checkpoint=checkpoint_file)

            # compute PGD
            net = models.__dict__[c.model]()
            models.fit_to_dataset(net, c.dataset).eval()
            checkpoint = torch.load(checkpoint_file)
            net.load_state_dict(checkpoint['state_dict'])

            test_epsilons = combinations['epsilon']
            if c.dataset == 'MNIST':
                test_epsilons = [e for e in test_epsilons if e >= 0.1]
            elif c.dataset == 'CIFAR10':
                test_epsilons = [e for e in test_epsilons if e <= 0.1]
            for test_epsilon in test_epsilons:
                results = compute_robustness(
                    net,
                    c.dataset,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    attack_kwargs=dict(epsilon=test_epsilon))
                torch.save({
                    'model': c.model,
                    'dataset': c.dataset,
                    'accuracy': checkpoint['best_acc1'],
                    'epsilon': c.epsilon,
                    'learning_rate': c.learning_rate,
                    'seed': None,
                    'subset': None,
                    'restarts': 1,
                    'test_epsilon': test_epsilon,
                    'robustness': results.robustness,
                    'fooling_rate': results.fooling_rate,
                    'sorted_errors': results.sorted_errors,
                }, directory / f'pgd-{test_epsilon}.pth')


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
