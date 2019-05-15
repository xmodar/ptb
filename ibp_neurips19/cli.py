"""Command line interface (CLI)."""

from itertools import product

import click

from .__version__ import __version__
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
    datasets = ['MNIST', 'CIFAR10', 'SVHN', 'CIFAR100']
    epsilons = [0.001, 0.01, 0.03, 0.1, 0.2, 0.3, 0.4]
    learning_rates = [1e-1, 1e-2, 1e-3]
    models = ['small_cnn', 'medium_cnn', 'large_cnn']
    for i, (dataset, epsilon, learning_rate, model) in enumerate(
            product(datasets, epsilons, learning_rates, models)):
        if index is not None and i != index:
            continue
        directory = f'{dataset}-{model}-{epsilon}/{learning_rate}'
        command = (f'ibp basic --train -d {dataset} -m {model} -e {epsilon}'
                   f' -lr {learning_rate} -l {directory}'
                   f' -c {directory}/checkpoint.pth')
        print(i, command)
        if run:
            ctx.invoke(
                basic,
                evaluate_only=False,
                dataset=dataset,
                model=model,
                epsilon=epsilon,
                learning_rate=learning_rate,
                log_dir=directory,
                checkpoint=f'{directory}/checkpoint.pth')


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
