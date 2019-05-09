"""Command line interface (CLI)."""

from pathlib import Path

import click

from .__version__ import __version__
from .train import train_classifier

__all__ = ['main', 'basic']

DEBUG = False
PREFIX = Path(__file__).parent.name.upper()


class Config:

    def __init__(self):
        self.debug = DEBUG


pass_config = click.make_pass_decorator(Config, ensure=True)


@click.group()
@click.option(
    '-r/-d',
    '--run/--debug',
    'run_mode',
    is_flag=True,
    default=not DEBUG,
    show_default=True,
    envvar='_'.join([PREFIX, 'RUN']),
    help='Whether to run without debugging.')
@click.version_option(__version__, '-v', '--version')
@pass_config
def main(config, run_mode):
    """Interval Bound Propagation (NeurIPS 2019)."""
    config.debug = not run_mode
    print(f'Running in debug mode: {config.debug}')


@main.command()
@click.option(
    '-e/-t',
    '--evaluate/--train',
    'evaluate_only',
    is_flag=True,
    default=True,
    show_default=True,
    help='Whether to run in evaluation or training mode.')
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
    '--gpu',
    '-g',
    type=click.IntRange(min=0),
    default=None,
    help='The GPU to use (None uses all available GPUs).')
@click.option(
    '--jobs',
    '-j',
    type=click.IntRange(min=0),
    default=4,
    help='Number of threads for data loading when using cuda.')
@click.option(
    '--resume',
    '-r',
    type=click.Path(path_type=str),
    default='',
    help='A checkpoint file to resume from.')
@click.option(
    '--checkpoint',
    '-c',
    type=click.Path(path_type=str),
    default='checkpoint.pth',
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
def basic(*args, **kwargs):
    """Start basic neural network training."""
    train_classifier(*args, **kwargs)
