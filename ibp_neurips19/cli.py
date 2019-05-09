"""Command line interface (CLI)."""

from pathlib import Path

import click

from .__version__ import __version__

__all__ = ['main']

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
