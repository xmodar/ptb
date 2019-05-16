"""Adversarial attacks module."""

import warnings
from collections import namedtuple

import foolbox.attacks
import torch
from foolbox.distances import Linfinity
from foolbox.models import PyTorchModel
from torch.utils.data import Subset
from tqdm import tqdm

from .. import datasets

__all__ = [
    'get_attack_model',
    'get_attack_type',
    'get_attack',
    'get_default_attack',
    'compute_robustness',
]

DEFAULT_TYPE_KWARGS = {
    foolbox.attacks.PGD: {
        'distance': Linfinity,
    }
}
DEFAULT_KWARGS = {
    foolbox.attacks.PGD: {
        'epsilon': 0.1,
        'binary_search': False,
        'stepsize': 0.1,
        'iterations': 200,
        'random_start': True,
        'return_early': True,
    },
}


def get_attack_model(model, dataset, device):
    """Wrap an `torch.nn.Module` as an attack model."""
    mean, std = datasets.MEANS[dataset], datasets.STDS[dataset]
    mean, std = [[[m]] for m in mean], [[[s]] for s in std]
    return PyTorchModel(
        model.eval().to(device),
        bounds=(0, 1),
        num_classes=datasets.NUM_CLASSES[dataset],
        preprocessing=(mean, std),
        device=device,
    )


def get_attack_type(attack_model, attack_name, **kwargs):
    """Get an attack callable on the attack model."""
    attack_class = foolbox.attacks.__dict__[attack_name]
    if attack_class in DEFAULT_TYPE_KWARGS:
        for k, v in DEFAULT_TYPE_KWARGS[attack_class].items():
            if k not in kwargs:
                kwargs[k] = v
    return attack_class(attack_model, **kwargs)


def get_attack(attack_type, **kwargs):
    """Wrap an attack callable with the default keyword arguments."""
    attack_class = attack_type.__class__
    if attack_class in DEFAULT_KWARGS:
        for k, v in DEFAULT_KWARGS[attack_class].items():
            if k not in kwargs:
                kwargs[k] = v

    def attack(*args, **new_kwargs):
        for k, v in kwargs.items():
            if k not in new_kwargs:
                new_kwargs[k] = v
        if 'unpack' not in new_kwargs:
            new_kwargs['unpack'] = False
        return attack_type(*args, **new_kwargs)

    return attack


def get_default_attack(attack_name, model, dataset, device):
    """Get the an attack with the default parameters."""
    attack_model = get_attack_model(model, dataset, device)
    attack_type = get_attack_type(attack_model, attack_name)
    return get_attack(attack_type)


@torch.enable_grad()
def compute_robustness(  # pylint: disable=dangerous-default-value
        model,
        dataset,
        device,
        attack_name='PGD',
        subset=None,
        subset_seed=None,
        restarts=1,
        attack_kwargs={},
        type_kwargs={},
        **tqdm_kwargs):
    """Compute the adversarial robustness of the model using foolbox."""
    device = torch.device(device)
    attack_model = get_attack_model(model, dataset, device)
    attack_type = get_attack_type(attack_model, attack_name, **type_kwargs)
    attack = get_attack(attack_type, **attack_kwargs)
    errors = []
    fooling_rate = 0
    progress = None
    warn = warnings.warn
    warnings.warn = lambda *a, **k: None
    if 'desc' not in tqdm_kwargs:
        tqdm_kwargs['desc'] = f'Computing {attack_name} robustness'
    mean, std = datasets.MEANS[dataset], datasets.STDS[dataset]
    if subset:
        dataset = datasets.get_dataset(dataset, False)
        with torch.random.fork_rng([], subset_seed is not None):
            if subset_seed:
                torch.default_generator.manual_seed(subset_seed)
            dataset = Subset(dataset, torch.randperm(len(dataset))[:subset])
    loader = datasets.get_loader(dataset, False, 100, device.type == 'cuda', 4)
    for images, _ in loader:
        if progress is None:
            progress = tqdm(total=len(loader) * len(images), **tqdm_kwargs)
        with torch.no_grad():
            labels = model(images.to(device)).argmax(-1)
        for image, label in zip(images, labels):
            [  # pylint: disable=expression-not-assigned
                c.mul_(s).add_(m).clamp_(0, 1)
                for c, m, s in zip(image, mean, std)
            ]
            for _ in range(restarts):
                adversarial = attack(image.numpy(), int(label))
                failed = (adversarial.distance.value == 0 or
                          adversarial.reached_threshold() or
                          adversarial.image is None)
                if not failed:
                    break
            errors.append(adversarial.distance.value)
            if not failed:
                fooling_rate += 1
            progress.update()
    progress.update(progress.total - progress.n)
    progress.close()
    warnings.warn = warn
    fooling_rate /= len(errors)
    sorted_errors = sorted(errors)
    robustness = sorted_errors[len(sorted_errors) // 2]
    return compute_robustness.output_type(robustness, fooling_rate,
                                          sorted_errors)


compute_robustness.output_type = namedtuple(
    'robustness_results', ['robustness', 'fooling_rate', 'sorted_errors'])
