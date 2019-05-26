## Probabilistically True and Tight Bounds for Robust DNN Training

Training Deep Neural Networks (DNNs) that are robust to norm bounded adversarial attacks remains an elusive problem. While verification based methods are generally too expensive to robustly train large networks, it was demonstrated in Gowal et. al. that bounded input intervals can be inexpensively propagated per layer through large networks. This interval bound propagation (IBP) approach lead to high robustness and was the first to be employed on large networks. However, due to the very loose nature of the IBP bounds, particularly for large networks, the required training procedure is complex and involved. In this paper, we closely examine the bounds of a block of layers composed of an affine layer followed by a ReLU nonlinearity followed by another affine layer. In doing so, we propose _probabilistic_ bounds, true bounds with overwhelming probability, that are provably tighter than IBP bounds in expectation. We then extend this result to deeper networks through blockwise propagation and show that we can achieve orders of magnitudes tighter bounds compared to IBP. With such tight bounds, we demonstrate that a simple standard training procedure can achieve the best robustness-accuracy trade-off across several architectures on both MNIST and CIFAR10.

### Usage

Install it using [poetry 0.12.11](https://github.com/sdispater/poetry) (the recommended way):
```sh
git clone https://github.com/ModarTensai/ptb && cd ptb
poetry install
```

For Jupyter support:
```sh
poetry install --extras jupyter
poetry run python -m ipykernel install --user --name ptb
```

Or using pip:
```sh
pip install git+https://github.com/ModarTensai/ptb
```

Then, check this out:
```
$ ptb basic --help
Usage: ptb basic [OPTIONS]

  Start basic neural network training.

Options:
  -v, --validate / -t, --train    Whether to run in validation or training
                                  mode.  [default: True]
  -d, --dataset TEXT              Which dataset to use.
  -m, --model TEXT                Which model architecture to use.
  -p, --pretrained / -s, --scratch
                                  Whether to load a pretrained model.
                                  [default: False]
  -lr, --learning-rate FLOAT RANGE
                                  Learning rate.
  -mm, --momentum FLOAT RANGE     SGD momentum.
  -w, --weight-decay FLOAT RANGE  SGD weight decay.
  -f, --factor FLOAT RANGE        The trade-off coefficient.
  -t, --temperature FLOAT RANGE   Softmax temperature.
  -n, --number-of-epochs INTEGER RANGE
                                  The maximum number of epochs.
  -b, --batch-size INTEGER RANGE  Mini-batch size.
  -j, --jobs INTEGER RANGE        Number of threads for data loading when
                                  using cuda.
  -c, --checkpoint PATH           A checkpoint file to save the best model.
  -r, --resume PATH               A checkpoint file to resume from.
  -l, --log-dir PATH              A tensorboard logs directory.
  -sd, --seed INTEGER RANGE       Seed the random number generators (slow!).
  -e, --epsilon FLOAT RANGE       Epsilon used for training with interval
                                  bounds.
  --help                          Show this message and exit.
```

Please, note that in this current version, training using PTB on multiple GPUs is not supported. However, the code will try to use all available GPUs which will lead to an error with DataParallel. In case you have more than one GPU, don't forget to set the environment variable `CUDA_VISIBLE_DEVICES` to only one GPU.

### Cite

This is the official implementation of the method described in [this paper](https://arxiv.org/pdf/1905.12418.pdf):

```bibtex
@misc{alsubaihi2019probabilistically,
    title={Probabilistically True and Tight Bounds for Robust Deep Neural Network Training},
    author={Salman Alsubaihi and Adel Bibi and Modar Alfadly and Bernard Ghanem},
    year={2019},
    eprint={1905.12418},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

### License

MIT

### Author

[Modar M. Alfadly](https://modar.me/)

### Contributors

I would gladly accept any pull request that improves any aspect of this repository.