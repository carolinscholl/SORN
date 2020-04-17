# The Self-Organizing Recurrent Neural Network (SORN_V2)

<p align="center">
  <img src="https://github.com/delpapa/SORN_V2/blob/master/imgs/sorn.png" width="400">

A SORN repository for general purposes, containing a few experiments and examples. This repository is based on the original [SORN repository](https://github.com/chrhartm/SORN) by Christoph Hartmann combined with adaptations to new experiments I did for my PhD thesis. It is also an update of my [old SORN repository](https://github.com/delpapa/SORN) to python 3, combined with a few other optimizations and (slightly) better software maintenance practices.

## Getting started with the repository

The scripts for each experiment are stored in different folders with the experiments' respective name. Each folder contain a minimum of three scripts: the experiment parameters (`param.py`), instructions (`experiment.py`), and the input details (`source.py`). The parameters in these scripts can be modified at will to reproduce various results. Additionally, each folder may contain the relevant plot scripts, for visualizing the results.

To simulate a single experiment, run `python3 common/run_single.py <EXPERIMENT_NAME> <EXPERIMENT_TAG>` from the main folder, in which `<EXPERIMENT_NAME>` must be the name of an experiment module to import (i.e., CountingTask) and `<EXPERIMENT_TAG>` must be a string, to name your particular experiment instance (i.e., test).

Currently implemented experiments:

* CountingTask (from [Lazar et al. 2009](http://journal.frontiersin.org/article/10.3389/neuro.10.023.2009/full))
* RandomSequenceTask (from Del Papa et al. 2019)
* NeuronalAvalanches (from [Del Papa et al. 2017](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178683), Del Papa et al. 2019)
* MemoryAvalanches (from Del Papa et al. 2019)
* GrammarTask (for artificially built dictionaries)
* TextTask (for general language encoding and generation with SORNs)

### Prerequisites

* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikitlearn](http://scikit-learn.org/)
* [matplotlib](https://matplotlib.org/)
* [bunch](https://pypi.python.org/pypi/bunch)
* [powerlaw](https://pypi.python.org/pypi/powerlaw) (for the NeuronalAvalanches and MemoryAvalanche experiments)

### Directory structure

```bash
.
├── backup                         # simulations are backed up in this folder
├── common                         # contains main simulation scripts and model classes
│   ├── run_single.py              # single run of a particular simulation
│   ├── sorn.py                    # main sorn class and update methods
│   ├── stats.py                   # stats tracker
│   ├── synapses.py                # script with all the functions relating to weights and weight updates
├── CountingTask                   # scripts for the CountingTask (all other experiments should follow this example)
│   ├── experiment.py              # experiment instructions
│   ├── param.py                   # experiment parameters
│   ├── plot_performance.py        # plot script (convention: start with 'plot_')
│   └── source.py                  # script containing the input source for this particular task
├── GrammarTask
├── MemoryAvalanche
├── NeuronalAvalanches
├── RandomSequenceTask
├── TextTask
├── README.md
├── LICENSE
├── requirements.txt               # requirements to build the python environment
└── utils                          # contains bunch and the backup functions
    ├── backup.py
    └── bunch
```

### Installation guide

#### Forking the repository

Fork a copy of this repository onto your own GitHub account and `clone` your fork of the repository into your computer, inside your favorite SORN folder, using:

`git clone "PATH_TO_FORKED_REPOSITORY"`

#### Setting up the environment

Install [Python 3.6](https://www.python.org/downloads/release/python-360/) and the [conda package manager](https://conda.io/miniconda.html) (use miniconda). Navigate to the project directory inside a terminal and create a virtual environment (replace <ENVIRONMENT_NAME>, for example, with `sorn_env`) and install the [required packages](https://github.com/delpapa/SORN_V2/blob/master/requirements.txt):

`conda create -n <ENVIRONMENT_NAME> --file requirements.txt python=3.6`

Activate the virtual environment:

`source activate <ENVIRONMENT_NAME>`

By installing these packages in a virtual environment, you avoid dependency clashes with other packages that may already be installed elsewhere on your computer.

## Experiments

Each different experiment has it's own project folder (for example, `CountingTask`), which contains it's parameters, experiment instructions, sources and plot files. If you plan to extend the repository with new experiments, please keep this structure to avoid unnecessary conflicts. For details on the implementation of each experiment, please have a look at my thesis ~~here~~ (link to be added as soon as it is made available).

### CountingTask

This task is the reimplementation of the counting task from the original SORN paper by [Lazar et al. 2009](http://journal.frontiersin.org/article/10.3389/neuro.10.023.2009/full). The task consist of randomly alternating sequences of the form 'ABB..BBC' and 'DEE..EEF', with size L = n+2, that are repeatedly presented to the network as input. The model evolves due to plasticity action (STDP, IP, and SN), and a readout layer is trained to evaluate its performance.

### GrammarTask

This task evaluates the SORNs performance on predefined artificial dictionaries (see the source script for details). It also simulates the models in an autonomous phase, in which it generates sentences via a retro-feedback loop, without external inputs.

### MemoryAvalanches

This task is a variation of NeuronalAvalanches, in which the fading memory was calculated for different dynamical regimes, driven by the level of neuronal membrane noise. Spoilers: fading memory is improved at the noise level that also results in power-law distributed neuronal avalanches.

### NeuronalAvalanches

This task is the implementation of the neuronal avalanche analysis in SORNs from [Del Papa et al. 2017](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0178683). It simulates the model with additional plasticity mechanisms (STDP, IP, SN, iSTDP, and SP) and measures the distributions of avalanches in its spontaneous or evokes activity, depending on the source.

### RandomSequenceTask

This task implements a random sequence input, which was used to estimate the SORN's fading memory. The random sequence input consists of a sequence of A symbols, and the fading memory capacity is the SORN's capacity of recovering past inputs with its linear readout layer.

### TextTask

This task uses the SORN as a generative model for texts: it trains the model on a given text file and generates autonomous inputs after the input is cut-off (i.e., the network receives its own output as input). 

### MusicTask

Analogous to the text task, SORN is used as a generative model for sequences. It can be trained on monophonic or polyphonic MIDI tracks and eventually generates a spontaneous sequence of music (MIDI indices), using its own output as input at the next time step. A sample MIDI file can be generated from this spontaneous output. This task uses the package [pypianoroll](https://salu133445.github.io/pypianoroll/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
