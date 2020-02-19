"""This script runs a single sorn simulation for the experiment given as an
argument with the specified parameters and experiment instructions"""

import sys; sys.path.append('.')
from importlib import import_module
from common.sorn import Sorn
from common.stats import Stats
from utils.backup import backup_pickle

################################################################################
#                              SORN simulation                                 #
################################################################################

# 1. load param file
try:
    exp_dir = import_module(sys.argv[1])
except:
    raise ValueError('Please specify a valid experiment name as first argument.')
params = exp_dir.param

# 2. add experiment specific parameters
params.get_par()
params.get_aux()
params.aux.display = True
try:
    tag = sys.argv[2]
except:
    raise ValueError('Please specify a valid experiment tag.')
params.aux.experiment_tag = '_{}'.format(tag)

# 3. initialize experiment, sorn, and stats objects
#    the experiment class keeps a copy of the initial sorn main parameters
experiment = exp_dir.experiment.Experiment(params)
experiment.files_tosave = ['params', 'stats', 'scripts']

sorn = Sorn(params, experiment.inputsource)

stats = Stats(experiment.stats_cache, params)

# 4. run one experiment once
experiment.run(sorn, stats)


# 5. save initial sorn parameters and stats objects
backup_pickle(experiment, stats)
