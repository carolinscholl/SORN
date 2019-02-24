""" Random Sequence parameters

This script contains the parameters for the Random Sequence experiment.
"""

import os

import numpy as np

import utils
par = utils.Bunch()
aux = utils.Bunch()

# ####################
# SORN main parameters
# ####################
def get_par():
    """ Get main sorn parameters.

    For each sorn simulation, change these parameters manually.
    """
    par.N_e = 200                                  # excitatory neurons
    par.N_u = 10                                   # neurons in each input pool

    par.eta_stdp = 0.001                           # STDP learning rate
    par.prune_stdp = True                          # prune very small weights
    par.eta_ip = 0.001                             # IP learning rate
    par.h_ip = 0.1                                 # target firing rate

    # lognormal weights
    # par.h_ip = np.random.lognormal(mean=np.log(0.1), sigma=0.1962, size=par.N_e)

    par.input_gain = 1                             # input gain factor

    par.lamb = 10                                  # number of out connections

    par.T_e_max = 0.5                              # max initial threshold for E
    par.T_e_min = 0                                # min initial threshold for E
    par.T_i_max = 0.5                              # max initial threshold for I
    par.T_i_min = 0                                # min initial threshold for I

# #####################
# Experiment parameters
# #####################
    par.L = 1000000                                # sequence size, must be bigger than par.steps_plastic for a "true" random sequence input
    par.A = 10                                     # alphabet size
    par.task_type = 'FadingMemory'                 # type of task (FadingMemory or LearningCapacity)

    par.steps_plastic = 100000                      # sorn training time steps
    par.steps_readout = 5000                       # readout train and test steps

# #####################################
# Additional derivative SORN parameters
# #####################################
def get_aux():
    """ Get auxiliary sorn parameters.

    These auxiliary parameters do not have to be changed manually.
    """
    aux.N_i = int(np.floor(0.2*par.N_e))       # inhibitory neurons
    aux.N = par.N_e + aux.N_i             # total number of neurons

    # the experiment_name should be the same name of the directory containing it
    aux.experiment_name = os.path.split(os.path.dirname(\
                                        os.path.realpath(__file__)))[1]
    # training ans testing time steps
    aux.steps_readouttrain = 3*par.steps_readout# np.maximum(par.steps_readout, 3*par.L)
    aux.steps_readouttest = par.steps_readout

    aux.N_steps = (par.steps_plastic                # total number of time steps
                   + aux.steps_readouttrain
                   + aux.steps_readouttest)
    aux.readout_steps = (aux.steps_readouttrain     # number of readout steps
                        + aux.steps_readouttest)
