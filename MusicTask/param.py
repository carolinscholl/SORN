""" Music Task

This script contains the parameters for a music experiment.
"""

import os

import numpy as np

import utils
par = utils.Bunch()
aux = utils.Bunch()

################################################################################
#                           SORN main parameters                               #
################################################################################
def get_par():
    """ Get main sorn parameters.

    For each sorn simulation, change these parameters manually.
    """
    par.N_e = 1000                                  # excitatory neurons
    par.N_u = int(par.N_e/100) # int(par.N_e/50)     # neurons in each input pool

    par.eta_stdp = 0.005                           # STDP learning rate
    par.prune_stdp = False                         # prune very small weights
    par.eta_ip = 0.001                             # IP learning rate
    par.h_ip = 0.1                                 # target firing rate

    par.input_gain = 1                             # input gain factor

    par.lamb = int(0.1*par.N_e) #10                    # number of out connections

    par.T_e_max = 0.5                              # max initial threshold for E
    par.T_e_min = 0                                # min initial threshold for E
    par.T_i_max = 0.1 #0.5                             # max initial threshold for I
    par.T_i_min = 0.1 #0                             # min initial threshold for I

################################################################################
#                           Experiment parameters                              #
################################################################################
    par.path_to_music = os.path.join(os.path.dirname(__file__),'..', '..','music_data', '1.mid') # insert path to music here
    #'/Users/carolinscholl/Documents/PhD/rotations/2_triesch/midis/examples/1.mid' # insert path to music here
    par.max_corpus_size = 50000


    par.steps_plastic = 100000                      # sorn training time steps
    par.steps_readout = 30000                    # readout train and test steps
    par.steps_spont = 5000#50000                       # steps of spontaneous generation

################################################################################
#                    Additional derivative SORN parameters                     #
################################################################################
def get_aux():
    """ Get auxiliary sorn parameters.

    These auxiliary parameters do not have to be changed manually.
    """

    aux.N_i = int(0.2*par.N_e)       # inhibitory neurons
    aux.N = par.N_e + aux.N_i             # total number of neurons

    # the experiment_name should be the same name of the directory containing it
    aux.experiment_name = os.path.split(os.path.dirname(\
                                        os.path.realpath(__file__)))[1]
    # training ans testing time steps
    aux.steps_readouttrain = par.steps_readout
    aux.steps_readouttest = par.steps_readout

    aux.N_steps = (par.steps_plastic                # total number of time steps
                   + aux.steps_readouttrain
                   + aux.steps_readouttest)
    aux.readout_steps = (aux.steps_readouttrain     # number of readout steps
                        + aux.steps_readouttest)
