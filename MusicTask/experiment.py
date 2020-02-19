""" LanguageTask experiment

This script contains the experimental instructions for the MusicTask
experiment.
"""

import copy
from collections import Counter

import numpy as np
from sklearn import linear_model

from .source import MusicSource as experiment_source
import pypianoroll as piano

class Experiment:
    """Experiment class.

    It contains the source, the simulation procedure and back-up instructions.
    """
    def __init__(self, params):
        """Start the experiment. Initialize relevant variables and stats
        trackers.

        Arguments:
        params -- Bunch of all sorn inital parameters
        """


        # always keep track of initial sorn parameters
        self.init_params = copy.deepcopy(params.par)
        np.random.seed(42)

        # results directory name
        self.results_dir = (params.aux.experiment_name
                            + params.aux.experiment_tag
                            + '/N' + str(params.par.N_e))

        # define which stats to cache during the simulation
        self.stats_cache = [
            # 'ActivityStat', # number of active neurons
            # 'ActivityReadoutStat', # the total activity only for the readout
            'ConnectionFractionStat', # the fraction of active E-E connections
            'InputReadoutStat', # the input and input index for the readout
            'RasterReadoutStat', # the raster for the readout
        ]

        # define which parameters and files to save at the end of the simulation
        # params: save initial main sorn parameters
        # stats: save all stats trackers
        # scripts: back-up scripts used during the simulation
        # tracks: generate some exemplary MIDI files
        self.files_tosave = [
            'params',
            'stats',
            'scripts',
            'tracks'
        ]

        # create and load input source
        self.inputsource = experiment_source(self.init_params)

    def run(self, sorn, stats):
        """
        Run experiment once and store parameters and variables to save.

        Arguments:
        sorn -- Bunch of all sorn parameters
        stats -- Bunch of stats to save at the end of the simulation
        """

        display = sorn.params.aux.display

        # Step 1. Input with plasticity
        if display:
            print('Plasticity phase:')

        sorn.simulation(stats, phase='plastic')

        # Step 2. Input without plasticity: train (with STDP and IP off)
        if display:
            print('\nReadout training phase:')

        sorn.params.par.eta_stdp = 'off'
        # sorn.params.par.eta_ip = 'off'
        sorn.simulation(stats, phase='train')

        # Step 3. Train readout layer with logistic regression
        if display:
            print('\nTraining readout layer...')
        t_train = sorn.params.aux.steps_readouttrain
        X_train = stats.raster_readout[:t_train-1]
        y_train = stats.input_readout[1:t_train].T.astype(int)
        n_symbols = sorn.source.A
        lg = linear_model.LogisticRegression()
        readout_layer = lg.fit(X_train, y_train)

        # Step 4. Input without plasticity: test (with STDP and IP off)
        if display:
            print('\nReadout testing phase:')

        sorn.simulation(stats, phase='test')

        # Step 5. Estimate SORN performance
        if display:
            print('\nTesting readout layer...')
        t_test = sorn.params.aux.steps_readouttest
        X_test = stats.raster_readout[t_train:t_train+t_test-1]
        y_test = stats.input_readout[1+t_train:t_train+t_test].T.astype(int)

        # store the performance for each letter in a dictionary
        spec_perf = {}
        for symbol in np.unique(y_test):
            symbol_pos = np.where(symbol == y_test)
            spec_perf[sorn.source.index_to_symbol(symbol)]=\
                                         readout_layer.score(X_test[symbol_pos],
                                                             y_test[symbol_pos])

        # Step 6. Generative SORN with spont_activity (retro feed input)
        if display:
            print('\nSpontaneous phase:')

        # begin with the prediction from the last step
        symbol = readout_layer.predict(X_test[-1].reshape(1,-1))
        u = np.zeros(n_symbols)
        u[symbol] = 1

        # update sorn and predict next input
        spont_output = ''

        # save symbols for MIDI files
        MIDI_output = np.zeros((sorn.params.par.steps_spont, 128))

        for _ in range(sorn.params.par.steps_spont):
            sorn.step(u)
            ind = int(readout_layer.predict(sorn.x.reshape(1,-1)))

            one_hot = np.zeros(128) # make one-hot vector
            one = sorn.source.alphabet[ind] # translate SORN index to MIDI index
            if one != -1: # if network did not predict silence
                one_hot[one] = 1
            MIDI_output[_] = one_hot

            #print(sorn.source.index_to_symbol(ind))
            spont_output += sorn.source.index_to_symbol(ind)
            u = np.zeros(n_symbols)
            u[ind] = 1

        # Step 7. Generate a MIDI track
        track = piano.Track(MIDI_output)
        track.program = sorn.source.instrument
        track.binarize()
        track = piano.Multitrack(tracks=[track])
        #path_to_save = self.results_dir + '/sample.mid'

        track.write('sample.mid')

        # Step 8. Calculate parameters to save (exclude first and last sentences
        # and separate sentences by '.'. Also, remove extra spaces.
        #output_sentences = [s[1:]+'.' for s in spont_output.split('.')][1:-1]

        stats.output = spont_output   #''.join(output_sentences)
        stats.W_ee = sorn.W_ee.W
        stats.W_ei = sorn.W_ei.W
        stats.W_ie = sorn.W_ie.W
        stats.W_eu = sorn.W_eu.W
        stats.T_e = sorn.T_e
        stats.T_e = sorn.T_e

        # save some storage space by deleting some parameters.
        if hasattr(stats, 'aux'):
            del stats.aux
        if hasattr(stats, 'par'):
            del stats.par
        # stats.spec_perf = spec_perf

        if display:
            print('\ndone!')


'''
    def generate_midi(self, output, n = 1, time_steps=5000):
    """
    Turn network's output into n MIDI tracks of length time_steps and save them.
    TODO: save generated midis in experiment directory?
    """
    if len(output) / int(n * time_steps) > n :
        n = int(len(output) / int(n * time_steps))
    if n == 0: # output is shorter than time_steps, generate track of length of output
        time_steps = len(output)
        n = 1

    titles = []
    for t in range(n):
        titles.append('sample_track{}.mid'.format(t+1))
        track = np.zeros((time_steps, 128))

        for ts in time_steps:
            one_hot = np.zeros(128)
            one_hot[output[ts]] = 1
            track.append(one_hot)

        # now generate the track object
'''