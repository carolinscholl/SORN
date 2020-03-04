""" LanguageTask experiment

This script contains the experimental instructions for the MusicTask
experiment.
"""

import copy
from collections import Counter
import os
import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

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
        # track: generate some exemplary MIDI files
        self.files_tosave = [
            'params',
            'stats',
            'scripts',
            'track'
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
        sorn.params.par.eta_ip = 'off'
        sorn.simulation(stats, phase='train')

        # Step 3. Train readout layer
        if display:
            print('\nTraining readout layer...')
        t_train = sorn.params.aux.steps_readouttrain
        X_train = stats.raster_readout[:t_train-1]
        n_symbols = sorn.source.A

        if sorn.params.par.input_size == 1: # multi-class mono-label classification (monophony)
            y_train = stats.input_readout[1:t_train].T.astype(int)
            lg = linear_model.LogisticRegression()
            readout_layer = lg.fit(X_train, y_train)
        else: # multi-class multi-label classification (polyphony)
            y_train = stats.input_readout[1:t_train,:]
            #clf = linear_model.RidgeClassifierCV() # https://github.com/scikit-learn/scikit-learn/issues/5562
            #clf = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,100,100,), random_state=1)
            clf = RandomForestClassifier(n_estimators=100)
            readout_layer = clf.fit(X_train, y_train)



        # Step 4. Input without plasticity: test (with STDP and IP off)
        if display:
            print('\nReadout testing phase:')

        sorn.simulation(stats, phase='test')

        # Step 5. Estimate SORN performance
        if display:
            print('\nTesting readout layer...')
        t_test = sorn.params.aux.steps_readouttest
        X_test = stats.raster_readout[t_train:t_train+t_test-1]
        if sorn.params.par.input_size == 1:
            y_test = stats.input_readout[1+t_train:t_train+t_test].T.astype(int)
        else:
            y_test = stats.input_readout[1+t_train:t_train+t_test,:]



        # store the performance for each letter in a dictionary
        if sorn.params.par.input_size == 1:
            spec_perf = {}
            for symbol in np.unique(y_test):
                symbol_pos = np.where(symbol == y_test)
                spec_perf[sorn.source.index_to_symbol(symbol)]=\
                                             readout_layer.score(X_test[symbol_pos],
                                                                 y_test[symbol_pos])
        # save specific performance per letter for plot
            stats.spec_perf = spec_perf

        #else: TODO not yet implemented for polyphony


        # Step 6. Generative SORN with spont_activity (retro feed input)
        if display:
            print('\nSpontaneous phase:')

        # begin with the prediction from the last step
        if sorn.params.par.input_size == 1:
            symbol = readout_layer.predict(X_test[-1].reshape(1,-1))
            u = np.zeros(n_symbols)
            u[symbol] = 1
        else:
            u = readout_layer.predict(X_test[-1,:].reshape(1, -1).astype(np.int8))
            #u = readout_layer.decision_function(X_test[-1,:].reshape(1,-1))
            u = np.squeeze(u).astype(np.bool_)
            #u = (u > 0).astype(int)

        # update sorn and predict next input
        spont_output = ''

        # save symbols for MIDI files
        MIDI_output = np.zeros((sorn.params.par.steps_spont, 128)).astype(np.bool_)

        for _ in range(sorn.params.par.steps_spont):
            sorn.step(u)

            if sorn.params.par.input_size == 1:
                ind = int(readout_layer.predict(sorn.x.reshape(1,-1)))

                one_hot = np.zeros(128) # make one-hot vector
                one = sorn.source.alphabet[ind] # translate SORN index to MIDI index
                if one != -1: # if network did not predict silence
                    one_hot[one] = 1
                MIDI_output[_] = one_hot

                #print(sorn.source.index_to_symbol(ind))
                spont_output += sorn.source.index_to_symbol(ind)+' '
                u = np.zeros(n_symbols)
                u[ind] = 1

            else:
                u = readout_layer.predict(sorn.x.reshape(1, -1).astype(np.int8))
                SORN_states.append(sorn.x.reshape(1,-1))
                #u= readout_layer.decision_function(sorn.x.reshape(1,-1)) # for ridge regression classifier
                #u = (u > 0).astype(int)

                u = np.squeeze(u).astype(np.bool_)

                MIDI_output[_, 21:109] = u

                if np.count_nonzero(u) == 0:
                    spont_output += 'silence, '
                else:
                    nonzeros = u.nonzero()[0] # get indices
                    for i in range(len(nonzeros)):
                        spont_output += sorn.source.midi_index_to_symbol(int(nonzeros[i]+21))+' '

                    spont_output+=',' # add comma to mark next time step

        # Step 7. Generate a MIDI track
        track = piano.Track(MIDI_output)
        track.program = sorn.source.instrument
        track.binarize()
        track = piano.Multitrack(tracks=[track])
        track.beat_resolution = sorn.source.beat_resolution
        #track.tempo = int(sorn.source.tempo)

        # save the MIDI file
        stats.track = track

        stats.output = spont_output   #''.join(output_sentences)
        stats.W_ee = sorn.W_ee.W
        stats.W_ei = sorn.W_ei.W
        stats.W_ie = sorn.W_ie.W
        stats.W_eu = sorn.W_eu.W
        stats.T_e = sorn.T_e
        stats.T_e = sorn.T_e

        # save a few stats about training data
        stats.lowest_pitch = sorn.source.lowest_pitch
        stats.highest_pitch = sorn.source.highest_pitch
        stats.alphabet = sorn.source.alphabet
        stats.input_size = sorn.source.input_size
        stats.beat_resolution = sorn.source.beat_resolution

        # save some storage space by deleting some parameters.
        if hasattr(stats, 'aux'):
            del stats.aux
        if hasattr(stats, 'par'):
            del stats.par


        if display:
            print('\ndone!')
