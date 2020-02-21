""" Music Generation Source

This script contains a Music source.
Assumes that source is a single MIDI file
"""

import random

import numpy as np

from common import synapses

import pypianoroll as piano

import os

class MusicSource(object):
    """
    """

    def __init__(self, params):
        """
        Initialize music source.

        Arguments:
        params -- bunch of simulation parameters from param.py
        """

        self.notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] # length of octave
        self.octaves = list(range(11))

        # self.file_path= params.file_path
        self.steps_plastic = params.steps_plastic
        self.steps_readout = params.steps_readout
        self.path_to_music = params.path_to_music
        self.which_alphabet = params.which_alphabet
        self.tempo = 120 # BPM
        self.beat_resolution = 24 # time steps per beat
        self.instrument = 1 # set the instrument (MIDI ID, 1 is grand piano)

        # set corpus
        generate_music_corpus_from_monophonic_MIDI_files()

        # set the alphabet
        if self.which_alphabet == 0: # just the notes that appear in corpus (like in text task)
            self.alphabet = sorted(set(self.corpus))
        elif self.which_alphabet == 1: # notes between lowest and highest pitch in the training data
            min_pitch = self.corpus[np.where(self.corpus > 0, self.corpus, np.inf).argmin()] # need to ignore -1
            max_pitch = self.corpus.max()
            self.alphabet = list(range(min_pitch, max_pitch+1))
            self.alphabet.append(-1)
        else: # otherwise we assume the alphabet to be all possible notes on a grand piano (MIDI ID 21-108)
            self.alphabet = list(range(21,109))
            self.alphabet.append(-1) # add this for silence (all zeros)

        self.A = len(self.alphabet) # alphabet size
        print('alphabet size: ', self.A)
        #print('alphabet:', self.alphabet)

        self.lowest_pitch = self.corpus[np.where(self.corpus > 0, self.corpus, np.inf).argmin()]
        self.highest_pitch = self.corpus.max()

        print('Lowest midi index, ', self.lowest_pitch)
        print('Highest midi index, ', self.highest_pitch)
        print('Lowest pitch in training data:', self.midi_index_to_symbol(self.lowest_pitch))
        print('Highest pitch in training data:', self.midi_index_to_symbol(self.highest_pitch))

        self.N_e = params.N_e
        self.N_u = params.N_u

        # time step counter
        self.ind = -1                # index in the corpus

    def generate_music_corpus_from_monophonic_MIDI_files(self):
        '''
        Reads in MIDI files placed in dir self.path_to_music in a pianoroll format.
        Generates a symbolic one-dimensional corpus array of length sum of time
        steps of all files or self.max_corpus_size and saves it in self.corpus
        '''
        sym_sequence = np.array([]).astype(np.int8)
        curr_alphabet = {}
        for i in os.listdir(self.path_to_music):
            if i.endswith('.mid') or i.endswith('.midi'):
                # remove trailing silence and binarize (same loudness)
                multitrack = piano.parse(os.path.join(self.path_to_music, i))
                multitrack.trim_trailing_silence()
                multitrack.binarize()
                #multitrack.downsample(int(multitrack.beat_resolution/self.beat_resolution)) # sample down to self_beat_resolution time steps per beat
                #multitrack.tempo = self.tempo
                #multitrack.beat_resolution = self.beat_resolution

                singletrack = multitrack.tracks[0]
                singletrack.instrument = self.instrument #singletrack.program

                p_roll = singletrack.pianoroll
                if piano.metrics.polyphonic_rate(p_roll) > 0: # we only work with monophonic input for now
                    pass
                else:
                    temp = np.full(p_roll.shape[0], -1) # make a sequence of -1 (stands for silence)
                    indices_non_zero = p_roll.nonzero() # find all time steps where a tone is played
                    temp[indices_non_zero[0]] = indices_non_zero[1] # set non_zero time steps with the current index for pitch played
                    sym_sequence = np.append(sym_sequence, temp)
                    if len(sym_sequence) > params.max_corpus_size:
                        break

        self.corpus = np.array(sym_sequence).flatten().astype(np.int8)
        if len(self.corpus) > params.max_corpus_size:
            self.corpus = self.corpus[:params.max_corpus_size]

        '''
        # get dominant key in corpus, right now only implemented if we use just one piece as input
        kind = ['minor', 'major']
        max_perc = 0
        key = np.zeros(3) # base tone, minor(0)/major(1), percent
        for i in range(12):
            for j in range(2):
                curr_perc = piano.metrics.in_scale_rate(p_roll, key=i, kind=kind[j])
                if curr_perc > max_perc:
                    max_perc = curr_perc
                    key[0] = i
                    key[1] = j
                    key[2] = curr_perc
        print('Key of current piece: ', key[0], kind[int(key[1])], 'percentage: ', key[2])
        self.key = key
        '''


    def generate_connection_e(self, params):
        """
        Generate the W_eu connection matrix. TODO: params should not be passed
        as an argument again!

        Arguments:
        params -- bunch of simulation parameters from param.py

        Returns:
        W_eu -- FullSynapticMatrix of size (N_e, A), containing 1s and 0s
        """

        # choose random, non-overlapping input neuron pools
        W = np.zeros((self.N_e, self.A))
        available = set(range(self.N_e))
        for a in range(self.A):
            temp = random.sample(available, self.N_u)
            W[temp, a] = 1
            available = set([n for n in available if n not in temp])
            assert len(available) > 0,\
                   'Input alphabet too big for non-overlapping neurons'
        # always use a full synaptic matrix
        ans = synapses.FullSynapticMatrix(params, (self.N_e, self.A))
        ans.W = W

        return ans

    def sequence_ind(self):
        """
        Return the sequence index. The index is the current position in the
        input (here, the whole corpus is considered a sequence). TODO: the names
        'ind' and 'index' are confusing, improve their names.

        Returns:
        ind -- sequence (corpus) index of the current input
        """

        ind = self.ind
        return ind

    def midi_index_to_symbol(self, index):
        '''
        Convert MIDI index to a string of the note

        Arguments:
        index -- int, MIDI index from 0-127 or -1 for silence

        Returns:
        string -- string corresponding to a note
        '''
        if index == -1:
            note_as_string = 'silence'
        else:
            octave = (index // len(self.notes))-1
            assert octave in self.octaves, 'Illegal MIDI index'
            assert 0 <= index <= 127, 'Illegal MIDI index'
            note = self.notes[index % len(self.notes)]

            note_as_string = note + str(octave)

        return note_as_string

    def index_to_symbol(self, index):
        """
        Convert an alphabet index (from the network) to the corresponding note

        Arguments:
        index -- int, alphabet index

        Returns:
        symbol -- string corresponding to note
        """

        # first get the MIDI index
        midi = self.alphabet[index]
        return self.midi_index_to_symbol(midi)

    def next(self):
        """
        Update current index and return next symbol of the corpus, in the form
        of a one-hot array (from Christoph's implementation). TODO: this is very
        ugly, there must be a better way to do implement this.

        Returns:
        one_hot -- one-hot array of size A containing the next input symbol
        """

        self.ind += 1

        # in case the sequence ends, restart it
        if self.ind == len(self.corpus):
            self.ind = 0

        one_hot = np.zeros(self.A)
        one_hot[self.alphabet.index(self.corpus[self.ind])] = 1
        return one_hot
