""" Music Generation Source

This script contains a Music source.
Assumes that source is a single midi file
"""

import random

import numpy as np

from common import synapses

import pypianoroll as piano



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
        #self.steps_plastic = params.steps_plastic
        #self.steps_readout = params.steps_readout
        self.path_to_music = params.path_to_music

        multitrack = piano.parse(self.path_to_music)
        assert len(multitrack.tracks) == 1, 'Multiple tracks, need monophonic input'

        # remove trailing silence and binarize (same loudness)
        multitrack.trim_trailing_silence()
        multitrack.binarize()
        multitrack = piano.utilities.downsample(multitrack, 2) # only take every 2nd time step

        singletrack = multitrack.tracks[0]

        p_roll = singletrack.pianoroll
        assert piano.metrics.polyphonic_rate(p_roll) == 0, 'Polyphonic rate above 0, need monophonic input'

        # get most likely key of piece
        kind = ['minor', 'major']
        max_perc = 0
        key = np.zeros(3) # base tone, minor(0)/major(1), percent
        for i in range(12):
            for j in range(2):
                curr_perc = piano.metrics.in_scale_rate(p_roll, key=i, kind=kind[j])
                if curr_perc > max_perc
                    max_perc = curr_perc
                    key[0] = i
                    key[1] = kind[j]
                    key[3] = curr_perc
        print('Key of current piece: ', key[0], kind[int(key[1])], 'percentage: ', key[2])

        self.lowest_pitch = singletrack.get_active_pitch_range()[0]
        self.highest_pitch = singletrack.get_active_pitch_range()[1]

        self.alphabet = list(range(self.lowest_pitch, self.highest_pitch)) # indices for piano roll
        self.alphabet.append(-1) # add this for silence (all zeros)

        self.A = len(self.alphabet) # alphabet size

        indices_non_zero = p_roll.nonzero() # find all time steps were a tone is played
        self.corpus = np.full(p_roll.shape[0], -1) # make a sequence of -1 (stands for silence)
        self.corpus[indices_non_zero[0]] = indices_non_zero[1] # set non_zero time steps with the current index for pitch played

        if len(self.corpus) > params.max_corpus_size:
            self.corpus = self.corpus[:params.max_corpus_size]

        self.N_e = params.N_e
        self.N_u = params.N_u

        # letter counter
        self.ind = -1                # index in the corpus

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

    def index_to_symbol(self, index):
        """
        Convert a MIDI index to the corresponding note

        Arguments:
        index -- int, alphabet index (NOT to sequence index ind)

        Returns:
        symbol -- string corresponding to note
        """

        if index == -1:
            note_as_string = 'silence'
        else:
            octave = (index // len(self.notes))-1
            assert octave in self.octaves, 'Illegal MIDI index'
            assert 0 <= index <= 127, 'Illegal MIDI index'
            note = self.notes[index % len(self.notes)]

            note_as_string = note + str(octave)

        return note_as_string


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
        # this does not really affect FDT because the corpus is much bigger
        # than the number of simulation steps.
        if self.ind == len(self.corpus):
            self.ind = 0

        one_hot = np.zeros(self.A)
        one_hot[self.alphabet.find(self.corpus[self.ind])] = 1
        return one_hot
