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

        self.input_size = params.input_size
        if self.input_size == 1: # if we use symbolic alphabet of MIDI indices
            assert params.which_alphabet in ['train', 'minmax', 'all']
            self.which_alphabet = params.which_alphabet

        self.tempo = int(120) # BPM
        self.beat_resolution = int(24) # time steps per beat
        self.instrument = int(1) # set the instrument (MIDI ID, 1 is grand piano)

        # set corpus
        # TODO: preprocessing step: transpose all MIDI files to the same key, e.g. C major
        if self.input_size != 1:
            print('Use pianoroll as input, no intermediate conversion to symbolic alphabet.')
            self.generate_pianoroll_music_corpus(params.max_corpus_size)
        else:
            print('Generate intermediate symbolic alphabet for monophonic input.')
            self.generate_MIDIindex_music_corpus(params.max_corpus_size)

        # set alphabet
        self.set_alphabet()
        self.A = len(self.alphabet) # alphabet size
        print('alphabet size: ', self.A)

        self.N_e = params.N_e
        self.N_u = params.N_u

        # time step counter
        self.ind = -1                # index in the corpus


    def set_alphabet(self):
        '''
        Sets the alphabet for a corpus, depending on self.which_alphabet
        '''
        if self.input_size != 1:
            #sets alphabet for corpus of of n-hot arrays.
            #Set alphabet to be all notes that can be played on a grand piano.
            self.alphabet = list(range(21,109)) # sorted(set(indices_non_zero[1]))

            self.lowest_pitch = min(self.alphabet)
            self.highest_pitch = max(self.alphabet)

            print('Lowest midi index, ', self.lowest_pitch)
            print('Highest midi index, ', self.highest_pitch)
            print('Lowest possible pitch:', self.midi_index_to_symbol(self.lowest_pitch))
            print('Highest possible pitch:', self.midi_index_to_symbol(self.highest_pitch))

        else:
            # the corpus consists of converted pianorolls to an intermediate symbolic representation (MIDI indices),
            # we define an alphabet for this
            if self.which_alphabet == 'train': # just the notes that appear in corpus (like in text task)
                self.alphabet = sorted(set(self.corpus))
            elif self.which_alphabet == 'minmax': # notes between lowest and highest pitch in the training data
                min_pitch = self.corpus[np.where(self.corpus > 0, self.corpus, np.inf).argmin()] # need to ignore -1
                max_pitch = self.corpus.max()
                self.alphabet = list(range(min_pitch, max_pitch+1))
                self.alphabet.append(-1)
            elif self.which_alphabet == 'all': # otherwise we assume the alphabet to be all possible notes on a grand piano (MIDI ID 21-108)
                self.alphabet = list(range(21,109))
                self.alphabet.append(-1) # add this for silence (all zeros)

            #print('alphabet:', self.alphabet)

            self.lowest_pitch = self.corpus[np.where(self.corpus > 0, self.corpus, np.inf).argmin()] # have to ignore silence symbol
            self.highest_pitch = self.corpus.max()

            print('Lowest midi index, ', self.lowest_pitch)
            print('Highest midi index, ', self.highest_pitch)
            print('Lowest pitch in training data:', self.midi_index_to_symbol(self.lowest_pitch))
            print('Highest pitch in training data:', self.midi_index_to_symbol(self.highest_pitch))


    def generate_pianoroll_music_corpus(self, max_length):
        '''
        Reads in MIDI files placed in dir self.path_to_music in a pianoroll format
        and generates a corpus array of shape (time steps x 88) or (max_length x 88).
        Corpus consists of pianorolls which are basically concatenated n-hot arrays.
        Like that, polyphonic music is allowed.
        Also sets the dominant key for the corpus.

        Arguments:
        max_length -- maximum length of the corpus
        '''
        self.corpus = np.empty((0, 128)).astype(bool)
        for i in os.listdir(self.path_to_music):
            if i.endswith('.mid') or i.endswith('.midi'):
                multitrack = piano.parse(os.path.join(self.path_to_music, i))
                multitrack.trim_trailing_silence()
                multitrack.binarize()
                multitrack.downsample(int(multitrack.beat_resolution/self.beat_resolution)) # sample down to self.beat_resolution time steps per beat
                #multitrack.tempo = self.tempo
                #multitrack.beat_resolution = self.beat_resolution
                singletrack = multitrack.tracks[0]
                singletrack.program = self.instrument

                p_roll = singletrack.pianoroll
                self.corpus = np.append(self.corpus, p_roll, axis=0)
                if len(self.corpus) > max_length:
                    self.corpus = self.corpus[:max_length]
                    # stop reading in MIDIs if max corpus size is reached
                    break

        # get dominant key in corpus (right now only implemented if we use n-hot arrays for corpus)
        kind = ['minor', 'major']
        max_perc = 0
        key = np.zeros(3) # base tone, minor(0)/major(1), percent
        for i in range(12):
            for j in range(2):
                curr_perc = piano.metrics.in_scale_rate(self.corpus, key=i, kind=kind[j])
                if curr_perc > max_perc:
                    max_perc = curr_perc
                    key[0] = i
                    key[1] = j
                    key[2] = curr_perc
        print('Key of current piece: ', key[0], kind[int(key[1])], 'percentage: ', key[2])
        self.key = key

        self.corpus = self.corpus[:, 21:109] # only select MIDI indices that correspond to notes on a grand piano

    def generate_MIDIindex_music_corpus(self, max_length):
        '''
        Reads in MIDI files placed in dir self.path_to_music in a pianoroll format.
        Generates a symbolic one-dimensional corpus array of length sum of time
        steps of all files or self.max_corpus_size and saves it in self.corpus.
        It is an array of MIDI indices, with -1 corresponding to silence

        Arguments:
        max_length -- maximum length of the corpus
        '''
        sym_sequence = np.array([]).astype(np.int8)
        for i in os.listdir(self.path_to_music):
            if i.endswith('.mid') or i.endswith('.midi'):
                # remove trailing silence and binarize (same loudness)
                multitrack = piano.parse(os.path.join(self.path_to_music, i))
                multitrack.trim_trailing_silence()
                multitrack.binarize()
                multitrack.downsample(int(multitrack.beat_resolution/self.beat_resolution)) # sample down to self_beat_resolution time steps per beat
                #multitrack.tempo = self.tempo
                #multitrack.beat_resolution = self.beat_resolution

                singletrack = multitrack.tracks[0]
                singletrack.program = self.instrument

                p_roll = singletrack.pianoroll
                if piano.metrics.polyphonic_rate(p_roll) > 0: # we only work with monophonic input when we have symbolic alphabet
                    pass
                else:
                    temp = np.full(p_roll.shape[0], -1) # make a sequence of -1 (stands for silence)
                    indices_non_zero = p_roll.nonzero() # find all time steps where a tone is played
                    temp[indices_non_zero[0]] = indices_non_zero[1] # set non_zero time steps with the current index for pitch played
                    sym_sequence = np.append(sym_sequence, temp)
                    if len(sym_sequence) > max_length:
                        break

        self.corpus = np.array(sym_sequence).flatten().astype(np.int8)
        if len(self.corpus) > max_length:
            self.corpus = self.corpus[:max_length]


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
        of a one-hot array

        Returns:
        n_hot -- n-hot array of size A containing the next input symbol
        """

        self.ind += 1

        # in case the sequence ends, restart it
        if self.ind == len(self.corpus):
            self.ind = 0

        if self.input_size != 1:
            # corpus is already sequence of n-hot arrays, we do not need to convert it
            n_hot = self.corpus[self.ind]
        else:
            # convert symbolic MIDI index to an n-hot array
            n_hot = np.zeros(self.A)
            n_hot[self.alphabet.index(self.corpus[self.ind])] = 1
        return n_hot
