import numpy as np
import random

from common import synapses

class RandomSequenceSource(object):
    """
    Random Sequence source
    """
    def __init__(self, params):

        self.L = params.L                        # sequence lenght
        self.N_a = params.A                      # alphabet size
        self.N_u = int(params.N_u)               # input pool size

        # generate sequence (make sure it contains all symbols!)
        # TODO: there must be a better way to do this?
        contain_all = 0
        while not contain_all:
            self.sequence = np.random.randint(self.N_a, size=self.L)
            for elem in range(self.N_a):
                if (self.sequence == elem).sum() == 0:
                    break
                else:
                    contain_all = 1

        # letter and word counters
        self.symbol = self.sequence[0]           # return symbol index
        self.sequence_index = 1                  # index for sequences
        self.ind = 0                    # index within sequence

    def generate_connection_e(self, N_e):
        """
        Generate the W_eu connection matrix

        Parameters:
            N_e: number of excitatory neurons
        """

        # choose random, overlapping input neuron pools
        W = np.zeros((N_e, self.N_a))
        available = set(range(N_e))
        for a in range(self.N_a):
            temp = random.sample(available, self.N_u)
            W[temp, a] = 1

        # always use a full synaptic matrix
        ans = synapses.FullSynapticMatrix((N_e, self.N_a))
        ans.W = W

        return ans

    def symbol(self):
        """
        Return symbol index
        """
        return self.symbol

    def sequence_ind(self):

        return self.ind

    def next_sequence(self):
        """
        Restart the sequence
        """
        self.ind = 0
        self.sequence_index += 1
        self.symbol = self.sequence[self.ind]

    def next(self):
        """
        Return next symbol or first symbol of sequence
        """

        if self.ind >= self.sequence.size-1:
            self.next_sequence()

        else:
            self.ind += 1
            self.symbol = self.sequence[self.ind]

        ans = np.zeros(self.N_a)
        ans[self.symbol] = 1
        return ans