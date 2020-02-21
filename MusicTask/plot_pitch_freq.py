import os
import sys; sys.path.append('.')

from collections import Counter
import pickle
import numpy as np
import matplotlib.pylab as plt

import common.stats


# parameters to include in the plot
N = np.array([1000, 10000], dtype=np.int)
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(10, 6))

# TODO: only implemented for symbolic MIDI input/output scalar values at the moment
# NEED TO READ IN CORPUS ACCORDING TO SOME VALUES SAVED IN STATS:
# max_corpus_size, beat_resolution etc.
# first assert that they are the same for all models, then use them to generate the corpus


stats = {}
for n in N:
    experiment_tag = 'MIDIS_N{}_Tp100000_Tr30000'.format(n)
    experiment_folder = 'MusicTask_' + experiment_tag
    experiment_path = 'backup/' + experiment_folder + '/'

    stats[str(n)] = {}
    for exp, exp_name in enumerate(os.listdir(experiment_path)):
        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        stats[str(n)][str(exp_n[0])] = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))

    output = []
    for k, v in stats.items():
        output.append(v.output)
    output = ''.join(output)

    output_counter = Counter(output)
    freq_output = np.array([output_counter[x] for x in sorted_chars])
    freq_output = freq_output/freq_output.sum()
