import os
import sys; sys.path.append('.')

from collections import Counter
import pickle
import numpy as np
import matplotlib.pylab as plt

import common.stats
#from .common import stats

# parameters to include in the plot
N = np.array([1000], dtype=np.int)
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# plots the error per pitch for the next predicted symbol (not in the spontaneous phase)

fig = plt.figure(1, figsize=(15, 6))

stats = {}
for n in N:
    experiment_tag = 'MIDIS_N{}_Tp100000_Tr30000'.format(n)
    experiment_folder = 'MusicTask_' + experiment_tag
    experiment_path = 'backup/' + experiment_folder + '/'

    stats[str(n)] = {}
    for exp, exp_name in enumerate(os.listdir(experiment_path)):
        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        stats[str(exp)] = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
        #exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]
        #stats[str(n)][str(exp_n[0])] = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
width = 0.2

for i, (n, scores) in enumerate(stats.items()):
    n_elems = len(scores)
    new_scores = {}
    for j, value in scores['1'].spec_perf.items():
        new_scores[j] = 0
        for exp in range(n_elems):
            new_scores[j] += scores[str(exp+1)].spec_perf[j]/n_elems

    xticks = list(new_scores.keys())
    #import ipdb; ipdb.set_trace()
    plt.bar(np.arange(len(xticks))+(i-1)*width, [x for x in new_scores.values()], width=width,
            alpha=0.7, label='{}'.format(n))

leg = plt.legend(loc='best', frameon=False, fontsize=15)
leg.set_title('Network size',prop={'size':15})
plt.xticks(np.arange(len(xticks)), xticks, fontsize=15)
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
           ['0%', '20%', '40%', '60%', '80%', '100%',], fontsize=15)
plt.ylim([0, 1])
plt.ylabel('correct predictions', fontsize=15)

if SAVE_PLOT:
    plots_dir = 'plots/MusicTask/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'MusicTask_error_pitches.pdf', format='pdf')
plt.show()
