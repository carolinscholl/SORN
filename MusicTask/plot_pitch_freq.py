import os
import sys; sys.path.append('.')

from collections import Counter
import pickle
import numpy as np
import matplotlib.pylab as plt

import common.stats
import pypianoroll as piano

# parameters to include in the plot
N = np.array([1000], dtype=np.int)
SAVE_PLOT = True

################################################################################
#                             Make plot                                        #
################################################################################

# 0. build figures
fig = plt.figure(1, figsize=(10, 6))

# TODO: only implemented for symbolic MIDI input/output scalar values at the moment

params = {}
stats = {}
max_corpus_size = 0
beat_resolution = 0
for n in N:
    experiment_tag = 'MIDIS_N{}_Tp1000_Tr300'.format(n)
    experiment_folder = 'MusicTask_' + experiment_tag
    experiment_path = 'backup/' + experiment_folder + '/'

    #stats[str(n)] = {}
    #params[str(n)] = {}
    for exp, exp_name in enumerate(os.listdir(experiment_path)):
        exp_n = [int(s) for s in exp_name.split('_') if s.isdigit()]

        stats[str(exp)] = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
        params[str(exp)] = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))

        #stats[str(n)][str(exp_n[0])] = pickle.load(open(experiment_path+exp_name+'/stats.p', 'rb'))
        #params[str(n)][str(exp_n[0])] = pickle.load(open(experiment_path+exp_name+'/init_params.p', 'rb'))#np.load(experiment_path+exp_name+'/init_params.p')

        # first assert that beat resolution and max_corpus_size are the same for all models, then use them to generate the corpus
        if max_corpus_size == 0: # first iteration
            max_corpus_size = params[str(exp)].max_corpus_size #params[str(n)][str(exp_n[0])].max_corpus_size
            beat_resolution = stats[str(exp)].beat_resolution#stats[str(n)][str(exp_n[0])].beat_resolution
        else:
            assert max_corpus_size == params[str(exp)].max_corpus_size#params[str(n)][str(exp_n[0])].max_corpus_size
            assert beat_resolution == stats[str(exp)].beat_resolution#stats[str(n)][str(exp_n[0])].beat_resolution

# now we have to read in the corpus
path_to_music = os.path.join(os.getcwd(), '..','midis/')
sym_sequence = np.array([]).astype(np.int8)
for i in os.listdir(path_to_music):
    if i.endswith('.mid') or i.endswith('.midi'):
        # remove trailing silence and binarize (same loudness)
        multitrack = piano.parse(os.path.join(path_to_music, i))
        multitrack.trim_trailing_silence()
        multitrack.binarize()
        multitrack.downsample(int(multitrack.beat_resolution/beat_resolution)) # sample down to self_beat_resolution time steps per beat
        singletrack = multitrack.tracks[0]

        p_roll = singletrack.pianoroll
        if piano.metrics.polyphonic_rate(p_roll) > 0: # we only work with monophonic input when we have symbolic alphabet
            pass
        else:
            temp = np.full(p_roll.shape[0], -1) # make a sequence of -1 (stands for silence)
            indices_non_zero = p_roll.nonzero() # find all time steps where a tone is played
            temp[indices_non_zero[0]] = indices_non_zero[1] # set non_zero time steps with the current index for pitch played
            sym_sequence = np.append(sym_sequence, temp)
            if len(sym_sequence) > max_corpus_size:
                break

sym_sequence = sym_sequence[:max_corpus_size]
# now we have a sequence of MIDI indidces, need to convert to strings of notes
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] # length of octave
octaves = list(range(11))
corpus = []
for i in range(len(sym_sequence)):
    if sym_sequence[i] == -1:
        note_as_string = 'silence'
    else:
        octave = (sym_sequence[i] // len(notes))-1
        assert octave in octaves, 'Illegal MIDI index'
        assert 0 <= sym_sequence[i] <= 127, 'Illegal MIDI index'
        note = notes[sym_sequence[i] % len(notes)]
        note_as_string = note + str(octave)
    corpus.append(note_as_string)


notes_input = Counter(corpus)
sorted_notes = [x[0] for x in sorted(notes_input.items(), key=lambda kv: kv[1], reverse=True)]
freq_input = np.array([notes_input[x] for x in sorted_notes])
freq_input = freq_input/freq_input.sum()



# now we make the plot
for n in N:
    experiment_tag = 'MIDIS_N{}_Tp1000_Tr300'.format(n)
    experiment_folder = 'MusicTask_' + experiment_tag
    experiment_path = 'backup/' + experiment_folder + '/'

    for exp, exp_name in enumerate(os.listdir(experiment_path)):
        output = stats[str(exp)].output[:-1]
        output=output.split(' ')
        output_counter = Counter(output)
        freq_output = np.array([output_counter[x] for x in sorted_notes])
        freq_output = freq_output/freq_output.sum()

    plt.plot(np.arange(len(freq_input)), freq_output, linewidth='1', label='{}'.format(n))


plt.plot(np.arange(len(freq_input)), freq_input, linewidth='3', color='k', label='MIDIS')
leg = plt.legend(loc='best', frameon=False, fontsize=18)
leg.set_title('Network size',prop={'size':18})
plt.xticks(np.arange(len(sorted_notes)), sorted_notes, fontsize=10, rotation=90)
plt.yticks([0, 0.1, 0.2, 0.3, 0.4],
           ['0%', '10%', '20%', '30%', '40%'], fontsize=18)
plt.ylim([0, 0.4])
plt.ylabel('frequency', fontsize=18)
plt.tight_layout()
if SAVE_PLOT:
    plots_dir = 'plots/MusicTask/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    plt.savefig(plots_dir+'MIDIS_pitch_counts.pdf', format='pdf')
plt.show()
