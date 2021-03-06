# make segment of subject - trial eeg dataset
import numpy as np
import os
import matplotlib.pyplot as plt

channels = 14
sampling_rate = 128
segment_size = 1      # sec
segment_length = sampling_rate * segment_size

normalization = True
#  normal method: 1 - Z-score / 2 - min-max
normal_method = 1

num_subject = 40
num_trial = 16

base_source_path = '/home/hj/PycharmProjects/EEG/dataset/Amigos/data_preprocessed_python/'
base_target_path = '/home/hj/PycharmProjects/EEG/dataset/Amigos/data_preprocessed_python_1s_norm/'

if not os.path.exists(base_target_path):
    os.makedirs(base_target_path)

for i in range(1, num_subject+1):
    for j in range(1, num_trial+1):
        eeg_file = base_source_path + 'S%02dT%02d.npy'%(i, j)
        label_self_file = base_source_path + 'S%02dT%02d_label_self.txt' % (i, j)

        eegs = np.load(eeg_file)
        labels = np.loadtxt(label_self_file)

        if normalization == True:
            if normal_method  == 1:
                mean = np.transpose(np.mean(eegs, axis=1).reshape(-1, eegs.shape[0]))
                stdev = np.transpose(np.std(eegs, axis=1).reshape(-1, eegs.shape[0]))
                eegs = (eegs - mean) / (stdev * 2)

        # for kk in range(eegs.shape[0]):
        #     plt.plot(eegs[kk, :])
        # plt.show()

        # valence = [labels[0]]
        # arousal = [labels[1]]

        num_segment = int(eegs.shape[1] / segment_length)

        for k in range(num_segment):
            eeg = eegs[:, segment_length*k: segment_length*(k+1)]
            eeg_segment_file = base_target_path + 'S%02dT%02d_%04d.npy'%(i, j, k)
            # label_segment_valence_file = base_target_path + 'S%02dT%02d_%04d_valence.txt' % (i, j, k)
            # label_segment_arousal_file = base_target_path + 'S%02dT%02d_%04d_arousal.txt' % (i, j, k)
            label_segment_arousal_file = base_target_path + 'S%02dT%02d_%04d_arousal.txt' % (i, j, k)


            np.save(eeg_segment_file, eeg)
            # np.savetxt(label_segment_valence_file, valence, delimiter=' ', fmt='%.4f')
            # np.savetxt(label_segment_arousal_file, arousal, delimiter=' ', fmt='%.4f')
            np.savetxt(label_segment_arousal_file, labels, delimiter=' ', fmt='%.4f')
