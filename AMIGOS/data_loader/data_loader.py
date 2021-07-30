import numpy as np
import os
import sys
from torch.utils.data import Dataset

sys.path.append('../')

########################################################################################################################

class amigos_cnn_loader(Dataset):
    def __init__(self, param):

        if param.use_predefined_idx == True: # Check whether data is pre-defined
            self.use_pre_idx = True
            return
        else:
            self.use_pre_idx = False

        self.data_path = param.data_path
        target_label = param.target_label
        num_sub = param.num_subject
        num_trial = param.num_trial
        max_num_seq = 1000

        self.eeg_data_filenames = []
        self.eeg_label_filenames = []

        if target_label == 'valence':
            print('----------------valence--------------')
            gt_type = 1
        elif target_label == 'arousal':
            print('----------------arousal--------------')
            gt_type = 2
        elif target_label == 'both':
            print('----------------both--------------')
            gt_type = 3

        for s in range(num_sub):
            if param.target_subject[0] != 0 and not s + 1 in param.target_subject:
                print('%d is not in target subject list' % (s + 1))
                continue
            for v in range(num_trial):
                #  we don't know exact length of each trial. So, if the npy file is not exist, skip to next trial.
                t = 0
                while True:
                    eeg_name = self.data_path + 'S%02dT%02d_%04d.npy' % (s + 1, v + 1, t)
                    if os.path.exists(eeg_name):
                        self.eeg_data_filenames.append(self.data_path + 'S%02dT%02d_%04d.npy' % (s + 1, v + 1, t))
                        if gt_type == 1:
                            self.eeg_label_filenames.append(self.data_path + 'S%02dT%02d_%04d_valence.txt' % (s + 1, v + 1, t))
                        if gt_type == 2:
                            self.eeg_label_filenames.append(self.data_path + 'S%02dT%02d_%04d_arousal.txt' % (s + 1, v + 1, t))
                        t += 1
                    else:
                        # print('End: %d'%t)
                        break

        self.len = len(self.eeg_data_filenames)
        print(self.len)

    # get data in eeg_data files
    def __getitem__(self, index):
        x = np.load(self.eeg_data_filenames[index]).astype(np.float32)
        f = open(self.eeg_label_filenames[index], 'r')
        val = float(f.read().strip())
        if val > 4: # different with original codes!
            y = 1
        else:
            y = 0

        f.close()

        x = x.reshape(-1, x.shape[0], x.shape[1]) # -1 refers to the reshaped array[-1]'s dimension can be reffered automatically
        x = x.astype(np.float32)

        return x, y

    def __len__(self):
        return self.len

########################################################################################################################################