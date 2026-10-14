import numpy as np
import scipy.io
import os

root_path = 'E:/Research/EEGDataSet/SEED1/SEED/seed_save/'
save_path = 'E:/Research/EEGDataSet/SEED1/SEED/data_cv5fold/'
os.makedirs(save_path, exist_ok=True)

fixed_length = 177

for i in range(15):
    for j in range(3):
        one_session = []
        one_session_label = []

        for k in range(15):
            trial_tmp = scipy.io.loadmat(root_path + 'S%d_%d_%d.mat' % (i+1, j+1, k+1))
            trial_data = trial_tmp['trial_data']
            trial_label = int(np.squeeze(trial_tmp['trial_label']))

            trial_number = trial_data.shape[1] // 200  # Non-overlapping 200-sample segments.

            if trial_number < fixed_length:
                continue

            # Select fixed_length segments from the center of each trial.
            start_idx = (trial_number - fixed_length) // 2

            one_trial = []
            for tmp_num in range(fixed_length):
                one_trial.append(trial_data[:, (start_idx + tmp_num) * 200 : (start_idx + tmp_num + 1) * 200])

            one_session.append(one_trial)
            one_session_label.append([trial_label] * fixed_length)

        data = np.array(one_session)  # [trials, fixed_length, channels, samples]
        label = np.array(one_session_label)  # [trials, fixed_length]

        np.save(save_path + 'S%d_session%d.npy'%(i+1, j+1), data)
        np.save(save_path + 'S%d_session%d_label.npy'%(i+1, j+1), label)

        print('Finished Subject %d Session %d' % (i+1, j+1))
