# Export SEED sessions as 200-sample windows.

import numpy as np
import scipy.io

root_path = r'E:\Research\EEGDataSet\SEED1\SEED\seed_save/'
save_path = r'E:\Research\EEGDataSet\SEED1\SEED\/data_cv5fold/'
# i, j, k index subjects, sessions, and trials.
for i in range(15):
    one_subject = []
    one_subject_label = []
    for j in range(3):

        one_session = []
        one_session_label = []
        for k in range(15):
            one_trial = []
            trial_tmp = scipy.io.loadmat(root_path + 'S%d_%d_%d.mat' % (i+1, j+1, k+1))
            trial_data = trial_tmp['trial_data']
            trial_label = np.squeeze(trial_tmp['trial_label'])
            trial_number = np.int32(trial_data.shape[1]/200)
            for tmp_num in range(trial_number):
                one_trial.append(trial_data[:,tmp_num*200:(tmp_num+1)*200])
            one_trial_label = [trial_label]*trial_number
            one_session.append(one_trial)
            one_session_label.append(one_trial_label)

        data = np.array(one_session)
        label = np.array(one_session_label)
        np.save(save_path + 'S%d_session%d.npy'%(i+1, j+1), data)
        np.save(save_path + 'S%d_session%d_label'%(i+1, j+1), label)

        print('Finished Subject%d Session%d' % (i+1, j+1))


