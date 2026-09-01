import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import scipy.io
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
from sklearn import manifold
from dataloader import *
from torch.utils.data import TensorDataset, DataLoader
import pickle
import os

def z_score(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z = (mean - X) / (std+0.000000001)

    return z, mean, std

def normalize(X, mean, std):
    z = (mean - X) / (std+0.0000001)
    return z

def one_hot(y, n_cls):
    y_new = []
    y = np.array(y, 'int32')
    for i in range(len(y)):
        target = [0] * n_cls
        target[y[i]] = 1
        y_new.append(target)
    return np.array(y_new, 'int32')

# Obtaining TRAIN and TEST from DATA
def split_data(X, Y, seed, test_size=0.3):

    s = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in s.split(X, Y):
        X_tr, X_ts = X[train_index], X[test_index]
        Y_tr, Y_ts = Y[train_index], Y[test_index]

    return X_tr, Y_tr, X_ts, Y_ts



# dataset definition
class PseudoLabeledData(Dataset):
    # load the dataset
    def __init__(self, X, Y, W):
        self.X = torch.Tensor(X).float()
        self.Y = torch.Tensor(Y).long()
        # weights
        self.W = torch.Tensor(W).float()

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.Y[idx], self.W[idx]]
    

def load_seed(args, path, session="all", feature="LDS", n_samples=185):
    import scipy.io
    import numpy as np

    session1 = ["1_20131027", "2_20140404", "3_20140603", "4_20140621", "5_20140411", 
                "6_20130712", "7_20131027", "8_20140511", "9_20140620", "10_20131130", 
                "11_20140618", "12_20131127", "13_20140527", "14_20140601", "15_20130709"]
    
    session2 = ["1_20131030", "2_20140413", "3_20140611", "4_20140702", "5_20140418", 
                "6_20131016", "7_20131030", "8_20140514", "9_20140627", "10_20131204",  
                "11_20140625", "12_20131201", "13_20140603", "14_20140615", "15_20131016"]
    
    session3 = ["1_20131107", "2_20140419", "3_20140629", "4_20140705", "5_20140506", 
                "6_20131113", "7_20131106", "8_20140521", "9_20140704", "10_20131211",
                "11_20140630", "12_20131207", "13_20140610", "14_20140627", "15_20131105"]

    if session == 1:
        x_session = session1
    elif session == 2:
        x_session = session2
    elif session == 3:
        x_session = session3
    else:
        raise ValueError("Session must be 1, 2, or 3")

    # Load labels
    y_session = scipy.io.loadmat(path + "label.mat", mat_dtype=True)["label"][0]
    y_session = y_session + 1  # To [1, 2, 3]
    
    X_subjects = {}
    Y_subjects = {}
    
    for subj_idx, subj in enumerate(x_session):
        print("Subject load:", subj)
        dataMat = scipy.io.loadmat(path + subj + ".mat", mat_dtype=True)

        subj_X_list = []
        subj_Y_list = []
        
        for trial_idx in range(15):
            features = dataMat[feature + str(trial_idx + 1)]  # shape: (T, 62, 5)
            features = np.swapaxes(features, 0, 1)  # shape: (T, 62, 5)

            if features.shape[0] > n_samples:
                features = features[-n_samples:]  # keep last n samples

            # Sliding window
            window_size = 12
            temp_feats = [np.expand_dims(features[i:i + window_size], axis=0) 
                          for i in range(len(features) - window_size + 1)]
            temp_feats = np.concatenate(temp_feats, axis=0)  # shape: (N, 12, 62, 5)

            labels = np.array([y_session[trial_idx]] * temp_feats.shape[0])
            subj_X_list.append(temp_feats)
            subj_Y_list.append(labels)
        
        # Once per subject
        X_subjects[subj_idx] = np.concatenate(subj_X_list, axis=0)
        Y_subjects[subj_idx] = np.concatenate(subj_Y_list, axis=0)
        print(f"Subject {subj_idx+1}: {X_subjects[subj_idx].shape}, Labels: {Y_subjects[subj_idx].shape}")
    
    trg_subj = args.target - 1
    Tx = X_subjects[trg_subj]
    Ty = Y_subjects[trg_subj]
    Tx, m, std = z_score(Tx)

    # Train loader
    train_loader = UnalignedDataLoader()
    train_loader.initialize(len(x_session), X_subjects, Y_subjects, Tx, Ty, trg_subj,
                            args.batch_size, args.batch_size,
                            shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()

    # Test loader
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()

    return datasets, dataset_test, X_subjects, Y_subjects




def load_seed_raw(args, path, session="all", feature="LDS", n_samples=185):
    """
    SEED I
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials. 
    In one trial, the participant watch one of the film clips, while his(her) EEG 
    signals and eye movements are collected with the 62-channel ESI NeuroScan System 
    and SMI eye-tracking glasses.
    """
    
    

         
    
    
    # Load samples
    samples_by_subject = 0
    X = []
    Y = []
    flag = False
    X_subjects = {}
    Y_subjects = {}
    n = 15*185
    r = 0
    for subj in range(15):
        save_dir_data = f"/home/lyc/research/research_6/data_cv5fold/S{subj+1}_session{session}.npy"
        save_dir_label = f"/home/lyc/research/research_6/data_cv5fold//S{subj+1}_session{session}_label.npy"
        dataMat = np.load(save_dir_data, allow_pickle=True)
        labelMat = np.load(save_dir_label, allow_pickle=True)
        print("Subject load:", subj)

        trial_data = dataMat.reshape(-1, 62, 200)
        trial_label = np.squeeze(labelMat.reshape(-1))
        trial_label[trial_label == -1] = 2

        X_subjects[subj] = trial_data
        Y_subjects[subj] = trial_label
        # increment rang
        print(X_subjects[subj].shape)
  
    trg_subj = args.target - 1
    Tx = np.array(X_subjects[trg_subj])
    Ty = np.array(Y_subjects[trg_subj])   
    subject_ids = X_subjects.keys()
    num_domains = len(subject_ids)
    Tx, m, std = z_score(Tx)    
    # Train dataset
    train_loader = UnalignedDataLoader()
    train_loader.initialize(num_domains, X_subjects, Y_subjects, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()
    #classes = np.unique(Ty)
    # Test dataset
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()
    
    return datasets, dataset_test



def load_seed_three_feature(args, path, session="all", feature="LDS", n_samples=185):
    """
    SEED I
    A total number of 15 subjects participated the experiment. For each participant,
    3 sessions are performed on different days, and each session contains 24 trials. 
    In one trial, the participant watch one of the film clips, while his(her) EEG 
    signals and eye movements are collected with the 62-channel ESI NeuroScan System 
    and SMI eye-tracking glasses.
    """
    
    
    session1 = [
        "1_20131027",
        "2_20140404", 
        "3_20140603", 
        "4_20140621", 
        "5_20140411", 
        "6_20130712", 
        "7_20131027",
        "8_20140511",
        "9_20140620",
        "10_20131130", 
        "11_20140618",
        "12_20131127",
        "13_20140527", 
        "14_20140601", 
        "15_20130709"
        ]
        
    session2 = [
        "1_20131030", 
        "2_20140413", 
        "3_20140611", 
        "4_20140702",
        "5_20140418",  
        "6_20131016", 
        "7_20131030", 
        "8_20140514", 
        "9_20140627", 
        "10_20131204",  
        "11_20140625",
        "12_20131201", 
        "13_20140603", 
        "14_20140615",
        "15_20131016",
        ]
        
    # SESSION 3
    
    session3 = [
        "1_20131107",
        "2_20140419",
        "3_20140629",
        "4_20140705",
        "5_20140506", 
        "6_20131113",
        "7_20131106",
        "8_20140521",
        "9_20140704",
        "10_20131211",
        "11_20140630",
        "12_20131207",
        "13_20140610", 
        "14_20140627",
        "15_20131105"
        ]
        
    feature_2 = 'psd_LDS'
    feature_3 = 'PLV'    

    # LABELS
    labels = scipy.io.loadmat(path + "label.mat", mat_dtype=True)
    y_session = labels["label"][0]
    # relabel to neural networks [0,1,2]
    for i in range(len(y_session)):
        y_session[i] += 1
    print(y_session)
    
    # select session
    if session == 1:
        x_session = session1
    elif session == 2:
        x_session = session2
    elif session == 3:
        x_session = session3
    
    # Load samples
    samples_by_subject = 0
    X = []
    plv = []
    Y = []
    flag = False
    contact = False
    index = 0
    for subj in x_session:
        # load data .mat
        dataMat = scipy.io.loadmat(path + subj + ".mat", mat_dtype=True)
        plv_dir_data = f"/home/lyc/dataset/seed_plv//S{index+1}_session{session}_plv.npy"
        plvMat = np.load(plv_dir_data, allow_pickle=True)        
        # psdMat = scipy.io.loadmat(path + subj + "_PSD.mat", mat_dtype=True)
        print("Subject load:", subj)
        plv_feature = plvMat.reshape(-1,62,5)
        # print("plv_feature:", plv_feature.shape)
        if contact == 0:
                plv = plv_feature 
                contact = True
        else:
                plv = np.concatenate((plv, plv_feature), axis=0)
               
        # print("plv:", plv.shape)
        index += 1
        for i in range(15):

            # "Differential_entropy (DE)"
            #   62 channels
            #   42 epochs
            #   5 frequency band
            features = dataMat[feature+str(i+1)]
            PSD_feature = dataMat[feature_2+str(i+1)]
            # [1D]
            features = np.swapaxes(features, 0, 1)
            PSD_feature = np.swapaxes(PSD_feature, 0, 1)
            # [select last 'n_samples' samples]
            if (features.shape[0] - n_samples) > 0:
                pos = features.shape[0] - n_samples
                features = features[pos:]
                PSD_feature = PSD_feature[pos:]


            # [Build temporal samples]
            # + ++ + + + + + + + + ++  +

            # feats = features
            # window_size = 9
            # temp_feats = None
            # b = False
            # for a in range(len(feats) - window_size + 1):
            #     f = feats[a:a+window_size]
            #     f = np.expand_dims(f, axis=0)
            #     if not b:
            #         temp_feats = f
            #         b = True
            #     else:
            #         temp_feats = np.concatenate((temp_feats, f), axis=0)
            features = np.stack([features, PSD_feature], axis=1)
            # ++ + ++ + + + + + ++ +

            # set labels for each epoch
            labels = np.array([y_session[i]] * features.shape[0])
            # print("labels:", labels.shape)
            # print("features:", features.shape)
            
            # add to arrays
            if flag == 0:
                X = features
                Y = labels
                flag = True
            else:
                X = np.concatenate((X, features), axis=0)
                Y = np.concatenate((Y, labels), axis=0)
        
        if samples_by_subject == 0:
            samples_by_subject = len(X)
    print("X:", X.shape)
    print("plv:", plv.shape)
    plv = plv[:, np.newaxis, :, :]  # shape: [100, 1, 62, 5]
    # plv = plv[:, np.newaxis, :, :]  # shape: [100, 1, 62, 5]
    X = np.concatenate([X, plv], axis=1)  # shape: [100, 3, 62, 5]
    zero_pad = np.zeros((X.shape[0], 1, 62, 5), dtype=X.dtype)
    X = np.concatenate([X, zero_pad], axis=1)
    # reorder data by subject
    X_subjects = {}
    Y_subjects = {}
    n = samples_by_subject
    r = 0
    for subj in range(len(x_session)):
        X_subjects[subj] = X[r:r+n]
        Y_subjects[subj] = Y[r:r+n]
        # increment range
        r += n
        print(X_subjects[subj].shape)
    trg_subj = args.target - 1
    Tx = np.array(X_subjects[trg_subj])
    Ty = np.array(Y_subjects[trg_subj])   
    subject_ids = X_subjects.keys()
    num_domains = len(subject_ids)
    Tx, m, std = z_score(Tx)    
    # Train dataset
    train_loader = UnalignedDataLoader()
    train_loader.initialize(num_domains, X_subjects, Y_subjects, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()
    #classes = np.unique(Ty)
    # Test dataset
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()
    
    return datasets, dataset_test, X_subjects, Y_subjects

def load_seedv(args, path, session="all", feature="LDS", n_samples=185):
    import scipy.io
    import numpy as np

    session1 = ["1_123", "2_123", "3_123", "4_123", "5_123", 
                "6_123", "7_123", "8_123", "9_123", "10_123", 
                "11_123", "12_123", "13_123", "14_123", "15_123", "16_123"]


    if session == 1:
        x_session = session1
    elif session == 2:
        x_session = session1
    elif session == 3:
        x_session = session1
    else:
        raise ValueError("Session must be 1, 2, or 3")

    X_subjects = {}
    Y_subjects = {}
    
    for subj_idx, subj in enumerate(x_session):
        print("Subject load:", subj)
        dataMat = np.load(path + subj + ".npz", allow_pickle=True)
    
        subj_X_list = []
        subj_Y_list = []
        data_bytes = dataMat["data"].item()
        label_bytes = dataMat["label"].item()
        # 用 pickle 反序列化
        data = pickle.loads(data_bytes)
        label = pickle.loads(label_bytes)
        concat_fea = []
        concat_label = []
        first = False
        for trial_idx in range(15,30): #(30,45)
            feature = data[trial_idx]  # shape: (T, 62, 5)
            one_label = label[trial_idx]
            if first is False:
                concat_fea = feature
                concat_label = one_label
                first = True
            else:
                concat_fea = np.concatenate((concat_fea, feature), axis=0)
                concat_label = np.concatenate((concat_label, one_label), axis=0)
            
        
        # Once per subject
        X_subjects[subj_idx] = concat_fea
        Y_subjects[subj_idx] = concat_label
        print(f"Subject {subj_idx+1}: {X_subjects[subj_idx].shape}, Labels: {Y_subjects[subj_idx].shape}")
    
    trg_subj = args.target - 1
    Tx = X_subjects[trg_subj]
    Ty = Y_subjects[trg_subj]
    Tx, m, std = z_score(Tx)

    # Train loader
    train_loader = UnalignedDataLoader()
    train_loader.initialize(len(x_session), X_subjects, Y_subjects, Tx, Ty, trg_subj,
                            args.batch_size, args.batch_size,
                            shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()

    # Test loader
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()

    return datasets, dataset_test, X_subjects, Y_subjects

def fine_tuning_load_XY(args, X, Y):
    dset_loaders = {}

    if args.dataset in ["seed", "seed-iv", "bnci2014012", "mi"]:
        print("DATA:", args.dataset)

        if hasattr(args, "session") and args.dataset in ["seed", "seed-iv"]:
            print("SESSION:", args.session)

        subjects = X.keys()
        print(subjects)

        Sx = Sy = None
        i = 0
        flag = False
        selected_subject = args.target - 1
        trg_subj = -1

        for s in subjects:
            if i != selected_subject:
                tr_x = np.array(X[s])
                tr_y = np.array(Y[s])

                # 每个源受试者单独做 z-score，保持你原逻辑不变
                tr_x, m, std = z_score(tr_x)

                if not flag:
                    Sx = tr_x
                    Sy = tr_y
                    flag = True
                else:
                    Sx = np.concatenate((Sx, tr_x), axis=0)
                    Sy = np.concatenate((Sy, tr_y), axis=0)
            else:
                trg_subj = s
            i += 1

        print("[+] Target subject:", trg_subj)

        # 目标受试者
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # 这里保持和你原来一样：
        # Tx 用来估计目标域均值方差
        # Vx 用同样统计量归一化，作为 test
        Vx = Tx.copy()
        Vy = Ty.copy()

        Tx, m, sd = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=sd)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Vx.shape, "Ty_test:", Vy.shape)

        # tensor
        Sx_tensor = torch.tensor(Sx, dtype=torch.float32)
        Sy_tensor = torch.tensor(Sy, dtype=torch.long)

        Tx_tensor = torch.tensor(Tx, dtype=torch.float32)
        Ty_tensor = torch.tensor(Ty, dtype=torch.long)

        Vx_tensor = torch.tensor(Vx, dtype=torch.float32)
        Vy_tensor = torch.tensor(Vy, dtype=torch.long)

        # dataset
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)
        # Keep target labels in the held-out loader only; optimization sees Tx.
        target_tr = TensorDataset(Tx_tensor)
        target_ts = TensorDataset(Vx_tensor, Vy_tensor)

        # dataloader
        dset_loaders["source"] = DataLoader(
            source_tr,
            batch_size=args.batch_size_fine,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        dset_loaders["target"] = DataLoader(
            target_tr,
            batch_size=args.batch_size_fine,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        dset_loaders["test"] = DataLoader(
            target_ts,
            batch_size=200,
            shuffle=False,
            num_workers=0
        )

        print("Data were succesfully loaded")

    else:
        print("This dataset does not exist.")
        exit()

    return dset_loaders


def load_seedveye(args, path, session="all", feature="LDS", n_samples=185):
    import scipy.io
    import numpy as np

    session1 = ["1_123", "2_123", "3_123", "4_123", "5_123", 
                "6_123", "7_123", "8_123", "9_123", "10_123", 
                "11_123", "12_123", "13_123", "14_123", "15_123", "16_123"]


    if session == 1:
        x_session = session1
    elif session == 2:
        x_session = session1
    elif session == 3:
        x_session = session1
    else:
        raise ValueError("Session must be 1, 2, or 3")

    X_subjects = {}
    Y_subjects = {}
    eyepath = "E:\Research\EEGDataSet\SEED-V\Eye_movement_features/"
    for subj_idx, subj in enumerate(x_session):
        print("Subject load:", subj)
        dataMat = np.load(path + subj + ".npz", allow_pickle=True)
        eyedataMat = np.load(eyepath + subj + ".npz", allow_pickle=True)
        subj_X_list = []
        subj_Y_list = []
        data_bytes = dataMat["data"].item()
        label_bytes = dataMat["label"].item()
        eyedata_bytes = eyedataMat["data"].item()
        eyelabel_bytes = eyedataMat["label"].item()
        # 用 pickle 反序列化
        data = pickle.loads(data_bytes)
        label = pickle.loads(label_bytes)
        eyedata = pickle.loads(eyedata_bytes)
        eyelabel = pickle.loads(eyelabel_bytes)
        concat_fea = []
        concat_label = []
        first = False
        for trial_idx in range(15,30): #(30,45)15,30
            feature = data[trial_idx]  # shape: (T, 62, 5)
            eye_fea = eyedata[trial_idx]
            one_label = label[trial_idx]
            feature = np.concatenate((feature, eye_fea), axis=1)
            if first is False:
                concat_fea = feature
                concat_label = one_label
                first = True
            else:
                concat_fea = np.concatenate((concat_fea, feature), axis=0)
                concat_label = np.concatenate((concat_label, one_label), axis=0)
            
        
        # Once per subject
        X_subjects[subj_idx] = concat_fea
        Y_subjects[subj_idx] = concat_label
        print(f"Subject {subj_idx+1}: {X_subjects[subj_idx].shape}, Labels: {Y_subjects[subj_idx].shape}")
    
    trg_subj = args.target - 1
    Tx = X_subjects[trg_subj]
    Ty = Y_subjects[trg_subj]
    Tx, m, std = z_score(Tx)

    # Train loader
    train_loader = UnalignedDataLoader()
    train_loader.initialize(len(x_session), X_subjects, Y_subjects, Tx, Ty, trg_subj,
                            args.batch_size, args.batch_size,
                            shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()

    # Test loader
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()

    return datasets, dataset_test, X_subjects, Y_subjects


import os
import scipy.io
import numpy as np

# def load_bnci2014012(args,
#                      path=r"E:\Research\EEGDataSet\BNCI2014012/saved_features/",
#                      n_samples=750):
#     """
#     BNCI2014012 运动想象数据集加载函数（不做滑动窗口）

#     每个受试者文件:
#         A1.mat ~ A9.mat

#     文件内变量:
#         x: [22, 750, 144]   -> [channels, time, trials]
#         y: [144, 1]         -> 标签

#     输出:
#         datasets, dataset_test, X_subjects, Y_subjects

#     最终:
#         X_subjects[subj_idx]: [144, 750, 22]
#         Y_subjects[subj_idx]: [144]
#     """

#     subject_files = [f"A{i}_features.mat" for i in range(1, 10)]
#     X_subjects = {}
#     Y_subjects = {}

#     for subj_idx, subj_file in enumerate(subject_files):
#         full_path = os.path.join(path, subj_file)
#         print("Subject load:", subj_file)

#         dataMat = scipy.io.loadmat(full_path, mat_dtype=True)

#         # 读取数据
#         x = np.asarray(dataMat["x_cov"], dtype=np.float32)   # [22, 750, 144]
#         y = np.asarray(dataMat["y_sub"]).squeeze()           # [144]

#         if y.ndim > 1:
#             y = y.reshape(-1)

#         # 标签映射成连续整数 0,1,...
#         unique_labels = np.unique(y)
#         label_map = {lab: idx for idx, lab in enumerate(sorted(unique_labels))}
#         y = np.array([label_map[lab] for lab in y], dtype=np.int64)

#         subj_X_list = []
#         subj_Y_list = []

#         n_trials = x.shape[2]   # 144

#         for trial_idx in range(n_trials):
#             # 单个 trial: [22, 750] -> [750, 22]
#             features = x[:, :, trial_idx].transpose(1, 0)

#             # 如果需要，只保留最后 n_samples 个时间点
#             if features.shape[0] > n_samples:
#                 features = features[-n_samples:]

#             subj_X_list.append(np.expand_dims(features, axis=0))  # [1, 750, 22]
#             subj_Y_list.append(y[trial_idx])

#         X_subjects[subj_idx] = np.concatenate(subj_X_list, axis=0).astype(np.float32)  # [144, 750, 22]
#         Y_subjects[subj_idx] = np.array(subj_Y_list, dtype=np.int64)                    # [144]

#         print(f"Subject {subj_idx + 1}: {X_subjects[subj_idx].shape}, Labels: {Y_subjects[subj_idx].shape}")

#     # 目标受试者
#     trg_subj = args.target - 1
#     Tx = X_subjects[trg_subj]
#     Ty = Y_subjects[trg_subj]

#     # 和你原逻辑一致：只对目标域做 z-score
#     Tx, m, std = z_score(Tx)

#     # Train loader
#     train_loader = UnalignedDataLoader()
#     train_loader.initialize(
#         len(subject_files),
#         X_subjects,
#         Y_subjects,
#         Tx,
#         Ty,
#         trg_subj,
#         args.batch_size,
#         args.batch_size,
#         shuffle_testing=True,
#         drop_last_testing=True
#     )
#     datasets = train_loader.load_data()

#     # Test loader
#     test_loader = UnalignedDataLoaderTesting()
#     test_loader.initialize(
#         Tx,
#         Ty,
#         200,
#         shuffle_testing=False,
#         drop_last_testing=False
#     )
#     dataset_test = test_loader.load_data()

#     return datasets, dataset_test, X_subjects, Y_subjects


def fine_tuning_load_XY_MI(args, X, Y):
    dset_loaders = {}

    if args.dataset in ["seed", "seed-iv", "bnci2014012", "mi"]:
        print("DATA:", args.dataset)

        if hasattr(args, "session") and args.dataset in ["seed", "seed-iv"]:
            print("SESSION:", args.session)

        subjects = X.keys()
        print(subjects)

        Sx = Sy = None
        i = 0
        flag = False
        selected_subject = args.target - 1
        trg_subj = -1

        for s in subjects:
            if i != selected_subject:
                tr_x = np.array(X[s])
                tr_y = np.array(Y[s])

                # 每个源受试者单独做 z-score，保持你原逻辑不变
                tr_x, m, std = z_score(tr_x)

                if not flag:
                    Sx = tr_x
                    Sy = tr_y
                    flag = True
                else:
                    Sx = np.concatenate((Sx, tr_x), axis=0)
                    Sy = np.concatenate((Sy, tr_y), axis=0)
            else:
                trg_subj = s
            i += 1

        print("[+] Target subject:", trg_subj)

        # 目标受试者
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # 这里保持和你原来一样：
        # Tx 用来估计目标域均值方差
        # Vx 用同样统计量归一化，作为 test
        Vx = Tx.copy()
        Vy = Ty.copy()

        Tx, m, sd = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=sd)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Vx.shape, "Ty_test:", Vy.shape)

        # tensor
        Sx_tensor = torch.tensor(Sx, dtype=torch.float32)
        Sy_tensor = torch.tensor(Sy, dtype=torch.long)

        Tx_tensor = torch.tensor(Tx, dtype=torch.float32)
        Ty_tensor = torch.tensor(Ty, dtype=torch.long)

        Vx_tensor = torch.tensor(Vx, dtype=torch.float32)
        Vy_tensor = torch.tensor(Vy, dtype=torch.long)

        # dataset
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)
        # Keep target labels in the held-out loader only; optimization sees Tx.
        target_tr = TensorDataset(Tx_tensor)
        target_ts = TensorDataset(Vx_tensor, Vy_tensor)

        # dataloader
        dset_loaders["source"] = DataLoader(
            source_tr,
            batch_size=args.batch_size_fine,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        dset_loaders["target"] = DataLoader(
            target_tr,
            batch_size=args.batch_size_fine,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )

        dset_loaders["test"] = DataLoader(
            target_ts,
            batch_size=200,
            shuffle=False,
            num_workers=0
        )

        print("Data were succesfully loaded")

    else:
        print("This dataset does not exist.")
        exit()

    return dset_loaders



import os
import numpy as np
import scipy.io
import h5py


import os
import warnings
import numpy as np
import scipy.io
import h5py


def _get_first_existing_key(container, candidates):
    for k in candidates:
        if k in container:
            return k
    raise KeyError(f"找不到任何一个变量: {candidates}")


def _safe_scalar(container, key, default=None, is_h5=False):
    if key not in container:
        return default
    try:
        value = container[key][()] if is_h5 else container[key]
        value = np.asarray(value).squeeze()
        if value.size == 0:
            return default
        return int(value.reshape(-1)[0])
    except Exception:
        return default


def _mat_numeric_to_real_numpy(x, var_name="array", imag_tol=1e-6):
    """
    将 MATLAB 读出的数值数组安全转换成 numpy 实数数组

    支持:
    1) 普通 float/int 数组
    2) numpy complex 数组
    3) MATLAB v7.3 经 h5py 读取后的复数 structured dtype:
       dtype([('real','<f4'), ('imag','<f4')])

    返回:
        np.float32 实数数组
    """
    x = np.asarray(x)

    # 情况1：MATLAB复数通过 h5py 读成 structured dtype
    if x.dtype.names is not None:
        names = set(x.dtype.names)
        if "real" in names and "imag" in names:
            real = np.asarray(x["real"], dtype=np.float32)
            imag = np.asarray(x["imag"], dtype=np.float32)

            imag_max = float(np.max(np.abs(imag))) if imag.size > 0 else 0.0
            if imag_max > imag_tol:
                warnings.warn(
                    f"{var_name} 含有非零虚部，max|imag|={imag_max:.6e}，当前将只保留实部。",
                    RuntimeWarning
                )
            return real.astype(np.float32)
        else:
            raise TypeError(
                f"{var_name} 是结构化 dtype，但不是支持的 real/imag 格式: {x.dtype}"
            )

    # 情况2：标准 complex 数组
    if np.iscomplexobj(x):
        imag_max = float(np.max(np.abs(np.imag(x)))) if x.size > 0 else 0.0
        if imag_max > imag_tol:
            warnings.warn(
                f"{var_name} 是 complex 数组，max|imag|={imag_max:.6e}，当前将只保留实部。",
                RuntimeWarning
            )
        return np.real(x).astype(np.float32)

    # 情况3：普通数值数组
    return np.asarray(x, dtype=np.float32)


def _fix_x_shape_mi1(x, n_windows=3, feat_dim_hint=None, n_samples_hint=None):
    """
    统一把 X 调整成 [N, W, F]

    兼容:
    - [N, W, F]
    - [F, W, N]
    - [W, N, F] 等任意三维排列
    - 当 W=1 时，也兼容二维:
        [N, F] 或 [F, N]
    """
    x = _mat_numeric_to_real_numpy(x, var_name="X")
    x = np.asarray(x)

    # 不要一上来直接 squeeze 掉所有 1 维
    # 只在维度超过 3 时，逐步去掉多余 singleton 维
    while x.ndim > 3 and 1 in x.shape:
        axis_to_squeeze = list(x.shape).index(1)
        x = np.squeeze(x, axis=axis_to_squeeze)

    # -------------------------
    # 情况1：三维，正常处理
    # -------------------------
    if x.ndim == 3:
        perms = [
            x,
            np.transpose(x, (0, 2, 1)),
            np.transpose(x, (1, 0, 2)),
            np.transpose(x, (1, 2, 0)),
            np.transpose(x, (2, 0, 1)),
            np.transpose(x, (2, 1, 0)),
        ]

        candidates = []
        for arr in perms:
            # 目标格式必须是 [N, W, F]
            if arr.shape[1] != n_windows:
                continue

            score = 0
            if feat_dim_hint is not None and arr.shape[2] == feat_dim_hint:
                score += 10
            if n_samples_hint is not None and arr.shape[0] == n_samples_hint:
                score += 10

            candidates.append((score, arr))

        if len(candidates) == 0:
            raise ValueError(
                f"无法识别三维 X 的顺序: raw shape={x.shape}, "
                f"n_windows={n_windows}, feat_dim_hint={feat_dim_hint}, "
                f"n_samples_hint={n_samples_hint}"
            )

        candidates.sort(key=lambda z: z[0], reverse=True)
        return candidates[0][1].astype(np.float32)

    # -------------------------
    # 情况2：二维，只允许在 W=1 时出现
    # -------------------------
    if x.ndim == 2:
        if n_windows != 1:
            raise ValueError(
                f"X 当前是二维 shape={x.shape}，但 n_windows={n_windows}，无法解释。"
            )

        # 两种可能:
        # 1) [N, F] -> 变成 [N, 1, F]
        # 2) [F, N] -> 转置后变成 [N, 1, F]
        cands = [
            x[:, None, :],     # [N,1,F]
            x.T[:, None, :]    # [N,1,F] （由[F,N]转来）
        ]

        scored = []
        for arr in cands:
            score = 0
            if feat_dim_hint is not None and arr.shape[2] == feat_dim_hint:
                score += 10
            if n_samples_hint is not None and arr.shape[0] == n_samples_hint:
                score += 10
            scored.append((score, arr))

        scored.sort(key=lambda z: z[0], reverse=True)
        best = scored[0][1]

        return best.astype(np.float32)

    raise ValueError(f"X 应该是 2 维或 3 维数组，但当前 shape={x.shape}")

def _read_mi1_loso_mat(full_path, n_windows=3):
    """
    读取 MI1 的 LOSO 特征文件

    返回:
        X_all      : [N, W, F]
        Y_all      : [N]
        subject_id : [N]
        is_target  : [N]
        meta       : dict
    """
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"找不到文件: {full_path}")

    meta = {}

    try:
        # 普通 mat
        data = scipy.io.loadmat(full_path)

        x_key = _get_first_existing_key(data, ["X_split", "X_all"])
        y_key = _get_first_existing_key(data, ["Y_split", "Y_all"])
        s_key = _get_first_existing_key(data, ["subj_split", "subject_id"])
        t_key = _get_first_existing_key(data, ["is_target"])

        X_raw = data[x_key]
        Y_all = np.asarray(data[y_key]).squeeze()
        subject_id = np.asarray(data[s_key]).squeeze()
        is_target = np.asarray(data[t_key]).squeeze()

        meta["feat_dim"] = _safe_scalar(data, "feat_dim", default=None, is_h5=False)
        meta["feat_dim_out"] = _safe_scalar(data, "feat_dim_out", default=None, is_h5=False)
        meta["k"] = _safe_scalar(data, "k", default=None, is_h5=False)
        meta["USE_GFK"] = _safe_scalar(data, "USE_GFK", default=None, is_h5=False)

    except (NotImplementedError, ValueError, OSError):
        # MATLAB -v7.3
        with h5py.File(full_path, "r") as f:
            x_key = _get_first_existing_key(f, ["X_split", "X_all"])
            y_key = _get_first_existing_key(f, ["Y_split", "Y_all"])
            s_key = _get_first_existing_key(f, ["subj_split", "subject_id"])
            t_key = _get_first_existing_key(f, ["is_target"])

            X_raw = f[x_key][()]
            Y_all = np.asarray(f[y_key][()]).squeeze()
            subject_id = np.asarray(f[s_key][()]).squeeze()
            is_target = np.asarray(f[t_key][()]).squeeze()

            meta["feat_dim"] = _safe_scalar(f, "feat_dim", default=None, is_h5=True)
            meta["feat_dim_out"] = _safe_scalar(f, "feat_dim_out", default=None, is_h5=True)
            meta["k"] = _safe_scalar(f, "k", default=None, is_h5=True)
            meta["USE_GFK"] = _safe_scalar(f, "USE_GFK", default=None, is_h5=True)

    # 优先使用 feat_dim_out
    feat_dim_hint = meta.get("feat_dim_out", None)
    if feat_dim_hint is None:
        feat_dim_hint = meta.get("feat_dim", None)

    X_all = _fix_x_shape_mi1(X_raw, n_windows=n_windows, feat_dim_hint=feat_dim_hint, n_samples_hint=len(Y_all))
    Y_all = np.asarray(Y_all).reshape(-1)
    subject_id = np.asarray(subject_id).reshape(-1).astype(np.int64)
    is_target = np.asarray(is_target).reshape(-1).astype(np.int64)

    if X_all.shape[0] != len(Y_all):
        raise ValueError(f"X 与 Y 数量不一致: X={X_all.shape}, Y={Y_all.shape}")
    if X_all.shape[0] != len(subject_id):
        raise ValueError(f"X 与 subject_id 数量不一致: X={X_all.shape}, subject_id={subject_id.shape}")
    if X_all.shape[0] != len(is_target):
        raise ValueError(f"X 与 is_target 数量不一致: X={X_all.shape}, is_target={is_target.shape}")

    return X_all.astype(np.float32), Y_all, subject_id, is_target, meta


def load_mi1(args,
             path=r"E:\Research\EEGDataSet\BNCI20140mi1\saved_loso_window_logmap_gfk",
             n_windows=3,
             k=25):
    """
    读取 MI1 数据集的 LOSO + 时间窗 + logmap + GFK 特征

    文件名格式:
        MI1_LOSO_target_01_W3_logmap_gfk_k25.mat

    返回:
        datasets, dataset_test, X_subjects, Y_subjects

    其中:
        X_subjects[subj_idx]: [200, W, F]
        Y_subjects[subj_idx]: [200]
    """
    file_name = f"MI5_LOSO_target_{args.target:02d}_W{n_windows}_logmap_gfk_k25.mat"
    full_path = os.path.join(path, file_name)

    print("MI1 LOSO file load:", full_path)

    X_all, Y_all, subject_id, is_target, meta = _read_mi1_loso_mat(
        full_path,
        n_windows=n_windows
    )

    print("Raw loaded shapes:")
    print("  X_all      :", X_all.shape)       # [N, W, F]
    print("  Y_all      :", Y_all.shape)       # [N]
    print("  subject_id :", subject_id.shape)  # [N]
    print("  is_target  :", is_target.shape)   # [N]
    print("  meta       :", meta)
    print("Loaded feature dim:", X_all.shape[2])

    # 标签映射到连续整数 0,1,...
    unique_labels = np.unique(Y_all)
    label_map = {lab: idx for idx, lab in enumerate(sorted(unique_labels))}
    Y_all = np.array([label_map[lab] for lab in Y_all], dtype=np.int64)

    # 按受试者拆分
    X_subjects = {}
    Y_subjects = {}

    num_subjects = int(subject_id.max())

    for subj in range(1, num_subjects + 1):
        mask = (subject_id == subj)
        subj_X = X_all[mask].astype(np.float32)   # [200, W, F]
        subj_Y = Y_all[mask].astype(np.int64)     # [200]

        X_subjects[subj - 1] = subj_X
        Y_subjects[subj - 1] = subj_Y

        print(f"Subject {subj}: X={subj_X.shape}, Y={subj_Y.shape}")

    # 当前目标受试者
    trg_subj = args.target - 1
    if trg_subj < 0 or trg_subj >= num_subjects:
        raise ValueError(
            f"args.target={args.target} 越界，当前共有 {num_subjects} 个受试者"
        )

    Tx = X_subjects[trg_subj].copy()
    Ty = Y_subjects[trg_subj].copy()

    # 只对目标域做 z-score
    Tx, m, std = z_score(Tx)

    # train loader
    train_loader = UnalignedDataLoader()
    train_loader.initialize(
        num_subjects,
        X_subjects,
        Y_subjects,
        Tx,
        Ty,
        trg_subj,
        args.batch_size,
        args.batch_size,
        shuffle_testing=True,
        drop_last_testing=True
    )
    datasets = train_loader.load_data()

    # test loader
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(
        Tx,
        Ty,
        10,
        shuffle_testing=False,
        drop_last_testing=False
    )
    dataset_test = test_loader.load_data()

    return datasets, dataset_test, X_subjects, Y_subjects
