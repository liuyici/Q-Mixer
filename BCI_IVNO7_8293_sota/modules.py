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

def split_data(X, Y, seed, test_size=0.3):

    s = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for train_index, test_index in s.split(X, Y):
        X_tr, X_ts = X[train_index], X[test_index]
        Y_tr, Y_ts = Y[train_index], Y[test_index]

    return X_tr, Y_tr, X_ts, Y_ts


class PseudoLabeledData(Dataset):
    def __init__(self, X, Y, W):
        self.X = torch.Tensor(X).float()
        self.Y = torch.Tensor(Y).long()
        self.W = torch.Tensor(W).float()

    def __len__(self):
        return len(self.X)

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

    y_session = scipy.io.loadmat(path + "label.mat", mat_dtype=True)["label"][0]
    y_session = y_session + 1  # Map -1, 0, 1 to 0, 1, 2.

    X_subjects = {}
    Y_subjects = {}

    for subj_idx, subj in enumerate(x_session):
        print("Subject load:", subj)
        dataMat = scipy.io.loadmat(path + subj + ".mat", mat_dtype=True)

        subj_X_list = []
        subj_Y_list = []

        for trial_idx in range(15):
            features = dataMat[feature + str(trial_idx + 1)]
            features = np.swapaxes(features, 0, 1)

            if features.shape[0] > n_samples:
                features = features[-n_samples:]

            window_size = 12
            temp_feats = [np.expand_dims(features[i:i + window_size], axis=0)
                          for i in range(len(features) - window_size + 1)]
            temp_feats = np.concatenate(temp_feats, axis=0)

            labels = np.array([y_session[trial_idx]] * temp_feats.shape[0])
            subj_X_list.append(temp_feats)
            subj_Y_list.append(labels)

        X_subjects[subj_idx] = np.concatenate(subj_X_list, axis=0)
        Y_subjects[subj_idx] = np.concatenate(subj_Y_list, axis=0)
        print(f"Subject {subj_idx+1}: {X_subjects[subj_idx].shape}, Labels: {Y_subjects[subj_idx].shape}")

    trg_subj = args.target - 1
    Tx = X_subjects[trg_subj]
    Ty = Y_subjects[trg_subj]
    Tx, m, std = z_score(Tx)

    train_loader = UnalignedDataLoader()
    train_loader.initialize(len(x_session), X_subjects, Y_subjects, Tx, Ty, trg_subj,
                            args.batch_size, args.batch_size,
                            shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()

    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()

    return datasets, dataset_test, X_subjects, Y_subjects


def load_seed_raw(args, path, session="all", feature="LDS", n_samples=185):
    """Load per-subject SEED segments saved as NumPy arrays."""


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
        print(X_subjects[subj].shape)

    trg_subj = args.target - 1
    Tx = np.array(X_subjects[trg_subj])
    Ty = np.array(Y_subjects[trg_subj])
    subject_ids = X_subjects.keys()
    num_domains = len(subject_ids)
    Tx, m, std = z_score(Tx)
    train_loader = UnalignedDataLoader()
    train_loader.initialize(num_domains, X_subjects, Y_subjects, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()
    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()

    return datasets, dataset_test


def load_seed_three_feature(args, path, session="all", feature="LDS", n_samples=185):
    """Load and concatenate SEED differential-entropy, PSD, and PLV features."""


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

    labels = scipy.io.loadmat(path + "label.mat", mat_dtype=True)
    y_session = labels["label"][0]
    # Map -1, 0, 1 to 0, 1, 2.
    for i in range(len(y_session)):
        y_session[i] += 1
    print(y_session)

    if session == 1:
        x_session = session1
    elif session == 2:
        x_session = session2
    elif session == 3:
        x_session = session3

    samples_by_subject = 0
    X = []
    plv = []
    Y = []
    flag = False
    contact = False
    index = 0
    for subj in x_session:
        dataMat = scipy.io.loadmat(path + subj + ".mat", mat_dtype=True)
        plv_dir_data = f"/home/lyc/dataset/seed_plv//S{index+1}_session{session}_plv.npy"
        plvMat = np.load(plv_dir_data, allow_pickle=True)
        print("Subject load:", subj)
        plv_feature = plvMat.reshape(-1,62,5)
        if contact == 0:
                plv = plv_feature
                contact = True
        else:
                plv = np.concatenate((plv, plv_feature), axis=0)

        index += 1
        for i in range(15):

            features = dataMat[feature+str(i+1)]
            PSD_feature = dataMat[feature_2+str(i+1)]
            features = np.swapaxes(features, 0, 1)
            PSD_feature = np.swapaxes(PSD_feature, 0, 1)
            if (features.shape[0] - n_samples) > 0:
                pos = features.shape[0] - n_samples
                features = features[pos:]
                PSD_feature = PSD_feature[pos:]


            features = np.stack([features, PSD_feature], axis=1)

            labels = np.array([y_session[i]] * features.shape[0])

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
    plv = plv[:, np.newaxis, :, :]
    X = np.concatenate([X, plv], axis=1)
    zero_pad = np.zeros((X.shape[0], 1, 62, 5), dtype=X.dtype)
    X = np.concatenate([X, zero_pad], axis=1)
    X_subjects = {}
    Y_subjects = {}
    n = samples_by_subject
    r = 0
    for subj in range(len(x_session)):
        X_subjects[subj] = X[r:r+n]
        Y_subjects[subj] = Y[r:r+n]
        r += n
        print(X_subjects[subj].shape)
    trg_subj = args.target - 1
    Tx = np.array(X_subjects[trg_subj])
    Ty = np.array(Y_subjects[trg_subj])
    subject_ids = X_subjects.keys()
    num_domains = len(subject_ids)
    Tx, m, std = z_score(Tx)
    train_loader = UnalignedDataLoader()
    train_loader.initialize(num_domains, X_subjects, Y_subjects, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()
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
        data = pickle.loads(data_bytes)
        label = pickle.loads(label_bytes)
        concat_fea = []
        concat_label = []
        first = False
        for trial_idx in range(15,30):
            feature = data[trial_idx]
            one_label = label[trial_idx]
            if first is False:
                concat_fea = feature
                concat_label = one_label
                first = True
            else:
                concat_fea = np.concatenate((concat_fea, feature), axis=0)
                concat_label = np.concatenate((concat_label, one_label), axis=0)


        X_subjects[subj_idx] = concat_fea
        Y_subjects[subj_idx] = concat_label
        print(f"Subject {subj_idx+1}: {X_subjects[subj_idx].shape}, Labels: {Y_subjects[subj_idx].shape}")

    trg_subj = args.target - 1
    Tx = X_subjects[trg_subj]
    Ty = Y_subjects[trg_subj]
    Tx, m, std = z_score(Tx)

    train_loader = UnalignedDataLoader()
    train_loader.initialize(len(x_session), X_subjects, Y_subjects, Tx, Ty, trg_subj,
                            args.batch_size, args.batch_size,
                            shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()

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

                # Standardize each source subject independently.
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

        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # Reuse target-domain statistics for evaluation.
        Vx = Tx.copy()
        Vy = Ty.copy()

        Tx, m, sd = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=sd)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Vx.shape, "Ty_test:", Vy.shape)

        Sx_tensor = torch.tensor(Sx, dtype=torch.float32)
        Sy_tensor = torch.tensor(Sy, dtype=torch.long)

        Tx_tensor = torch.tensor(Tx, dtype=torch.float32)
        Ty_tensor = torch.tensor(Ty, dtype=torch.long)

        Vx_tensor = torch.tensor(Vx, dtype=torch.float32)
        Vy_tensor = torch.tensor(Vy, dtype=torch.long)

        source_tr = TensorDataset(Sx_tensor, Sy_tensor)
        # Unlabeled target batches and labeled evaluation batches.
        target_tr = TensorDataset(Tx_tensor)
        target_ts = TensorDataset(Vx_tensor, Vy_tensor)

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
    eyepath = r"E:\Research\EEGDataSet\SEED-V\Eye_movement_features/"
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
        data = pickle.loads(data_bytes)
        label = pickle.loads(label_bytes)
        eyedata = pickle.loads(eyedata_bytes)
        eyelabel = pickle.loads(eyelabel_bytes)
        concat_fea = []
        concat_label = []
        first = False
        for trial_idx in range(15,30):
            feature = data[trial_idx]
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


        X_subjects[subj_idx] = concat_fea
        Y_subjects[subj_idx] = concat_label
        print(f"Subject {subj_idx+1}: {X_subjects[subj_idx].shape}, Labels: {Y_subjects[subj_idx].shape}")

    trg_subj = args.target - 1
    Tx = X_subjects[trg_subj]
    Ty = Y_subjects[trg_subj]
    Tx, m, std = z_score(Tx)

    train_loader = UnalignedDataLoader()
    train_loader.initialize(len(x_session), X_subjects, Y_subjects, Tx, Ty, trg_subj,
                            args.batch_size, args.batch_size,
                            shuffle_testing=True, drop_last_testing=True)
    datasets = train_loader.load_data()

    test_loader = UnalignedDataLoaderTesting()
    test_loader.initialize(Tx, Ty, 200, shuffle_testing=False, drop_last_testing=False)
    dataset_test = test_loader.load_data()

    return datasets, dataset_test, X_subjects, Y_subjects


import os
import scipy.io
import numpy as np

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

                # Standardize each source subject independently.
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

        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # Reuse target-domain statistics for evaluation.
        Vx = Tx.copy()
        Vy = Ty.copy()

        Tx, m, sd = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=sd)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Vx.shape, "Ty_test:", Vy.shape)

        Sx_tensor = torch.tensor(Sx, dtype=torch.float32)
        Sy_tensor = torch.tensor(Sy, dtype=torch.long)

        Tx_tensor = torch.tensor(Tx, dtype=torch.float32)
        Ty_tensor = torch.tensor(Ty, dtype=torch.long)

        Vx_tensor = torch.tensor(Vx, dtype=torch.float32)
        Vy_tensor = torch.tensor(Vy, dtype=torch.long)

        source_tr = TensorDataset(Sx_tensor, Sy_tensor)
        # Unlabeled target batches and labeled evaluation batches.
        target_tr = TensorDataset(Tx_tensor)
        target_ts = TensorDataset(Vx_tensor, Vy_tensor)

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
    """Convert MATLAB numeric arrays to float32, retaining the real component.

    Accepts NumPy numeric arrays and MATLAB v7.3 real/imag records.
    Warns when the discarded imaginary component exceeds imag_tol.
    """
    x = np.asarray(x)

    # MATLAB v7.3 stores complex values as real/imag fields.
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

    if np.iscomplexobj(x):
        imag_max = float(np.max(np.abs(np.imag(x)))) if x.size > 0 else 0.0
        if imag_max > imag_tol:
            warnings.warn(
                f"{var_name} 是 complex 数组，max|imag|={imag_max:.6e}，当前将只保留实部。",
                RuntimeWarning
            )
        return np.real(x).astype(np.float32)

    return np.asarray(x, dtype=np.float32)


def _fix_x_shape_mi1(x, n_windows=3, feat_dim_hint=None, n_samples_hint=None):
    """Arrange features as [trials, windows, features].

    Accepts three-dimensional axis permutations, or [trials, features]
    and its transpose when n_windows is 1.
    """
    x = _mat_numeric_to_real_numpy(x, var_name="X")
    x = np.asarray(x)

    # Preserve the window axis when W = 1.
    while x.ndim > 3 and 1 in x.shape:
        axis_to_squeeze = list(x.shape).index(1)
        x = np.squeeze(x, axis=axis_to_squeeze)

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

    if x.ndim == 2:
        if n_windows != 1:
            raise ValueError(
                f"X 当前是二维 shape={x.shape}，但 n_windows={n_windows}，无法解释。"
            )

        cands = [
            x[:, None, :],
            x.T[:, None, :]
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
    """Read a LOSO MAT file.

    Returns X [trials, windows, features], labels, subject IDs,
    target flags, and feature metadata.
    """
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"找不到文件: {full_path}")

    meta = {}

    try:
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
        # MATLAB v7.3 (HDF5).
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
    """Load one LOSO fold and build training and evaluation loaders.

    X_subjects maps subject indices to [trials, windows, features] arrays;
    Y_subjects contains the corresponding labels.
    """
    file_name = f"MI1_LOSO_target_{args.target:02d}_W{n_windows}_logmap_gfk_k10.mat"
    full_path = os.path.join(path, file_name)

    print("MI1 LOSO file load:", full_path)

    X_all, Y_all, subject_id, is_target, meta = _read_mi1_loso_mat(
        full_path,
        n_windows=n_windows
    )

    print("Raw loaded shapes:")
    print("  X_all      :", X_all.shape)       # [N, W, F]
    print("  Y_all      :", Y_all.shape)
    print("  subject_id :", subject_id.shape)
    print("  is_target  :", is_target.shape)
    print("  meta       :", meta)
    print("Loaded feature dim:", X_all.shape[2])

    # Map labels to contiguous class indices.
    unique_labels = np.unique(Y_all)
    label_map = {lab: idx for idx, lab in enumerate(sorted(unique_labels))}
    Y_all = np.array([label_map[lab] for lab in Y_all], dtype=np.int64)

    X_subjects = {}
    Y_subjects = {}

    num_subjects = int(subject_id.max())

    for subj in range(1, num_subjects + 1):
        mask = (subject_id == subj)
        subj_X = X_all[mask].astype(np.float32)   # [trials, windows, features]
        subj_Y = Y_all[mask].astype(np.int64)

        X_subjects[subj - 1] = subj_X
        Y_subjects[subj - 1] = subj_Y

        print(f"Subject {subj}: X={subj_X.shape}, Y={subj_Y.shape}")

    trg_subj = args.target - 1
    if trg_subj < 0 or trg_subj >= num_subjects:
        raise ValueError(
            f"args.target={args.target} 越界，当前共有 {num_subjects} 个受试者"
        )

    Tx = X_subjects[trg_subj].copy()
    Ty = Y_subjects[trg_subj].copy()

    Tx, m, std = z_score(Tx)

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
