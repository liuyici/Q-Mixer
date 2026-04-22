"""

"""

from einops import rearrange

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第2号物理GPU
import numpy as np
import math
import random
import datetime
import time
import scipy.io
from modules import load_mi1, load_seed, fine_tuning_load_XY_MI
from dataloader import *
from model.snn_layers import first_order_low_pass_layer, neuron_layer

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Parameter
# import lr_schedule
from   torch                            import autograd
from   torch.autograd                   import Variable
from   core_qnn.quaternion_layers       import *
import torchvision.transforms as transforms
import utils
from utils import LabelSmooth
import Adver_network
from torch import Tensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize

# ============================================================
# Utils
# ============================================================
def GaussianNoise(x, sigma=1.0):
    if sigma <= 0:
        return x
    noise = torch.randn_like(x) * sigma
    return x + noise


class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        r = input.norm(dim=1).detach()[0]
        cosine = F.linear(input, F.normalize(self.weight), r * torch.tanh(self.bias))
        output = cosine
        return output

class QuaternionFusionHead(nn.Module):
    """四元数特征混淆模块：融合通道和时间特征（两个独立分量，一个相乘，一个相加），使用四元数旋转最终整合一波"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        assert d_model % 4 == 0

        self.d_model = d_model
        self.q_dim = d_model // 4

        self.norm_c = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)

       
        self.r_proj = QuaternionLinear(d_model, self.q_dim)
        self.i_proj = QuaternionLinear(d_model, self.q_dim)
        self.j_proj = QuaternionLinear(d_model, self.q_dim)
        self.k_proj = QuaternionLinear(d_model, self.q_dim)

        self.rot = QuaternionLinearAutograd(
            4, 4,
            bias=False,
            init_criterion='glorot',
            weight_init='quaternion',
            seed=None,
            rotation=True,
            quaternion_format=True,
            scale=False
        )

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

       
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(self, channel, temporal):
        B, T, D = channel.shape
        c = self.norm_c(channel)   # [B,T,D]
        t = self.norm_t(temporal)  # [B,T,D]
        r = self.r_proj(0.5 * (c + t))   # [B,T,q_dim]
        i = self.i_proj(c)               # [B,T,q_dim]
        j = self.j_proj(t)               # [B,T,q_dim]
        k = self.k_proj(c * t)           # [B,T,q_dim]
        q = torch.stack([r, i, j, k], dim=-1)
        q_rot = self.rot(q.reshape(-1, 4)).view(B, T, self.q_dim, 4)
        fused = q_rot.reshape(B, T, self.d_model)
        fused = self.out_proj(fused)
        gate = self.gate(torch.cat([c, t], dim=-1))   # [B,T,d_model]
        fused = gate * fused
        return fused


class QuaternionGatedEEGBlockV3(nn.Module):
    def __init__(self, C=62, Freq=5, T=12, drop_p=0.1):
        super().__init__()
        self.C = C
        self.Freq = Freq
        self.T = T
        self.D_in = C * Freq
        self.D_q = 256

        self.norm = nn.LayerNorm(self.D_in)
        self.drop = nn.Dropout(drop_p)

        self.in_proj = nn.Linear(self.D_in, self.D_q)

        self.time_proj = nn.Linear(T, T)
        self.channel_proj = nn.Linear(self.D_q, self.D_q, bias=False)

        self.quat_fusion = QuaternionFusionHead(
            d_model=self.D_q,
            dropout=0.3
        )

        self.out_proj = nn.Linear(self.D_q, self.D_in)

    def forward(self, x):
        B, T, D = x.shape #[B,15,310] [B,750,22]
        assert T == self.T and D == self.D_in

        x0 = x
        z = self.drop(self.in_proj(self.norm(x)))      # [B,750,24]

        x_time = self.time_proj(z.transpose(1, 2)).transpose(1, 2)   #  [B,750,24]
        x_channel = self.channel_proj(z)                             # [B,750,24]

        fused = self.quat_fusion(x_channel, x_time)                   # [B,T,320]
        y = self.out_proj(fused)                                      # [B,T,310]

        return x0 + y


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, drop_p=0.1, C=62, Freq=5, T=12):
        super().__init__()
        self.block = QuaternionGatedEEGBlockV3(C=C, Freq=Freq, T=T, drop_p=drop_p)

    def forward(self, x):
        return self.block(x)


class TransformerEncoder(nn.Module):
    def __init__(self, depth, emb_size, drop_p=0.1, C=62, Freq=5, T=12):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size, drop_p=drop_p, C=C, Freq=Freq, T=T)
            for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return x

class DFN(nn.Sequential):
    def __init__(self, bottleneck_dim):
        super(DFN, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(1770, 512),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.module(x)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, bottleneck_dim, n_classes):
        super().__init__()
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            SLR_layer(32, n_classes)
        )
        
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1).float()
        out = self.fc2(x)
        
        return x, out


class Discriminator(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            SLR_layer(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1).float()
        out = self.fc2(x)
        
        return out

class QuantGate(nn.Sequential):
    def __init__(self, emb_size=310, depth=6, bottleneck_dim=256, n_classes=4, **kwargs):
        super().__init__()
        self.encoder = TransformerEncoder(
            depth=depth,
            emb_size=emb_size,
            drop_p=0.3,
            C=1770,
            Freq=1,
            T=args.window_size  
        )

        self.dfn = DFN(bottleneck_dim)
        self.head = ClassificationHead(emb_size, bottleneck_dim, n_classes)

    def forward(self, x):
        # x: [B, 5, 253]
        x = self.encoder(x)
        x = self.dfn(x)
        return self.head(x)



class ExGAN():
    def __init__(self, args, nsub, fold):
        super(ExGAN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.batch_size = 300
        self.n_epochs = 15  #1000
        self.lr = 0.002
        self.lr2 = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.radius = 10
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        self.model = QuantGate(emb_size=1770, depth=6, bottleneck_dim=128, n_classes=2).float().cuda()
        self.domain_Discriminator = Discriminator(emb_size=1770, n_classes=6).to(self.device).float()
        self.criterion = LabelSmooth(num_class=args.num_class).cuda()
        
    def schedule_lambda(self, epoch, total_epochs, max_lambda=0.6, k=5):
        p = epoch / total_epochs  # 归一化到 [0,1]
        return max_lambda * (2. / (1. + np.exp(-k * p)) - 1)


    def get_source_data(self, feature="de_LDS"):
        if self.args.dataset == "seed":
            datasets, dataset_test, X_subjects, Y_subjects = load_mi1(args, path=r"E:/Research/EEGDataSet/BNCI20140mi1/saved_loso_window_logmap_gfk", n_windows=1, k=25)
        return datasets, dataset_test, X_subjects, Y_subjects

    def get_source_data_for_fine(self, X, Y):
        if self.args.dataset == "seed":
            dset_loaders = fine_tuning_load_XY_MI(self.args, X, Y)
        return dset_loaders

    def test_suda(self, loader, model):
        start_test = True
        with torch.no_grad():
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.type(torch.FloatTensor).cuda()
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # 自动计算 62×5=310 [批次，3，310]
                labels = labels
                _, outputs = model(inputs.float())
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        _, predictions = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])
        y_true = all_label.cpu().data.numpy()
        y_pred = predictions.cpu().data.numpy()
        labels = np.unique(y_true)
    
        ytest = label_binarize(y_true, classes=labels)
        ypreds = label_binarize(y_pred, classes=labels)
    
        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
        matrix = confusion_matrix(y_true, y_pred)
    
        return accuracy, f1, auc, matrix

    def _to_tensor(self, x, device, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=dtype)
        return x

    def train(self, fold):
        
        train_dataset, test_dataset, X, Y = self.get_source_data(feature="de_LDS")
    
        self.optimizer = torch.optim.SGD(
            list(self.model.parameters()) + list(self.domain_Discriminator.parameters()),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.005
        )
    
        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0
        epochs_acc = []
    
        B = self.args.batch_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        for e in range(self.n_epochs):
            self.model.train()
            self.domain_Discriminator.train()
    
            for i, data in enumerate(train_dataset):
                x_src = [data[f"Sx{idx}"] for idx in range(1, 7)]  # 14 个域
                y_src = [data[f"Sy{idx}"] for idx in range(1, 7)]

                img = torch.cat(x_src, dim=0).to(device, non_blocking=True).float()
                label = torch.cat(y_src, dim=0).to(device, non_blocking=True).long()
    
                x_trg = data["Tx"].to(device, non_blocking=True).float()
                img = img.view(img.size(0), img.size(1), -1)
                x_trg = x_trg.view(x_trg.size(0), x_trg.size(1), -1)
                domain_label = torch.arange(6, device=device, dtype=torch.long).repeat_interleave(B)
                tok, outputs = self.model(img)            # tok: [14B, feat], outputs: [14B, C]
                tok_target, outputs_target = self.model(x_trg)  # [B, feat], [B, C]
                pre_target = torch.softmax(outputs_target, dim=1)  # [B, C]
                tok_s = tok.view(6, B, -1)      # [14, B, feat]
                lab_s = label.view(6, B)        # [14, B]
                tgt_tok_eq = tok_target          # [B, feat]
                tgt_prob_eq = pre_target         # [B, C]
    
                mmd_b_vals, mmd_t_vals = [], []
                for d in range(6):
                    src_tok_d = tok_s[d]                 # [B, feat]
                    src_lab_d = lab_s[d].reshape(B, 1)   # [B, 1]
    
                    mb = utils.marginal(src_tok_d, tgt_tok_eq)
                    mt = utils.conditional(
                        src_tok_d,
                        tgt_tok_eq,
                        src_lab_d,
                        tgt_prob_eq,
                        0.5,
                        5,
                        None
                    )
                    mb = self._to_tensor(mb, outputs.device)
                    mt = self._to_tensor(mt, outputs.device)
                    mmd_b_vals.append(mb)
                    mmd_t_vals.append(mt)
                mmd_b_loss = torch.stack(mmd_b_vals).mean()
                mmd_t_loss = torch.stack(mmd_t_vals).mean()
                MMD_loss = mmd_b_loss + mmd_t_loss
    
                lambda_adv = self.schedule_lambda(e, self.n_epochs)
                features_s_Adver = Adver_network.ReverseLayerF.apply(tok, lambda_adv)

                outputs_D = self.domain_Discriminator(features_s_Adver.float())
                Adver_domain_labels_loss = self.criterion(outputs_D, domain_label)
                slc_loss = self.criterion(outputs, label)
                loss = slc_loss + MMD_loss + Adver_domain_labels_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            out_epoch = time.time()

            if (e + 1) % 1 == 0:
                start_test = True
                with torch.no_grad():        
                    self.model.eval()
        
                    for batch_idx, tar_data in enumerate(test_dataset):
                        Tx = tar_data['Tx']
                        Ty = tar_data['Ty']
                        Tx = Tx.float().cuda()
                        Tx = Tx.view(Tx.size(0), Tx.size(1), -1)  # 自动计算 62×5=310 [批次，3，310]
                        Tok, Cls = self.model(Tx)
                        if start_test:
                            all_output = Cls.float().cpu()
                            all_label = Ty.float()
                            start_test = False
                        else:
                            all_output = torch.cat((all_output, Cls.float().cpu()), 0)
                            all_label = torch.cat((all_label, Ty.float()), 0)
                        loss_test = self.criterion_cls(Cls.float().cpu(), Ty.long())
                torch.cuda.empty_cache()  # 清理GPU缓存
                y_pred = torch.max(all_output, 1)[1]
                acc = float((y_pred == all_label).cpu().numpy().astype(int).sum()) / float(all_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                epochs_acc.append(acc)
                print('Epoch:', e,
                      '  Train loss: %.4f' % loss.item(),
                      '  cls: %.4f' % slc_loss.detach().cpu().numpy(),
                      '  MMD: %.4f' % MMD_loss.item(),
                      '  adv: %.4f' % Adver_domain_labels_loss.detach().cpu().numpy(),
                      '  lambda_adv: %.4f' % lambda_adv,
                      '  Train acc: %.4f' % train_acc,
                      '  Test acc: %.4f' % acc)
             
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = Ty
                    Y_pred = y_pred

        averAcc = averAcc / num
        print('The average accuracy of n_epochs%d is:' %(e+1), averAcc)
        print('The best accuracy of n_epochs%d is:' %(e+1), bestAcc)
     
        return bestAcc, averAcc, Y_true, Y_pred, X, Y, self.model, epochs_acc


    def fine_tuning(self, args, X, Y, model):
        dset_loaders = self.get_source_data_for_fine(X, Y)
        parameter_model = model.parameters()
        self.optimizer = torch.optim.Adam(parameter_model, lr=self.lr2, betas=(self.b1, self.b2))
    
        len_train_source = len(dset_loaders["source"])
        len_train_target = len(dset_loaders["target"])
        best_acc = 0.0
        final_acc = 0
        final_f1 = 0
        final_auc = 0
        final_mat = []
    
        iter_acc_list = []
        iter_f1_list = []
        iter_auc_list = []
    
        for i in range(args.max_iter2):
            if i % 1 == 0:
                with torch.no_grad():
                    model.eval()
                    best_acc, best_f1, best_auc, best_mat = self.test_suda(dset_loaders, model)
    
                    # 记录当前这一轮的结果
                    iter_acc_list.append(best_acc)
                    iter_f1_list.append(best_f1)
                    iter_auc_list.append(best_auc)
    
                    if final_acc < best_acc:
                        final_acc = best_acc
                        final_f1 = best_f1
                        final_auc = best_auc
                        final_mat = best_mat
    
                    if i == 0:
                        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(
                            i, best_acc, best_f1, best_auc
                        )
                    else:
                        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f} \t loss: {:.4f}".format(
                            i, best_acc, best_f1, best_auc, total_loss.item()
                        )
                    print(log_str)
    
            model.train()
            if i % len_train_source == 0:
                iter_source = iter(dset_loaders["source"])
            if i % len_train_target == 0:
                iter_target = iter(dset_loaders["target"])
    
            inputs_source_, labels_source = next(iter_source)
            inputs_target_, ture_labels_target = next(iter_target)
    
            inputs_source_ = inputs_source_.type(torch.FloatTensor)
            labels_source = labels_source.type(torch.LongTensor)
            inputs_target_ = inputs_target_.type(torch.FloatTensor)
            ture_labels_target = ture_labels_target.type(torch.LongTensor)
            inputs_source, labels_source = inputs_source_.cuda(), labels_source.cuda()
            inputs_target, ture_labels_target = inputs_target_.cuda(), ture_labels_target.cuda()
            inputs_source = inputs_source.view(inputs_source.size(0), inputs_source.size(1), -1)
            inputs_target = inputs_target.view(inputs_target.size(0), inputs_target.size(1), -1)
            features_source, outputs_source = model(inputs_source)
            features_target, outputs_target = model(inputs_target)
            classifier_loss = self.criterion_cls(outputs_source, labels_source.flatten())
            total_loss = classifier_loss   # + 2 * CORAL
    
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
    
        return final_acc, final_f1, final_auc, final_mat, model, iter_acc_list, iter_f1_list, iter_auc_list


def main(args):
    pre_train = []
    tuning = []
    result_write = open("E:\Research\第八篇论文科研\code\quant_Gate_2_forMI\snapshot.txt", "w")
    total_acc = []
    all_subject_ft_acc = []
    all_subject_ft_f1 = []
    all_subject_ft_auc = []
    for i in range(7):
        args.target = 7 - i
        seed_n = 1

        result_write.write('--------------------------------------------------')
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        print('Subject %d' % (i+1))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        ba = 0
        aa = 0
        pre_train_Acc = 0
        averAcc = 0

        exgan = ExGAN(args, i + 1, 1)
        
        ba, aa, _, _, X, Y, model, epochs_acc  = exgan.train(1)
        total_acc.append(epochs_acc)
        
        final_acc, final_f1, final_auc, final_mat, model, iter_acc_list, iter_f1_list, iter_auc_list = exgan.fine_tuning(args, X, Y, model)
        all_subject_ft_acc.append(iter_acc_list)
        all_subject_ft_f1.append(iter_f1_list)
        all_subject_ft_auc.append(iter_auc_list)

        result_write.write('pre_training acc is:' + str(ba) + "\n")
        result_write.write('fine_tuning acc is:' + str(final_acc) + "\n")

        pre_train_Acc = ba
        tuning_Acc = final_acc

        pre_train.append(pre_train_Acc)
        tuning.append(tuning_Acc)

        print('pre_training acc is:', pre_train)
        print('fine_tuning acc is:', tuning)


    total_acc = np.array(total_acc)
    epoch_mean_acc = np.mean(total_acc, axis=0)
    print(f"所有epochs的平均准确率: {epoch_mean_acc}")

    best_epoch = np.argmax(epoch_mean_acc) + 1
    best_epoch_acc = epoch_mean_acc[best_epoch - 1]
    print(f"\n最佳epoch为: {best_epoch}，对应平均准确率 = {best_epoch_acc:.4f}")

    all_subject_ft_acc = np.array(all_subject_ft_acc)   # [9, max_iter2]
    all_subject_ft_f1 = np.array(all_subject_ft_f1)
    all_subject_ft_auc = np.array(all_subject_ft_auc)

    mean_ft_acc = np.mean(all_subject_ft_acc, axis=0)   # [max_iter2]
    mean_ft_f1 = np.mean(all_subject_ft_f1, axis=0)
    mean_ft_auc = np.mean(all_subject_ft_auc, axis=0)

    best_ft_iter = np.argmax(mean_ft_acc) + 1
    best_ft_acc = mean_ft_acc[best_ft_iter - 1]
    best_ft_f1 = mean_ft_f1[best_ft_iter - 1]
    best_ft_auc = mean_ft_auc[best_ft_iter - 1]
    best_ft_idx = best_ft_iter - 1 

    subject_best_iter_acc = all_subject_ft_acc[:, best_ft_idx]
    subject_best_iter_f1  = all_subject_ft_f1[:, best_ft_idx]
    subject_best_iter_auc = all_subject_ft_auc[:, best_ft_idx]
    print("\n================= Fine-tuning平均结果 =================")
    print(f"每个微调iter在9个受试者上的平均准确率: {mean_ft_acc}")
    print(f"最佳微调iter为: {best_ft_iter}")
    print(f"该iter的平均准确率 = {best_ft_acc:.4f}")
    print(f"该iter的平均F1 = {best_ft_f1:.4f}")
    print(f"该iter的平均AUC = {best_ft_auc:.4f}")
    print("\n================= 每位受试者在最佳微调iter上的结果 =================")
    for subj in range(7):
        print(
            f"Subject {subj+1}: "
            f"acc = {subject_best_iter_acc[subj]:.4f}, "
            f"f1 = {subject_best_iter_f1[subj]:.4f}, "
            f"auc = {subject_best_iter_auc[subj]:.4f}"
        )
        pre_ave = sum(pre_train) / len(pre_train)
        tuning_ave = sum(tuning) / len(tuning)

    print('------------------------pre-training result--------------------------', pre_train)
    print('------------------------fin-tuning result--------------------------', tuning)
    print('------------------------pre-training average result--------------------------', pre_ave)
    print('------------------------fin-tuning average result--------------------------', tuning_ave)

    result_write.write('--------------------------------------------------\n')
    result_write.write(f"All accuracy is: {pre_train}\n")
    result_write.write(f"All subject Aver accuracy is: {tuning}\n")
    result_write.write(f"Best fine-tuning iter across 9 subjects: {best_ft_iter}\n")
    result_write.write(f"Best fine-tuning mean acc: {best_ft_acc:.4f}\n")
    result_write.write(f"Best fine-tuning mean f1: {best_ft_f1:.4f}\n")
    result_write.write(f"Best fine-tuning mean auc: {best_ft_auc:.4f}\n")
    result_write.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spherical Space Domain Adaptation with Pseudo-label Loss')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset',type=str,default='seed')
    parser.add_argument('--source', type=str, default='amazon')
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--iteration', type=int, default=1, help="Iteration repetitions")
    parser.add_argument('--test_interval', type=int, default=1, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=1000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--mixed_sessions', type=str, default='per_session', help="[per_session | mixed]")
    parser.add_argument('--lr_a', type=float, default=0.1, help="learning rate 1")
    parser.add_argument('--lr_b', type=float, default=0.1, help="learning rate 2")
    parser.add_argument('--radius', type=float, default=10, help="radius")
    parser.add_argument('--num_class',type=int,default=2,help='the number of classes')
    parser.add_argument('--stages', type=int, default=1, help='the number of alternative iteration stages')
    parser.add_argument('--max_iter1',type=int,default=50)
    parser.add_argument('--max_iter2', type=int, default=55)
    parser.add_argument('--batch_size',type=int,default=50)
    parser.add_argument('--batch_size_fine',type=int,default=32)
    parser.add_argument('--seed', type=int, default=123, help="random seed number ")
    parser.add_argument('--hidden_size', type=int, default=512, help="Bottleneck (features) dimensionality")
    parser.add_argument('--bottleneck_dim', type=int, default=256, help="Bottleneck (features) dimensionality")
    parser.add_argument('--session', type=int, default=1, help="random seed number ")
    parser.add_argument('--gamma', type=int, default=1, help="gamma for Adver_network ")
    parser.add_argument('--file_path', type=str, default="E:\Research\EEGDataSet\BNCI20140mi1\saved_loso_window_logmap_gfk", help="Path from the current dataset")
    parser.add_argument('--log_file')
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--d_classes', type=int, default=6)
    parser.add_argument('--window_size', type=int, default=1)
    parser.add_argument('--tau_m', type=int, default=1)
    parser.add_argument('--train_coefficients', type=int, default=True)
    parser.add_argument('--train_bias', type=int, default=True)
    parser.add_argument('--membrane_filter', type=int, default=False)
    parser.add_argument('--length', type=int, default=25)
    #####
    parser.add_argument('--ila_switch_iter', type=int, default=1, help="number of iterations when only DA loss works and sim doesn't")
    parser.add_argument('--n_samples', type=int, default=2, help='number of samples from each src class')
    parser.add_argument('--mu', type=int, default=80, help="these many target samples are used finally, eg. 2/3 of batch")  # mu in number
    parser.add_argument('--k', type=int, default=3, help="k")
    parser.add_argument('--msc_coeff', type=float, default=1.0, help="coeff for similarity loss")
    parser.add_argument('--seq_len', type=int, default=12, help='Temporal length')
    parser.add_argument('--enc_in', type=int, default=310, help='Input feature dim = 62*5')    
    parser.add_argument('--d_model', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--d_ff', type=int, default=128, help='FFN hidden dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function')
    parser.add_argument('--v_layer', type=int, default=2, help='Channel encoder layers')
    parser.add_argument('--t_layer', type=int, default=1, help='Temporal encoder layers')
    parser.add_argument('--patch_len', type=int, default=3, help='Patch length for temporal branch')
    parser.add_argument('--augmentations', type=str, default='channel', help='Data augmentation names')

    args = parser.parse_args()

    main(args)
