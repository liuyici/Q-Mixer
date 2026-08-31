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

# ============================================================
# Utils
# ============================================================
def GaussianNoise(x, sigma=1.0):
    if sigma <= 0:
        return x
    noise = torch.randn_like(x) * sigma
    return x + noise


def hamilton_product(q1, q2):
    """Hamilton product for feature vectors packed as [r, i, j, k] blocks."""
    if q1.shape != q2.shape or q1.size(-1) % 4 != 0:
        raise ValueError("Hamilton inputs must have the same shape and a 4-way last dimension")
    a, b, c, d = q1.chunk(4, dim=-1)
    e, f, g, h = q2.chunk(4, dim=-1)
    return torch.cat((
        a * e - b * f - c * g - d * h,
        a * f + b * e + c * h - d * g,
        a * g - b * h + c * e + d * f,
        a * h + b * g - c * f + d * e,
    ), dim=-1)


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
    """Fuse channel and temporal features with a Hamilton-product interaction."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        assert d_model % 4 == 0

        self.d_model = d_model
        self.q_dim = d_model // 4

        self.norm_c = nn.LayerNorm(d_model)
        self.norm_t = nn.LayerNorm(d_model)

       
        self.temporal_proj = QuaternionLinear(d_model, self.q_dim)
        self.shared_proj = QuaternionLinear(d_model, self.q_dim)
        self.interaction_proj = QuaternionLinear(d_model, self.q_dim)
        self.spatial_proj = QuaternionLinear(d_model, self.q_dim)

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
        temporal = self.temporal_proj(t)             # q_1: temporal
        shared = self.shared_proj(0.5 * (c + t))     # q_2: shared
        interaction = self.interaction_proj(hamilton_product(c, t))  # q_3: interaction
        spatial = self.spatial_proj(c)               # q_4: spatial
        q = torch.stack([temporal, shared, interaction, spatial], dim=-1)
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
        self.D_q = 512

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
            nn.Linear(120 * 5, 512),
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
    def __init__(self, emb_size=310, depth=2, bottleneck_dim=256, n_classes=2, **kwargs):
        super().__init__()
        self.encoder = TransformerEncoder(
            depth=depth,
            emb_size=emb_size,
            drop_p=0.3,
            C=120,
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
        self.batch_size = 100
        self.n_epochs = args.n_epochs
        self.lr = 0.002
        self.lr2 = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.beta = getattr(args, 'beta', 1.0)
        self.gamma = getattr(args, 'gamma', 1.0)
        self.radius = 10
        self.criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
        self.model = QuantGate(emb_size=120, depth=6, bottleneck_dim=128, n_classes=2).float().to(self.device)
        self.domain_Discriminator = Discriminator(emb_size=120, n_classes=13).to(self.device).float()
        self.criterion = LabelSmooth(num_class=args.num_class).to(self.device)
        
    def schedule_lambda(self, epoch, total_epochs, max_lambda=0.6, k=5):
        p = epoch / total_epochs  # 归一化到 [0,1]
        return max_lambda * (2. / (1. + np.exp(-k * p)) - 1)


    def get_source_data(self, feature="de_LDS"):
        if self.args.dataset == "seed":
            datasets, dataset_test, X_subjects, Y_subjects = load_mi1(self.args, path=self.args.file_path, n_windows=self.args.window_size, k=25)
        return datasets, dataset_test, X_subjects, Y_subjects

    def get_source_data_for_fine(self, X, Y):
        if self.args.dataset == "seed":
            dset_loaders = fine_tuning_load_XY_MI(self.args, X, Y)
        return dset_loaders

    def _compute_metrics(self, logits, labels):
        """Compute final-report metrics without using them for model selection."""
        logits = logits.detach().float().cpu()
        labels = torch.as_tensor(labels).view(-1).long().cpu()
        probabilities = torch.softmax(logits, dim=1).numpy()
        predictions = probabilities.argmax(axis=1)
        y_true = labels.numpy()
        accuracy = float(np.mean(predictions == y_true))
        f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
        try:
            if probabilities.shape[1] == 2:
                auc = roc_auc_score(y_true, probabilities[:, 1])
            else:
                auc = roc_auc_score(y_true, probabilities, average='macro', multi_class='ovr')
        except ValueError:
            auc = float('nan')
        matrix = confusion_matrix(y_true, predictions)
        return accuracy, f1, auc, matrix, y_true, predictions

    def test_suda(self, loader, model):
        logits, labels = [], []
        with torch.no_grad():
            model.eval()
            for inputs, batch_labels in loader["test"]:
                inputs = inputs.float().to(self.device)
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
                batch_logits = model(inputs)[1]
                logits.append(batch_logits.cpu())
                labels.append(batch_labels.cpu())
        metrics = self._compute_metrics(torch.cat(logits), torch.cat(labels))
        return metrics[:4]

    def evaluate_target(self, test_dataset):
        """Evaluate once after training; target labels are never used for selection."""
        logits, labels = [], []
        with torch.no_grad():
            self.model.eval()
            for tar_data in test_dataset:
                inputs = tar_data['Tx'].float().to(self.device)
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
                logits.append(self.model(inputs)[1].cpu())
                labels.append(tar_data['Ty'].cpu())
        return self._compute_metrics(torch.cat(logits), torch.cat(labels))

    def _to_tensor(self, x, device, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, device=device, dtype=dtype)
        return x

    def train(self, fold):
        
        train_dataset, test_dataset, X, Y = self.get_source_data(feature="de_LDS")
    
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.domain_Discriminator.parameters()),
            lr=self.lr,
            betas=(self.b1, self.b2),
            weight_decay=0.005
        )
    
        epochs_acc = []
    
        B = self.args.batch_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        for e in range(self.n_epochs):
            self.model.train()
            self.domain_Discriminator.train()
    
            for i, data in enumerate(train_dataset):
                x_src = [data[f"Sx{idx}"] for idx in range(1, 5)]  # 14 个域
                y_src = [data[f"Sy{idx}"] for idx in range(1, 5)]
                img = torch.cat(x_src, dim=0).to(device, non_blocking=True).float()
                label = torch.cat(y_src, dim=0).to(device, non_blocking=True).long()
    
                x_trg = data["Tx"].to(device, non_blocking=True).float()
                img = img.view(img.size(0), img.size(1), -1)
                x_trg = x_trg.view(x_trg.size(0), x_trg.size(1), -1)
                domain_label = torch.arange(4, device=device, dtype=torch.long).repeat_interleave(B)
                tok, outputs = self.model(img)            # tok: [14B, feat], outputs: [14B, C]
                tok_target, outputs_target = self.model(x_trg)  # [B, feat], [B, C]
                pre_target = torch.softmax(outputs_target, dim=1)  # [B, C]
                tok_s = tok.view(4, B, -1)      # [14, B, feat]
                lab_s = label.view(4, B)        # [14, B]
                tgt_tok_eq = tok_target          # [B, feat]
                tgt_prob_eq = pre_target         # [B, C]
    
                mmd_b_vals, mmd_t_vals = [], []
                for d in range(4):
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
                MMD_loss = 0.5 * mmd_b_loss + 0.5 * mmd_t_loss
    
                lambda_adv = self.schedule_lambda(e, self.n_epochs)
                features_s_Adver = Adver_network.ReverseLayerF.apply(tok, lambda_adv)

                outputs_D = self.domain_Discriminator(features_s_Adver.float())
                Adver_domain_labels_loss = self.criterion(outputs_D, domain_label)
                slc_loss = self.criterion(outputs, label)
                loss = (slc_loss + self.beta * MMD_loss
                        + self.gamma * Adver_domain_labels_loss)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            train_pred = torch.max(outputs, 1)[1]
            train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
            epochs_acc.append(train_acc)
            print('Epoch:', e,
                  '  Train loss: %.4f' % loss.item(),
                  '  cls: %.4f' % slc_loss.detach().cpu().numpy(),
                  '  MMD: %.4f' % MMD_loss.item(),
                  '  adv: %.4f' % Adver_domain_labels_loss.detach().cpu().numpy(),
                  '  lambda_adv: %.4f' % lambda_adv,
                  '  Source train acc: %.4f' % train_acc)

        # The target labels are read only once for the final, held-out report.
        final_acc, _, _, _, y_true, y_pred = self.evaluate_target(test_dataset)
        Y_true = torch.as_tensor(y_true)
        Y_pred = torch.as_tensor(y_pred)
        print('Fixed pre-training epochs:', self.n_epochs)
        print('Final target accuracy:', final_acc)
     
        return final_acc, final_acc, Y_true, Y_pred, X, Y, self.model, epochs_acc


    def fine_tuning(self, args, X, Y, model):
        dset_loaders = self.get_source_data_for_fine(X, Y)
        parameter_model = model.parameters()
        self.optimizer = torch.optim.Adam(parameter_model, lr=self.lr2, betas=(self.b1, self.b2))
    
        len_train_source = len(dset_loaders["source"])
        len_train_target = len(dset_loaders["target"])
        final_acc = 0
        final_f1 = 0
        final_auc = 0
        final_mat = []
    
        iter_acc_list = []
        iter_f1_list = []
        iter_auc_list = []
    
        for i in range(args.max_iter2):
            model.train()
            if i % len_train_source == 0:
                iter_source = iter(dset_loaders["source"])
            if i % len_train_target == 0:
                iter_target = iter(dset_loaders["target"])
    
            inputs_source_, labels_source = next(iter_source)
            inputs_target_, _ = next(iter_target)
    
            inputs_source_ = inputs_source_.type(torch.FloatTensor)
            labels_source = labels_source.type(torch.LongTensor)
            inputs_target_ = inputs_target_.type(torch.FloatTensor)
    
            inputs_source, labels_source = inputs_source_.to(self.device), labels_source.to(self.device)
            inputs_target = inputs_target_.to(self.device)
    
            inputs_source = inputs_source.view(inputs_source.size(0), inputs_source.size(1), -1)
            inputs_target = inputs_target.view(inputs_target.size(0), inputs_target.size(1), -1)
    
            features_source, outputs_source = model(inputs_source)
            model(inputs_target)
    
            classifier_loss = self.criterion_cls(outputs_source, labels_source.flatten())
            total_loss = classifier_loss   # + 2 * CORAL
    
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        # Select no checkpoint using target labels; report only the fixed final iteration.
        final_acc, final_f1, final_auc, final_mat = self.test_suda(dset_loaders, model)
        iter_acc_list.append(final_acc)
        iter_f1_list.append(final_f1)
        iter_auc_list.append(final_auc)
        print("Fixed fine-tuning iterations: {:d}; final target accuracy: {:.4f} \t weighted F1: {:.4f} \t AUC: {:.4f}".format(
            args.max_iter2, final_acc, final_f1, final_auc
        ))
        return final_acc, final_f1, final_auc, final_mat, model, iter_acc_list, iter_f1_list, iter_auc_list


def main(args):
    subject_count = 14
    seed_values = [args.seed + i for i in range(args.num_seeds)]
    pre_train, tuning, total_acc = [], [], []
    all_subject_ft_acc, all_subject_ft_f1, all_subject_ft_auc = [], [], []

    with open("snapshot.txt", "w") as result_write:
        for seed_n in seed_values:
            for i in range(subject_count):
                args.target = subject_count - i
                random.seed(seed_n)
                np.random.seed(seed_n)
                torch.manual_seed(seed_n)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_n)
                print('Subject %d (seed %d)' % (i + 1, seed_n))
                result_write.write('Subject %d, seed %d\n' % (i + 1, seed_n))
                exgan = ExGAN(args, i + 1, 1)
                ba, _, _, _, X, Y, model, epochs_acc = exgan.train(1)
                total_acc.append(epochs_acc)
                final_acc, final_f1, final_auc, _, model, iter_acc, iter_f1, iter_auc = exgan.fine_tuning(args, X, Y, model)
                all_subject_ft_acc.append(iter_acc)
                all_subject_ft_f1.append(iter_f1)
                all_subject_ft_auc.append(iter_auc)
                pre_train.append(ba)
                tuning.append(final_acc)
                result_write.write('pre_training acc: %.6f\n' % ba)
                result_write.write('fine_tuning acc: %.6f\n' % final_acc)

    total_acc = np.asarray(total_acc, dtype=float)
    print('Mean source-train accuracy by fixed epoch:', total_acc.mean(axis=0))
    ft_acc = np.asarray(all_subject_ft_acc, dtype=float)
    ft_f1 = np.asarray(all_subject_ft_f1, dtype=float)
    ft_auc = np.asarray(all_subject_ft_auc, dtype=float)
    mean_ft_acc, mean_ft_f1, mean_ft_auc = float(ft_acc.mean()), float(ft_f1.mean()), float(ft_auc.mean())
    print('\n================= Final held-out target results =================')
    print('Fixed fine-tuning iterations:', args.max_iter2)
    print('Mean accuracy = %.4f' % mean_ft_acc)
    print('Mean weighted F1 = %.4f' % mean_ft_f1)
    print('Mean AUC = %.4f' % mean_ft_auc)
    with open("snapshot.txt", "a") as result_write:
        result_write.write('Seeds: %s\n' % seed_values)
        result_write.write('Fixed pre-training epochs: %d\n' % args.n_epochs)
        result_write.write('Fixed fine-tuning iterations: %d\n' % args.max_iter2)
        result_write.write('Mean final target accuracy: %.6f\n' % mean_ft_acc)
        result_write.write('Mean final target weighted F1: %.6f\n' % mean_ft_f1)
        result_write.write('Mean final target AUC: %.6f\n' % mean_ft_auc)


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
    parser.add_argument('--max_iter2', type=int, default=33)
    parser.add_argument('--batch_size',type=int,default=50)
    parser.add_argument('--batch_size_fine',type=int,default=32)
    parser.add_argument('--seed', type=int, default=1, help="first of five random seeds")
    parser.add_argument('--num_seeds', type=int, default=5, help="number of independent random seeds")
    parser.add_argument('--hidden_size', type=int, default=512, help="Bottleneck (features) dimensionality")
    parser.add_argument('--bottleneck_dim', type=int, default=256, help="Bottleneck (features) dimensionality")
    parser.add_argument('--session', type=int, default=1, help="random seed number ")
    parser.add_argument('--n_epochs', type=int, default=100, help="fixed pre-training epochs")
    parser.add_argument('--beta', type=float, default=1.0, help="MMD loss weight")
    parser.add_argument('--gamma', type=float, default=1.0, help="adversarial loss weight")
    parser.add_argument('--file_path', type=str, default="E:/Research/EEGDataSet/BNCI2014002/feature", help="Path from the current dataset")
    parser.add_argument('--log_file')
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--d_classes', type=int, default=13)
    parser.add_argument('--window_size', type=int, default=5)
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
