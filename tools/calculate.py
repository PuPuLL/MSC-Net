from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
from scipy.io import loadmat
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, auc

def sencitivity_specificity(output, target):
    tn, fp, fn, tp = confusion_matrix(np.array(target.cpu()), np.array(output.cpu()).ravel()).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)

    return sensitivity, specificity


def max_length(root_path, all_train, dataset):
    max_len = 0
    for item in all_train:
        if len(item) < 7 and dataset != 'ABIDE2':
            item = item.zfill(7)
        if dataset == 'CamCAN':
            file_name = item
        else:
            file_name = 'sub-' + item
        filepath = os.path.join(root_path, file_name)
        fc = loadmat(filepath)['ts']
        temp = fc.shape[-1]
        if temp > max_len:
            max_len = temp

    return max_len

class Symmetric_loss(torch.nn.Module):
    def __init__(self, batch_size, length, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.t = torch.nn.Parameter(torch.tensor(0.07))


        self.softmax = torch.nn.Softmax(dim=1)
        self.batch_norm = torch.nn.BatchNorm1d(264)
        self.batch_norm_RCM = torch.nn.BatchNorm1d(264*768)
        self.batch_norm_net = torch.nn.BatchNorm1d(264*264)

    def cosine_similarity(self, A, B):
        
        A_norm = torch.sqrt(torch.sum(A ** 2, dim=1, keepdim=True))  # [8, 1]
        B_norm = torch.sqrt(torch.sum(B ** 2, dim=1, keepdim=True))  # [8, 1]

        
        A_expanded = A.unsqueeze(1)  # [8, 1, 128]
        B_expanded = B.unsqueeze(0)  # [1, 8, 128]

        
        dot_product = torch.sum(A_expanded * B_expanded, dim=2)  # [8, 8]

        
        norm_product = A_norm * B_norm.T  # [8, 8]

        
        cosine_similarity = dot_product / norm_product  # [8, 8]
        scaled_similarity = cosine_similarity * self.t

        return scaled_similarity

    def normalize_tensor(self, tensor, dim=-1, eps=1e-8):

        norm = torch.norm(tensor, p=2, dim=dim, keepdim=True) + eps
        return tensor / norm

    def normalize_to_neg1_1(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()
        # Scale to [0, 1]
        tensor = (tensor - min_val) / (max_val - min_val)
        # Scale to [-1, 1]
        tensor = tensor * 2 - 1
        return tensor


    def symmetric_cross_entropy_torch(self, pred_matrix, label_matrix):


        L_CE = F.cross_entropy(pred_matrix, label_matrix)
        
        L_RCE = F.cross_entropy(pred_matrix.t(), label_matrix)
        
        L_SCE = (L_CE + L_RCE) / 2
        return L_SCE

    def symmetric_mse_loss(self, pred_matrix, label_matrix):

        pred_matrix = self.normalize_to_neg1_1(pred_matrix)
        
        L_MSE = F.mse_loss(pred_matrix, label_matrix)

        return L_MSE


    def loss_compute(self, logits_pred, logits_target, labels, flag_age=False, flag_sex=False):

        if flag_age:
            target_matrix = (logits_pred.view(-1, 1) - logits_target.view(1, -1)).float()
            target_matrix = self.normalize_to_neg1_1(target_matrix)
            loss = self.symmetric_mse_loss(target_matrix, labels)

        else:
            target_matrix = self.cosine_similarity(logits_pred, logits_target)
            loss = self.symmetric_cross_entropy_torch(target_matrix * self.t, labels)

        return loss, target_matrix


    def forward(self, h_label, logits_pred, logits_target, fc_masked, fc_ori):
        labels = torch.arange(self.batch_size, device=h_label.device)

        sex = h_label[:, 0]
        sex_matrix = torch.eq(sex.view(-1, 1), sex.view(1, -1)).float()

        age = h_label[:, 1]
        age_diff_matrix = (age.view(-1, 1) - age.view(1, -1)).float()
        age_diff_matrix = self.normalize_to_neg1_1(age_diff_matrix)

        predict_RCM, predict_age = logits_pred
        target_RCM, target_age = logits_target

        predict_RCM = self.batch_norm_RCM(predict_RCM.view(self.batch_size, -1))
        target_RCM = self.batch_norm_RCM(target_RCM.view(self.batch_size, -1))

        loss_RCM, sim_RCM = self.loss_compute(predict_RCM, target_RCM, labels)

        loss_age, sim_age = self.loss_compute(predict_age, target_age, age_diff_matrix, flag_age=True)

        fc_masked = self.batch_norm_net(fc_masked.squeeze().view(self.batch_size, -1))
        fc_ori = self.batch_norm_net(fc_ori.squeeze().view(self.batch_size, -1))
        loss_NCM, sim_net = self.loss_compute(fc_masked,
                                              fc_ori, labels)

        return [loss_NCM, loss_age, loss_RCM]


def correlation_calculation(ts):
    epsilon = 1e-6
    # Reshape to remove the channel dimension (64, 264, 256)
    X_transposed = ts.squeeze(1)

    # Transpose to get features as columns (64, 256, 264)
    # X_transposed = X.permute(0, 2, 1)

    # Center the data by subtracting the mean along the time step dimension
    X_mean = X_transposed.mean(dim=2, keepdim=True)
    X_centered = X_transposed - X_mean

    # Calculate covariance matrix: batch-wise matrix multiplication (64, 264, 264)
    cov_matrix = torch.bmm(X_centered, X_centered.transpose(1, 2)) / (X_transposed.size(2) - 1)

    # Calculate standard deviations for each time step (64, 264)
    stddev = X_transposed.std(dim=2)

    # Compute the correlation matrix (64, 264, 264)
    stddev_matrix = stddev.unsqueeze(2) * stddev.unsqueeze(1)
    corr_matrix = cov_matrix / stddev_matrix

    # Add channel dimension back, so the shape is (64, 1, 264, 264)
    corr_matrix = corr_matrix.unsqueeze(1)

    r = corr_matrix.clamp(min=-1 + epsilon, max=1 - epsilon)
    corr_matrix_z = 0.5 * torch.log((1 + r) / (1 - r))

    return corr_matrix_z
