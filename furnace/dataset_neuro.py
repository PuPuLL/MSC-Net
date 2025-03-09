import math
import random
import numpy as np
import os
import torch
from scipy.io import loadmat
from torchvision import datasets, transforms
from timm.data import create_transform


import sys

import nibabel as nib
from torch.utils.data import Dataset
from furnace.masking_generator import RandomMaskingGenerator
from sklearn.preprocessing import MinMaxScaler

class Bulid_pretrain_dataset(Dataset):

    def __init__(self, root_path, fc_root_path, ts_name, label, length, args):
        self.root = root_path
        self.root_fc = fc_root_path
        self.ts = ts_name
        self.label = label
        self.dataname = args.dataset
        self.length = length
        self.embed_dim = 264

        self.patch_transform = transforms.Compose([
            transforms.ToTensor()])

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor()])

        self.masked_position_generator = RandomMaskingGenerator(
            args.window_size, self.length,
            ratio_masking_patches=args.ratio_mask_patches
        )

    def __len__(self):
        return len(self.ts)

    def label_convert(self, label):
        if self.dataname == 'ADHD':
            ts_label = 0 if label == '0' else 1
        elif self.dataname  == 'ABIDE1':
            ts_label = 0 if label == '1' else 1
        else:
            ts_label = 0 if label == 'Control' else 1

        return ts_label

    def h_label_convert(self, label):

        if label[0] == '0':
            dx_label = 0
        elif label[0] in ['1', '2', '3']:
            dx_label = 1

        if label[1] == '1':
            sex_label = 0
        elif label[1] == '2':
            sex_label = 1

        label_list = [sex_label, float(label[2])]

        label_arr = np.array(label_list)

        return label_arr


    def normalize(self, matrix):
        mean = np.mean(matrix)
        std = np.std(matrix)
        normalized_matrix = (matrix - mean) / std
        return normalized_matrix

    def __getitem__(self, idx):

        if len(self.ts[idx]) < 7 :
            ts = self.ts[idx].zfill(7)
        else:
            ts = self.ts[idx]
            
        ts_name = 'sub-' + ts + '.mat'

        filepath = os.path.join(self.root, ts_name)
        filepath_fc = os.path.join(self.root_fc, ts_name)

        h_label = self.h_label_convert(self.label[idx])

        ts = loadmat(filepath)['ts']

        if ts.shape[1] < self.length:
            ts = np.pad(ts, ((0, 0), (0, self.length - ts.shape[1])), mode='constant')

        ts = self.normalize(ts)
        fc = loadmat(filepath_fc)['fc']

        return \
            self.patch_transform(ts), self.visual_token_transform(fc), \
                self.masked_position_generator(), h_label

    def __repr__(self):
        repr = "(DataAugmentationForMSC,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


class Bulid_finetune_dataset(Dataset):
    """
    load nifiti image as dataset {image: value, label: label}
    """

    def __init__(self, root_path, ts_name, label, length, args):
        self.root = root_path
        self.ts = ts_name
        self.label = label
        self.dataname = args.dataset
        self.length = length
        self.args = args


        if isinstance(args.input_size, tuple):
            self.size = args.input_size
        else:
            self.size = (args.input_size, self.length)

        self.ts_token_transform = transforms.Compose([
            transforms.ToTensor()])

    def __len__(self):
        return len(self.ts)

    def label_convert(self, label):
        if self.dataname == 'ADHD':
            if label == '0':
                ts_label = 0
            elif label in ['1', '2', '3']:
                ts_label = 1
        elif self.dataname == 'ABIDE1':
            if label == '1':
                ts_label = 0
            elif label in ['2', '3']:
                ts_label = 1
        else:
            if label == 'Control':
                ts_label = 0
            elif label == 'Patient':
                ts_label = 1

        return ts_label


    def normalize(self, matrix):
        mean = np.mean(matrix)
        std = np.std(matrix)
        normalized_matrix = (matrix - mean) / std
        return normalized_matrix

    def __getitem__(self, idx):
        if len(self.ts[idx]) < 7:
            ts = self.ts[idx].zfill(7)
        else:
            ts = self.ts[idx]
            
        ts_name = 'sub-' + ts + '.mat'

        label = self.label_convert(str(self.label[idx]).strip())

        filepath = os.path.join(self.root, ts_name)

        ts = loadmat(filepath)['ts']

        if ts.shape[1] < self.length:
            ts = np.pad(ts, ((0, 0), (0, self.length - ts.shape[1])), mode='constant')

        elif ts.shape[1] > self.length:
            start = random.randint(0, ts.shape[1] - self.length)
            end = start + self.length
            ts = ts[:, start:end]

        ts = self.normalize(ts)


        return self.ts_token_transform(ts), label
