# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import torch
from scipy.io import savemat
import numpy as np
from timm.utils import accuracy
import furnace.utils as utils
from calculate import sencitivity_specificity
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def test(data_loader, model, device):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    mf_list, label_list = [], []
    embeddings_list = []
    output_list, attn_list = [], []

    model.eval()
    softmax = torch.nn.Softmax(dim=1).to(device)  
    attn_head_avg_single = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        ts = batch[0]
        target = batch[-1]

        target = target.to(device, non_blocking=True)
        label_list.append(target)

        ts = ts.float().to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(ts)
            output_list.append(output)

            preds = softmax(output)
            _, pred_small = preds.topk(1, dim=1, largest=True, sorted=True)
            mf_list.append(pred_small.squeeze())  
            embeddings_list.append(output.cpu().numpy())

    mf = torch.cat(mf_list).view(-1, 1) 
    label = torch.cat(label_list).view(-1)
    outputs = torch.cat(output_list)

    acc = accuracy(outputs, label, topk=(1,))[0].cpu().detach().numpy()

    metric_logger.meters["test_acc"].update(acc)

    sensitivity, specificity = sencitivity_specificity(mf, label)


    metric_logger.synchronize_between_processes()

    return acc, sensitivity * 100, specificity * 100, label, mf
