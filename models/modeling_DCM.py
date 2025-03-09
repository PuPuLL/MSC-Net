from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import math
import sys

from tools.calculate import correlation_calculation
from sklearn.metrics.pairwise import cosine_similarity


class age_predictor(nn.Module):
    def __init__(self, embed_dim, batch_size):
        super().__init__()
        '''
        Linear
        '''
        self.layers = nn.Sequential(
            nn.Linear(in_features=264*embed_dim, out_features=768),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=768),
        )
        self.regressor = nn.Linear(in_features=768, out_features=1)


    def forward(self, x):
        batch_size = x.size(0)

        '''
        Linear
        '''
        x = self.layers(x.view(batch_size, -1))
        x = self.regressor(x)

        return x