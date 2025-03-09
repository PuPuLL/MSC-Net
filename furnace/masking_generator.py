import random
import math
import numpy as np


class RandomMaskingGenerator:
    def __init__(
            self, input_size, max_length, ratio_masking_patches):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size
        self.max_length = max_length

        self.num_patches = 1 * self.max_length
        self.num_masking_patches = int(ratio_masking_patches * self.num_patches)


    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_masking_patches
        )
        return repr_str

    def __call__(self):
        mask = np.zeros((264, self.max_length))

        for i in range(mask.shape[0]):
            mask[i] = np.hstack([
                np.zeros(self.num_patches - self.num_masking_patches),
                np.ones(self.num_masking_patches),
            ])
        np.random.shuffle(mask)

        return mask
