from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import numpy as np

class Dataset(data.Dataset):
    """EEG feature arrays paired with class labels."""
    def __init__(self, data, label,
                 transform=None,target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = label

    def __getitem__(self, index):
        """Return the feature array and label at index."""

        img, target = self.data[index], self.labels[index]

        return img, target
    def __len__(self):
        return len(self.data)
