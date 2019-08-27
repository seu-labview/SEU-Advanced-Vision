# -*- coding: utf-8 -*-
import random
from torch.utils.data import Dataset


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, objclass=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4, bg_file_names=None):
        with open(root,'r') as myfile:
            self.lines = myfile.readlines()
        if shuffle:
            random.shuffle(self.lines)
        self.nSamples         = len(self.lines)
        self.transform        = transform
        self.target_transform = target_transform
        self.train            = train
        self.shape            = shape
        self.seen             = seen
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.bg_file_names    = bg_file_names
        self.objclass         = objclass
       
        self.cell_size = 32