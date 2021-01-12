#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.model == 'audioVisual':
        print('Loading audioVisual dataset')
        from data.audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    elif opt.model == 'tasnet' or opt.model == 'demucs':
        print('Loading tasnet dataset')
        from data.tasnet_dataset import TasnetDataset
        dataset = TasnetDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.model)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        if opt.mode != "test":
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.stepBatchSize,
                shuffle=opt.mode== "train",
                num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
