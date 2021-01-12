#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseLoss(nn.Module):
    def __init__(self):
        super(BaseLoss, self).__init__()

    def forward(self, preds, targets, weight=None):
        if isinstance(preds, list):
            N = len(preds)
            if weight is None:
                weight = preds[0].new_ones(1)

            errs = [self._forward(preds[n], targets[n], weight[n])
                    for n in range(N)]
            err = torch.mean(torch.stack(errs))

        elif isinstance(preds, torch.Tensor):
            if weight is None:
                weight = preds.new_ones(1)
            err = self._forward(preds, targets, weight)

        return err


class L1Loss(BaseLoss):
    def __init__(self):
        super(L1Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.abs(pred - target))


class L2Loss(BaseLoss):
    def __init__(self):
        super(L2Loss, self).__init__()

    def _forward(self, pred, target, weight):
        return torch.mean(weight * torch.pow(pred - target, 2))

class MSELoss(BaseLoss):
    def __init__(self):
        super(MSELoss, self).__init__()

    def _forward(self, pred, target):
        return F.mse_loss(pred, target)

class BCELoss(BaseLoss):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy(pred, target, weight=weight)

class BCEWithLogitsLoss(BaseLoss):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()

    def _forward(self, pred, target, weight):
        return F.binary_cross_entropy_with_logits(pred, target, weight=weight)

class LeastLoss(nn.Module):
    def __init__(self, loss):
        super(LeastLoss, self).__init__()
        self.loss = loss

    def forward(self, pred, target):
        first_loss = self.loss(pred, target)
        permuted_pred = pred[:,[1,0],:]
        second_loss = self.loss(permuted_pred, target)
        if first_loss.item() < second_loss.item():
            return first_loss
        return second_loss

class SISNRLoss(nn.Module):
    def __init__(self, PIT=False, negate=False):
        super(SISNRLoss, self).__init__()
        self.PIT = PIT
        self.negate = negate
        
    def sum_of_squares(self, x):
        return torch.sum(x**2, dim = 2).unsqueeze(2)

    def dot_product(self, a, b):
        return torch.sum(torch.mul(a,b),dim=2).unsqueeze(2)

    def calc_loss(self, x, y):
        s_target = torch.true_divide(self.dot_product(x, y) * y,self.sum_of_squares(y))
        e_noise = x - s_target
        l = 10*torch.log10(self.sum_of_squares(s_target)/self.sum_of_squares(e_noise))
        return torch.mean(l) * (-1 if self.negate else 1)

    def forward(self, pred, target):
        '''
        Scale invariant source-to-noise loss
        PIT = Permutation invariant training:
            The final loss is calculated comparing the predicted channels
            with the target channels that minimize the loss
        '''
        normalize = lambda x: x - torch.mean(x, dim = 2).unsqueeze(2)

        normalized_pred = normalize(pred)
        normalized_target = normalize(target)
        first_loss = self.calc_loss(normalized_pred, normalized_target)
        if not self.PIT:
            return first_loss
        permuted_pred = normalized_pred[:,[1,0],:]
        second_loss = self.calc_loss(permuted_pred, normalized_target)
        if first_loss > second_loss:
            return first_loss
        else:
            return second_loss
