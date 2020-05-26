import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class period_L1(nn.Module):
    def __init__(self, reduction='sum'):
        '''
        periodic Squared Error
        '''
        super().__init__()
        self.reduction = reduction

    def forward(self, theta_pred, theta_gt):
        # assert theta_pred.shape == theta_gt.shape
        dt = theta_pred - theta_gt

        # periodic SE
        dt = torch.abs(torch.remainder(dt-np.pi/2,np.pi) - np.pi/2)
        
        assert (dt >= 0).all()
        if self.reduction == 'sum':
            loss = dt.sum()
        elif self.reduction == 'mean':
            loss = dt.mean()
        elif self.reduction == 'none':
            loss = dt
        return loss


class period_L2(nn.Module):
    def __init__(self, reduction='sum'):
        '''
        periodic Squared Error
        '''
        super().__init__()
        if reduction == 'sum':
            self.reduction = reduction_sum
        elif reduction == 'mean':
            self.reduction = reduction_mean
        elif reduction == 'none':
            self.reduction = reduction_none
        else:
            raise Exception('unknown reduction')

    def forward(self, theta_pred, theta_gt):
        # assert theta_pred.shape == theta_gt.shape
        dt = theta_pred - theta_gt
        # periodic SE
        loss = (torch.remainder(dt-np.pi/2,np.pi) - np.pi/2) ** 2
        
        assert (loss >= 0).all()
        loss = self.reduction(loss)
        return loss


def reduction_sum(loss):
    return loss.sum()

def reduction_mean(loss):
    return loss.mean()

def reduction_none(loss):
    return loss
