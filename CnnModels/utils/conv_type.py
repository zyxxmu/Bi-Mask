import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from utils.options import args
import utils.common as utils
import numpy as np
import os
LearnedBatchNorm = nn.BatchNorm2d
N = args.N
M = args.M
I = args.iter
G = args.greedy_num
import math
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


DenseConv = nn.Conv2d

def get_n_m_sparse_matrix(w):
    length = w.numel()
    group = int(length / M)
    w_tmp = w.t().detach().abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(w.t().shape).t()
    return w * mask, mask

class MyConv2d(autograd.Function):
    @staticmethod
    def forward(ctx, weight, inp_unf, forward_mask, backward_mask, decay = 0.0002):
        ctx.save_for_backward(weight, inp_unf, backward_mask)
        w_s = weight * forward_mask

        ctx.decay = decay
        ctx.mask = forward_mask

        out_unf = inp_unf.matmul(w_s)
        return out_unf

    @staticmethod
    def backward(ctx, g):
        weight, inp_unf, backward_mask = ctx.saved_tensors
        w_s = (weight * backward_mask).t()

        g_w_s = inp_unf.transpose(1,2).matmul(g).sum(0)
        g_w_s = g_w_s + ctx.decay * (1 - ctx.mask) * weight
        g_inp_unf = g.matmul(w_s)
        return g_w_s , g_inp_unf, None, None, None


def get_best_permutation(w):
    length = w.numel()
    group = int(length / M)
    w = w.t()
    mask_sum_max = 0
    mask_sum_min = 1e8
    num_perm = G
    permutation = [v for v in range(w.size(0))]     
    p = np.random.permutation(permutation)        
    best_perm = permutation.copy()                  
    w_tmp = w.detach().abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask_f = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask_f = mask_f.scatter_(dim=1, index=index, value=0).reshape(w.shape)      

    for i in range(num_perm):
        mask = mask_f[p]
        w_t = w[p]
        # backward mask
        w_s = w_t * mask
        w_backward = w_s.t()
        
        w_tmp = w_backward.abs().reshape(group, M)
        index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
        mask_b = torch.ones(w_tmp.shape, device=w_tmp.device)
        mask_b = mask_b.scatter_(dim=1, index=index, value=0).reshape(w_backward.shape).t()
        final_mask = mask * mask_b
        if final_mask.sum() > mask_sum_max:
            mask_sum_max = final_mask.sum()
            # update thet best permutation
            best_perm = p
        # update permutation
        p = np.random.permutation(permutation)
    return best_perm



def get_n_m_backward_matrix(forward_mask, w_s, permutation):
    # import pdb; pdb.set_trace()
    w_s = w_s.t()
    forward_mask = forward_mask.t()
    length = w_s.numel()
    group = int(length / M)

    # change with the permutation
    mask = forward_mask[permutation]
    w_backward = w_s[permutation]
    w_backward = w_backward.t()
    
    # get the backwardmask
    w_tmp = w_backward.abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask_b = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask_b = mask_b.scatter_(dim=1, index=index, value=0).reshape(w_backward.shape).t()
    final_mask = mask * mask_b

    # recover the backwardmask with best permutation
    idx = torch.tensor(np.array([permutation])).t().repeat_interleave(w_s.size(1), dim=1)
    idx = idx.to(device=w_s.device)
    B_M = torch.zeros(w_s.size(), device=w_s.device).scatter_(dim=0, index=idx, src=final_mask).t()

    return B_M



# Bi-Mask
class NMConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N  # number of non-zeros
        self.M = M
        self.flag = False
        self.iter = 0
        self.max_iter = I
        self.permute_idx = [v for v in range(self.weight.view(self.weight.size(0), -1).size(0))]
        self.forward_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape).requires_grad_(False)
        self.backward_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape).requires_grad_(False)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)

    def forward(self, x):
        w = self.weight.view(self.weight.size(0), -1).t()
        if self.iter % self.max_iter == 0:
            self.permute_idx = get_best_permutation(w)
        w_s, self.forward_mask = get_n_m_sparse_matrix(w)
        self.backward_mask = get_n_m_backward_matrix(self.forward_mask, w_s, self.permute_idx)   
        inp_unf = self.unfold(x)
        out_unf = MyConv2d.apply(w, inp_unf.transpose(1, 2), self.forward_mask, self.backward_mask)
        if self.flag == False:
            self.fold = nn.Fold((int(math.sqrt(out_unf.shape[1])), int(math.sqrt(out_unf.shape[1]))), (1,1))
            self.flag = True
        out = self.fold(out_unf.transpose(1, 2))
        return out
