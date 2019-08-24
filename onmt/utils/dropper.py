import numpy as np
import torch
import torch.nn as nn

class LossDropper(nn.Module):
    def __init__(self, expc=0.999, dropc=0.9, min_count=1000):
        super().__init__()
        self.dropc = dropc
        self.count = 0
        self.min_count = min_count
        self.vals = np.zeros(self.min_count, dtype=np.float32)

        # FIXME
        self.recompute = 1000
        self.last_computed = 0
        self.percentil_val = 100000000.
        self.cur_idx = 0

    def forward(self, loss):
        if loss is None:
            return loss

        self.last_computed += loss.numel()
        self.count += loss.numel()
        if self.count < len(self.vals):
            self.vals[self.count - loss.numel() : self.count] = loss.detach().cpu().numpy()
            self.cur_idx += loss.numel()
            return loss
        else:
            for idx, item in enumerate(loss):
                self.vals[self.cur_idx] = item
                self.cur_idx += 1
                if self.cur_idx >= len(self.vals):
                    self.cur_idx = 0

        if self.last_computed > self.recompute:
            self.percentile_val = np.percentile(self.vals, self.dropc * 100)
            print('Using cutoff', self.percentile_val)
            self.last_computed = 0

        mask = (loss < self.percentile_val).type(loss.dtype)
        loss *= mask
        return loss

class MovingLossDropper(nn.Module):
    def __init__(self, expc=0.999, dropc=0.9, min_count=1000):
        super().__init__()
        self.expc = expc
        self.dropc = dropc
        self.min_count = min_count
        self.count = 0
        self.percentile_val = None
        self.avg = None
        self.var = None

    def forward(self, loss):
        if loss is None:
            return loss
        self.count += loss.numel()
        if self.avg is None:
            self.avg = loss.mean().item()
        else:
            self.avg = self.avg * self.expc + loss.mean().item() * (1. - self.expc)
        if self.var is None:
            self.var = (loss - self.avg)**2
            self.var = self.var.mean().item()
        else:
            tmp = (loss - self.avg)**2
            tmp = tmp.mean().item()
            self.var = self.var * self.expc + tmp * (1. - self.expc)

        if self.percentile_val is None:
            self.percentile_val = self.avg
        else:
            delta = 0.001 * self.var
            for tmp in loss:
                if tmp > self.percentile_val:
                    self.percentile_val += delta / (1 - self.dropc)
                else:
                    self.percentile_val -= delta / self.dropc

        if self.count > self.min_count:
            # print(loss)
            mask = (loss < self.percentile_val).type(loss.dtype)
            # print(mask)
            # print(self.avg, self.var, self.percentile_val)
            loss *= mask
            # print(loss)
        return loss
