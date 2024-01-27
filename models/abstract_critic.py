import torch
import torch.nn as nn

class AbstractCritic(nn.Module):
    def __init__(self, num_out, s_dim, num_a):
        super().__init__()
        self.s_dim = s_dim
        self.num_a = num_a
        self.num_out = num_out
