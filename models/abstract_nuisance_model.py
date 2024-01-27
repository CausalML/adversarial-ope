from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractNuisanceModel(ABC):
    def __init__(self, s_dim, num_a):
        super().__init__()
        self.s_dim = s_dim
        self.num_a = num_a

    @abstractmethod
    def get_q(self, s, a):
        pass

    @abstractmethod
    def get_q_v_xi(self, s, a, ss, pi_ss):
        pass

    @abstractmethod
    def get_w(self, s, a):
        pass

    @abstractmethod
    def get_all(self, s, a, ss, pi_ss):
        pass

    @abstractmethod
    def get_parameters(self):
        pass