from abc import ABC, abstractmethod

import torch

from models.abstract_nuisance_model import AbstractNuisanceModel

class AbstractLearner(ABC):
    def __init__(self, nuisance_model, gamma, adversarial_lambda,
                 worst_case=True):
        self.model = nuisance_model
        self.gamma = gamma
        self.adversarial_lambda = adversarial_lambda
        self.worst_case = worst_case
        assert isinstance(self.model, AbstractNuisanceModel)
        super().__init__()

    def rho_q_xi(self, s, a, ss, r, pi_ss):
        q, v, xi = self.model.get_q_v_xi(s, a, ss, pi_ss)
        lmbda = self.adversarial_lambda
        inv_lmbda = lmbda ** -1
        alpha = 1 / (1 + lmbda)
        assert inv_lmbda > 0
        assert inv_lmbda <= 1
        e_cvar_v = (inv_lmbda + (1 - inv_lmbda) * (1 + lmbda) * xi) * v
        rho_1 = q  - r.unsqueeze(-1) - self.gamma * e_cvar_v
        if self.worst_case:
            # estimating worst-case Q function within sensitivity model
            rho_2 = xi - alpha
        else:
            # esimating best-case Q function within sensitivity model
            rho_2 = xi - (1 - alpha)
        return torch.cat([rho_1, rho_2], dim=1)

    def estimate_policy_val_q(self, init_s, init_a):
        q = self.model.get_q(init_s.unsqueeze(0), init_a)
        return float(q[0])


    @abstractmethod
    def train(self, dataset, pi_e_name):
        pass