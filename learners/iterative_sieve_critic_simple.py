from copy import deepcopy
import sys

import torch
from torch.optim import Adam, LBFGS
import torch.nn as nn
import numpy as np

from learners.abstract_learner import AbstractLearner
from learners.iterative_sieve_critic import IterativeSieveLearner
from models.abstract_critic import AbstractCritic
from utils.oadam import OAdam


class IterativeSieveLearnerSimple(IterativeSieveLearner):

    def get_omega_inv(self, critic, dl, pi_e_name, s_init, gamma_tik,
                      batch_scale=1000.0, device=None):
        self.model.eval()
        f_mat = 0
        batch_size_sum = 0
        for batch in dl:
            f_basis = self.get_critic_basis_expansion(
                batch=batch, critic=critic
            )
            next_mat = torch.einsum("bkn,blm->knlm", f_basis, f_basis)
            f_mat = f_mat + (next_mat / batch_scale)
            batch_size_sum += len(batch["s"]) / batch_scale

        self.model.train()
        return f_mat / batch_size_sum
