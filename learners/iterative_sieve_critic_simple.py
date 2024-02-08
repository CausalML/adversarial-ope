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

    def critic_lbfgs_closure_simple(self, optim, dl, critic, s_init, pi_e_name,
                                    reg_alpha, batch_scale=1000.0):
        optim.zero_grad()
        obj_sum = 0
        reg_sum = 0
        batch_size_sum = 0

        for batch in dl:
            moments = self.get_batch_moments(
                batch=batch, critic=critic, s_init=s_init,
                pi_e_name=pi_e_name, critic_grad=True,
            )
            f_l2 = self.get_batch_l2_reg_critic(
                batch=batch, critic=critic
            ) 
            obj = moments.sum() - 0.5 * f_l2 * len(batch["s"])
            if reg_alpha:
                reg = reg_alpha * critic.get_next_func_batch_reg(
                    batch=batch, train_q_beta=self.train_q_beta,
                    train_eta=self.train_eta, train_w=self.train_w,
                )
            else:
                reg = 0

            batch_n = len(batch["s"])
            obj_sum += obj / batch_scale
            reg_sum += reg * batch_n / batch_scale
            batch_size_sum += batch_n / batch_scale

        obj_loss = -1.0 * obj_sum / batch_size_sum
        reg_loss = reg_sum / batch_size_sum
        loss = obj_loss + reg_loss
        loss.backward()
        return loss

    def train_next_critic(self, critic, dl, dl_val, pi_e_name, s_init,
                          max_num_epoch, min_num_epoch, max_no_improve,
                          critic_reg_alpha, eval_freq, lr, iter_i,
                          verbose=False):
        self.model.train()
        critic.train()
        critic_optim = LBFGS(critic.parameters(),
                             line_search_fn="strong_wolfe")
        closure = lambda : self.critic_lbfgs_closure_simple(
            optim=critic_optim, dl=dl, critic=critic, s_init=s_init,
            pi_e_name=pi_e_name, reg_alpha=critic_reg_alpha,
        )
        critic_optim.step(closure)

        if verbose:
            val_loss = self.get_mean_critic_loss(
                critic=critic, dl=dl_val,
                pi_e_name=pi_e_name, s_init=s_init,
            )
            train_loss = self.get_mean_critic_loss(
                critic=critic, dl=dl,
                pi_e_name=pi_e_name, s_init=s_init,
            )
            print(f"CRITIC: iter {iter_i}")
            print(f"mean train loss: {train_loss}")
            print(f"mean val loss: {val_loss}")
            print("")

        self.model.eval()
        critic.eval()

    def get_mean_critic_loss(self, critic, dl, pi_e_name, s_init,
                             batch_scale=1000.0):
        self.model.eval()
        critic.eval()
        loss_sum = 0
        batch_size_sum = 0
        for batch in dl:
            moments = self.get_batch_moments(
                batch=batch, critic=critic, s_init=s_init,
                pi_e_name=pi_e_name
            )
            f_l2 = self.get_batch_l2_reg_critic(
                batch=batch, critic=critic
            ) 
            obj = moments.sum() - 0.5 * f_l2 * len(batch["s"])
            loss = (-1.0 * obj)

            loss_sum += float(loss) / batch_scale
            batch_size_sum += len(batch["s"]) / batch_scale

        self.model.train()
        critic.train()

        return loss_sum / batch_size_sum
