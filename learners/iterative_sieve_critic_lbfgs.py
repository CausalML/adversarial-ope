from copy import deepcopy
import sys

import torch
from torch.optim import LBFGS
import torch.nn as nn
import numpy as np

from learners.iterative_sieve_critic import IterativeSieveLearner
from models.abstract_critic import AbstractCritic
from utils.oadam import OAdam


class IterativeSieveLearnerLBFGS(IterativeSieveLearner):

    def model_lbfgs_closure(self, optim, dl, critic, omega, s_init, pi_e_name,
                            m_scale, quantile_scale, reg_alpha,
                            batch_scale=1000.0):
        optim.zero_grad()
        rho_f_sum = 0
        reg_sum = 0
        beta_loss_sum = 0
        batch_size_sum = 0

        for batch in dl:
            moments = self.get_batch_moments(
                batch=batch, critic=critic, s_init=s_init,
                pi_e_name=pi_e_name, basis_expansion=True,
            )
            rho_f = moments.sum(0)
            rho_f_sum = rho_f_sum + rho_f.view(-1) / batch_scale
            beta_loss = self.get_batch_quantile_loss(
                batch=batch, pi_e_name=pi_e_name,
            )
            if reg_alpha:
                loss_reg = reg_alpha * self.get_batch_l2_reg_model(
                    batch=batch, pi_e_name=pi_e_name,
                )
            else:
                loss_reg = 0
            batch_n = len(batch["s"])
            beta_loss_sum += beta_loss * batch_n / batch_scale
            reg_sum += loss_reg * batch_n / batch_scale
            batch_size_sum += batch_n / batch_scale

        rho_f_mean = rho_f_sum / batch_size_sum
        m_loss = torch.einsum("xy,x,y->", omega, rho_f_mean, rho_f_mean)
        beta_loss = beta_loss_sum / batch_size_sum
        reg_loss = reg_sum / batch_size_sum
        loss = m_scale * m_loss + quantile_scale * beta_loss + reg_loss
        loss.backward()
        return loss

    def critic_lbfgs_closure(self, optim, dl, critic, s_init, pi_e_name,
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
            obj = moments.sum() - 0.5 * (moments ** 2).sum()
            if reg_alpha:
                reg_1 = self.get_batch_l2_reg_critic(
                    batch=batch, critic=critic,
                )
                reg_2 = critic.get_next_func_batch_reg(
                    batch=batch, train_q_beta=self.train_q_beta,
                    train_eta=self.train_eta, train_w=self.train_w,
                )
                reg = reg_alpha * (reg_1 + reg_2)
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

    def update_model(self, critic, dl, dl_2, dl_val, pi_e_name, max_num_epoch,
                     s_init, min_num_epoch, max_no_improve, grad_clip,
                     eval_freq, lr, gamma_0, gamma_tik, iter_i, reg_alpha,
                     m_scale, quantile_scale, device, verbose=False,
                     eig_threshold=1e-5):
        self.model.train()
        # first compute omega "weighting" matrix for loss function
        num_fail = 0
        while True:
            omega_inv = self.get_omega_inv(
                critic, dl, s_init=s_init, pi_e_name=pi_e_name,
                gamma_tik=gamma_tik, device=device
            )
            num_param = self.get_num_moments() * critic.get_num_basis_func()
            omega_inv_np = omega_inv.reshape(num_param, num_param).cpu().double().numpy()
            omega_inv_np = omega_inv_np + gamma_0 * np.eye(num_param)
            omega_np = np.linalg.inv(omega_inv_np)
            omega_np = (omega_np + omega_np.T) / 2.0

            # double check that Omega is PD, if not then repeat calculation
            # with extra regularization
            min_eig = np.linalg.eigvals(omega_np).real.min()
            if min_eig < eig_threshold:
                gamma_0 = gamma_0 * 10.0
                if verbose:
                    print(f"WARNING: BAD OMEGA (eig: {min_eig})")
                    print(f"REPEATING WITH gamma_0={gamma_0}")
                num_fail += 1
                if num_fail >= 10:
                    print("STUCK IN LOOP, ABORTING")
                    sys.exit(1)
            else:
                break
        omega = torch.FloatTensor(omega_np).to(omega_inv.device)

        # second do SGD on quadratic weighted loss
        model_optim = LBFGS(self.model.get_parameters(),
                            line_search_fn="strong_wolfe")
        closure = lambda : self.model_lbfgs_closure(
            optim=model_optim, dl=dl, critic=critic, omega=omega,
            s_init=s_init, pi_e_name=pi_e_name, m_scale=m_scale,
            quantile_scale=quantile_scale, reg_alpha=reg_alpha,
        )
        model_optim.step(closure)

        val_losses = self.get_mean_model_loss(
            critic=critic, s_init=s_init,
            dl=dl_val, pi_e_name=pi_e_name, omega=omega,
        )
        val_m_loss, val_beta_loss, val_moment_losses = val_losses
        val_loss = float(m_scale * val_m_loss
                            + quantile_scale * val_beta_loss)
        if verbose:
            train_losses = self.get_mean_model_loss(
                critic=critic, s_init=s_init, dl=dl,
                pi_e_name=pi_e_name, omega=omega,
            )
            train_m_loss, train_beta_loss, train_moment_losses = train_losses
            train_loss = float(m_scale * train_m_loss
                                + quantile_scale * train_beta_loss)

            print(f"MODEL: iter {iter_i}")
            print(f"mean train loss: {train_loss},"
                    f" moment/beta losses: {train_m_loss}/{train_beta_loss}"
                    f" per moment: {train_moment_losses}")
            print(f"mean val loss: {val_loss},"
                    f" moment/beta losses: {val_m_loss}/{val_beta_loss}"
                    f" per moment: {val_moment_losses}")
            print("")

        best_state = deepcopy(self.model.get_state())

        self.model.eval()
        return float(val_m_loss), float(val_beta_loss), best_state

    def train_next_critic(self, critic, dl, dl_val, pi_e_name, s_init,
                          max_num_epoch, min_num_epoch, max_no_improve,
                          critic_reg_alpha, eval_freq, lr, iter_i,
                          verbose=False):
        self.model.train()
        critic.train()
        critic_optim = LBFGS(critic.parameters(),
                             line_search_fn="strong_wolfe")
        closure = lambda : self.critic_lbfgs_closure(
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

