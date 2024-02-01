from abc import ABC, abstractmethod

import torch

from models.abstract_nuisance_model import AbstractNuisanceModel

class AbstractLearner(ABC):
    def __init__(self, nuisance_model, gamma, adversarial_lambda,
                 train_q_xi=False, train_eta=True, train_w=False,
                 worst_case=True):
        self.model = nuisance_model
        self.gamma = gamma
        self.adversarial_lambda = adversarial_lambda
        self.worst_case = worst_case
        self.train_q_xi = train_q_xi
        self.train_eta = train_eta
        self.train_w = train_w
        assert isinstance(self.model, AbstractNuisanceModel)
        super().__init__()

    def get_num_moments(self):
        return 2 * self.train_q_xi + 1 * self.train_eta + 1 * self.train_w

    def get_critic_basis_expansion(self, batch, critic):
        basis_list = []
        if self.train_q_xi:
            q_basis, xi_basis = critic.get_q_xi_basis_expansion(
                s=batch["s"], a=batch["a"]
            )
            basis_list.append(q_basis)
            basis_list.append(xi_basis)
        if self.train_eta:
            eta_basis = critic.get_eta_basis_expansion(
                s=batch["s"], a=batch["a"]
            )
            basis_list.append(eta_basis)
        if self.train_w:
            w_basis = critic.get_w_basis_expansion(
                s=batch["s"]
            )
            basis_list.append(w_basis)
        f_basis = torch.stack(basis_list, dim=2).detach()
        return f_basis

    def get_batch_moments(self, batch, critic, pi_e_name, s_init,
                          model_grad=False, critic_grad=False,
                          basis_expansion=False):
        s = batch["s"]
        a = batch["a"]
        ss = batch["ss"]
        r = batch["r"]
        pi_s = batch[f"pi_s::{pi_e_name}"]
        pi_ss = batch[f"pi_ss::{pi_e_name}"]

        moments_list = []
        if self.train_q_xi:
            q_m, xi_m = self.get_q_xi_batch_moments(
                critic=critic, s=s, a=a, ss=ss, r=r, pi_ss=pi_ss,
                model_grad=model_grad, critic_grad=critic_grad,
                basis_expansion=basis_expansion
            )
            moments_list.append(q_m)
            moments_list.append(xi_m)

        if self.train_eta:
            eta_m = self.get_eta_batch_moments(
                critic=critic, s=s, a=a, pi_s=pi_s,
                model_grad=model_grad, critic_grad=critic_grad,
                basis_expansion=basis_expansion
            )
            moments_list.append(eta_m)

        if self.train_w:
            w_m = self.get_w_batch_moments(
                critic=critic, s=s, a=a, ss=ss, pi_s=pi_s, pi_ss=pi_ss,
                s_init=s_init, model_grad=model_grad, critic_grad=critic_grad,
                basis_expansion=basis_expansion
            )
            moments_list.append(w_m)


        if basis_expansion:
            return torch.stack(moments_list, dim=2)
        else:
            return torch.cat(moments_list, dim=1).sum(1)

    def get_batch_l2_reg(self, batch, pi_e_name):
        s = batch["s"]
        a = batch["a"]
        ss = batch["ss"]
        pi_ss = batch[f"pi_ss::{pi_e_name}"]
        r = batch["r"]
        q, v, _, beta, eta, w  = self.model.get_all(s, a, ss, pi_ss)
        reg_sum = 0
        if self.train_q_xi:
            reg_sum += (q ** 2).mean()
            reg_sum += (v ** 2).mean()
            reg_sum += (beta ** 2).mean()
        if self.train_eta:
            reg_sum += (eta ** 2).mean()
        if self.train_w:
            reg_sum += (w ** 2).mean()
        return reg_sum

    def get_q_xi_batch_moments(self, critic, s, a, ss, r, pi_ss,
                               model_grad=False, critic_grad=False,
                               basis_expansion=False, use_dual_cvar=True):

        q, v, xi, beta = self.model.get_q_v_xi_beta(s, a, ss, pi_ss)
        lmbda = self.adversarial_lambda
        inv_lmbda = lmbda ** -1
        alpha = 1 / (1 + lmbda)
        assert inv_lmbda > 0
        assert inv_lmbda <= 1
        if use_dual_cvar:
            if self.worst_case:
                cvar_v = beta - (1 + lmbda) * xi * (beta - v)
            else:
                cvar_v = beta + (1 + lmbda) * xi * (v - beta)
        else:
            cvar_v = (1 + lmbda) * xi * v
        e_cvar_v = inv_lmbda * v + (1 - inv_lmbda) * cvar_v
        rho_q = q  - r.unsqueeze(-1) - self.gamma * e_cvar_v
        rho_q = rho_q * (1 - self.gamma)
        if self.worst_case:
            # estimating worst-case Q function within sensitivity model
            rho_xi = xi - alpha
        else:
            # esimating best-case Q function within sensitivity model
            rho_xi = xi - (1 - alpha)
        if not model_grad:
            rho_q, rho_xi = rho_q.detach(), rho_xi.detach()

        if basis_expansion:
            f_q, f_xi = critic.get_q_xi_basis_expansion(s, a)
        else:
            f_q, f_xi = critic.get_q_xi(s, a)

        if not critic_grad:
            f_q, f_xi = f_q.detach(), f_xi.detach()

        return rho_q * f_q, rho_xi * f_xi

    def get_eta_batch_moments(self, critic, s, a, pi_s,
                              model_grad=False, critic_grad=False,
                              basis_expansion=False):
        pi_e_match = (pi_s == a).reshape(-1, 1) * 1.0
        eta = self.model.get_eta(s, a) * pi_e_match
        if not model_grad:
            eta = eta.detach()
        if basis_expansion:
            f_eta_s_a = critic.get_eta_basis_expansion(s, a)
            f_eta_s_pi = critic.get_eta_basis_expansion(s, pi_s)
        else:
            f_eta_s_a = critic.get_eta(s, a)
            f_eta_s_pi = critic.get_eta(s, pi_s)
        if not critic_grad:
            f_eta_s_a = f_eta_s_a.detach()
            f_eta_s_pi = f_eta_s_pi.detach()

        return f_eta_s_a * eta - f_eta_s_pi

    def get_w_batch_moments(self, critic, s, a, ss, pi_s, pi_ss, s_init,
                            model_grad=False, critic_grad=False,
                            basis_expansion=False):
        w = self.model.get_w(s)
        w_ss = self.model.get_w(ss)
        pi_e_match = (pi_s == a).reshape(-1, 1) * 1.0
        eta = self.model.get_eta(s, a) * pi_e_match
        xi = self.model.get_xi(s, a, ss, pi_ss)
        if not model_grad:
            w = w.detach()
            w_ss = w_ss.detach()
            eta = eta.detach()
            xi = xi.detach()
        elif not self.train_q_xi:
            xi = xi.detach()
        elif not self.train_eta:
            eta = eta.detach()

        if basis_expansion:
            f_s = critic.get_w_basis_expansion(s)
            f_ss = critic.get_w_basis_expansion(ss)
            f_s0 = critic.get_w_basis_expansion(s_init.unsqueeze(0))
        else:
            f_s = critic.get_w(s)
            f_ss = critic.get_w(ss)
            f_s0 = critic.get_w(s_init.unsqueeze(0))
        if not critic_grad:
            f_s = f_s.detach()
            f_ss = f_ss.detach()
            f_s0 = f_s0.detach()

        lmbda = self.adversarial_lambda
        inv_lmbda = lmbda ** -1
        lambda_is = inv_lmbda + (1 - inv_lmbda) * (1 + lmbda) * xi
        # return (self.gamma * w * (eta * lambda_is).detach() * f_ss
        #         - lambda_is.detach() * f_ss * w_ss + (1 - self.gamma) * f_s0)
        return (self.gamma * w * (eta * lambda_is) * f_ss
                - f_s * w + (1 - self.gamma) * f_s0)

    @abstractmethod
    def train(self, dataset, pi_e_name):
        pass