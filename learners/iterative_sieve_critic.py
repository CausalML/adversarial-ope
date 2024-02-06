from copy import deepcopy
import sys

import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np

from learners.abstract_learner import AbstractLearner
from models.abstract_critic import AbstractCritic
from utils.oadam import OAdam


class SieveCritic(AbstractCritic):
    def __init__(self, prev_net_list, net_class, net_kwargs, init_basis_func,
                 num_init_basis, init_only=False):
        super().__init__(s_dim=net_kwargs["s_dim"], num_a=net_kwargs["num_a"])
        self.prev_net_list = prev_net_list
        self.net = net_class(**net_kwargs)
        self.init_basis_func = init_basis_func
        self.num_init_basis = num_init_basis
        assert isinstance(self.net, AbstractCritic)
        self.init_only = init_only
        num_func = self.get_num_basis_func()
        self.q_linear = nn.Linear(num_func, 1, bias=False)
        self.beta_linear = nn.Linear(num_func, 1, bias=False)
        self.eta_linear = nn.Linear(num_func, 1, bias=False)
        self.w_linear = nn.Linear(num_func, 1, bias=False)

    def disable_init_only(self):
        self.init_only = False

    def get_q(self, s, a):
        q_basis = self.get_q_basis_expansion(s, a)
        q = self.q_linear(q_basis)
        return q

    def get_eta(self, s, a):
        eta_basis = self.get_eta_basis_expansion(s, a)
        eta = self.eta_linear(eta_basis)
        return eta

    def get_w(self, s):
        w_basis = self.get_w_basis_expansion(s)
        w = self.w_linear(w_basis)
        return w 

    def get_all(self, s, a):
        q = self.get_q(s, a)
        eta = self.get_eta(s, a)
        w = self.get_w(s)
        return q, eta, w

    def get_next_func_batch_reg(self, batch, train_q_beta, train_eta, train_w):
        s = batch["s"]
        a = batch["a"]
        f_q, f_eta, f_w = self.net.get_all(s, a)
        reg_sum = 0
        if train_q_beta:
            reg_sum += (f_q ** 2).mean()
        if train_eta:
            reg_sum += (f_eta ** 2).mean()
        if train_w:
            reg_sum += (f_w ** 2).mean()
        return reg_sum

    def get_q_basis_expansion(self, s, a):
        # q_bias = torch.ones(len(s), 1).to(s.device)
        # beta_bias = torch.ones(len(s), 1).to(s.device)
        q_init = self.init_basis_func(s, a)
        if self.init_only:
            return q_init
        else:
            q_new = self.net.get_q(s, a)
            q_list = [q_init, q_new]
            for critic in self.prev_net_list:
                q = critic.get_q(s, a)
                q_list.append(q.detach())
            return torch.cat(q_list, dim=1)

    def get_eta_basis_expansion(self, s, a):
        # eta_bias = torch.ones(len(s), 1).to(s.device)
        eta_init = self.init_basis_func(s, a)
        if self.init_only:
            return eta_init
        else:
            eta_new = self.net.get_eta(s, a)
            eta_list = [eta_init, eta_new]
            for critic in self.prev_net_list:
                eta = critic.get_eta(s, a)
                eta_list.append(eta.detach())
            return torch.cat(eta_list, dim=1)

    def get_w_basis_expansion(self, s):
        # w_bias = torch.ones(len(s), 1).to(s.device)
        w_init = self.init_basis_func(s)
        if self.init_only:
            return w_init
        else:
            w_new = self.net.get_w(s)
            w_list = [w_init, w_new]
            for critic in self.prev_net_list:
                w = critic.get_w(s)
                w_list.append(w.detach())
            return torch.cat(w_list, dim=1)

    def get_num_basis_func(self):
        if self.init_only:
            return self.num_init_basis
        else:
            return len(self.prev_net_list) + 1 + self.num_init_basis

    def get_full_net_list(self):
        return self.prev_net_list + [self.net]


class IterativeSieveLearner(AbstractLearner):
    def __init__(self, nuisance_model, gamma, adversarial_lambda,
                 train_q_beta=True, train_eta=True, train_w=True,
                 use_dual_cvar=True, worst_case=True):
        super().__init__(
            nuisance_model=nuisance_model, gamma=gamma,
            adversarial_lambda=adversarial_lambda, worst_case=worst_case,
            train_q_beta=train_q_beta, train_eta=train_eta, train_w=train_w,
            use_dual_cvar=use_dual_cvar,
        )

    def train(self, dataset, pi_e_name, critic_class, critic_kwargs,
              s_init, init_basis_func, num_init_basis,
              evaluate_pv_kwargs=None, batch_size=1024, gamma_tik=1e-5,
              gamma_0=1e-2, total_num_iterations=20, val_frac=0.1, 
              model_max_epoch=50, model_min_epoch=2, 
              model_eval_freq=2, model_max_no_improve=3,
              model_lr=1e-4, beta_lr=1e-3, model_reg_alpha=1e-6,
              model_reg_alpha_final=1e-6, critic_reg_alpha=1e-5,
              model_max_epoch_final=500, model_min_epoch_final=50,
              model_eval_freq_final=2, model_max_no_improve_final=5,
              model_lr_final=1e-4, beta_lr_final=1e-3, num_beta_sub_epoch=5,
              critic_max_epoch=100, critic_min_epoch=4,
              critic_eval_freq=4, critic_max_no_improve=2,
              critic_lr=5e-3, verbose=False, device=None):

        train_data, val_data = dataset.get_train_dev_split(val_frac)
        dl = train_data.get_batch_loader(batch_size)
        dl_2 = train_data.get_batch_loader(batch_size)
        dl_val = val_data.get_batch_loader(batch_size)

        # get initial "bias only critic"
        critic = SieveCritic(
            prev_net_list=[], init_only=True, net_class=critic_class,
            net_kwargs=critic_kwargs, init_basis_func=init_basis_func,
            num_init_basis=num_init_basis,
        )
        if device is not None:
            critic.to(device)

        # now do iterative updates with weighted objective
        for iter_i in range(1, total_num_iterations+1):

            # do a training update on model
            self.update_model(
                critic=critic, dl=dl, dl_2=dl_2, dl_val=dl_val,
                s_init=s_init, pi_e_name=pi_e_name, gamma_0=gamma_0,
                gamma_tik=gamma_tik, min_num_epoch=model_min_epoch,
                max_num_epoch=model_max_epoch, beta_lr=beta_lr,
                num_beta_sub_epoch=num_beta_sub_epoch,
                max_no_improve=model_max_no_improve,
                eval_freq=model_eval_freq, 
                reg_alpha=model_reg_alpha, iter_i=iter_i, lr=model_lr,
                verbose=verbose, device=device,
            )
            if verbose and (evaluate_pv_kwargs is not None):
                self.print_policy_value_estimates(**evaluate_pv_kwargs)

            # prepare next critic for training
            if critic.init_only:
                prev_net_list = []
            else:
                prev_net_list = critic.get_full_net_list()
            critic = SieveCritic(
                prev_net_list=prev_net_list, net_class=critic_class,
                net_kwargs=critic_kwargs, init_basis_func=init_basis_func,
                num_init_basis=num_init_basis,
            )
            if device is not None:
                critic.to(device)

            # train next critic func
            self.train_next_critic(
                critic=critic, dl=dl, dl_val=dl_val, pi_e_name=pi_e_name,
                max_num_epoch=critic_max_epoch,
                s_init=s_init, min_num_epoch=critic_min_epoch,
                max_no_improve=critic_max_no_improve,
                eval_freq=critic_eval_freq, critic_reg_alpha=critic_reg_alpha,
                iter_i=iter_i, lr=critic_lr, verbose=verbose, 
            )

        # do a final round of training on model
        best_state = self.update_model(
            critic=critic, dl=dl, dl_2=dl_2, dl_val=dl_val,
            s_init=s_init, pi_e_name=pi_e_name, gamma_0=gamma_0,
            gamma_tik=gamma_tik, min_num_epoch=model_min_epoch_final,
            max_num_epoch=model_max_epoch_final,
            max_no_improve=model_max_no_improve_final,
            eval_freq=model_eval_freq_final,
            reg_alpha=model_reg_alpha_final, beta_lr=beta_lr_final,
            num_beta_sub_epoch=num_beta_sub_epoch,
            iter_i="FINAL", lr=model_lr_final, verbose=verbose, device=device,
        )
        self.model.set_state(best_state)

    def print_policy_value_estimates(self, s_init, a_init, pi_e_name, dl_test):
        xi_sum = 0
        batch_sum = 0
        for batch in dl_test:
            s = batch["s"]
            a = batch["a"]
            ss = batch["ss"]
            pi_ss = batch[f"pi_ss::{pi_e_name}"]
            v, beta = self.model.get_v_beta(s, a, ss, pi_ss)
            # print(v[:10])
            # print(beta[:10])
            mean, std = (beta - v).mean(), (beta - v).std()
            # print(mean, std)
            xi = (beta > v) * 1.0
            xi_sum += float(xi.sum()) / 1000.0
            # print(xi.mean())
            batch_sum += len(s) / 1000.0
            # print(f"mean beta: {beta.mean()}, diff: {mean}, std: {std}, p: {xi.mean()}")
        print(f"mean xi: {xi_sum / batch_sum}")

        q_pv = self.model.estimate_policy_val_q(
            s_init=s_init, a_init=a_init, gamma=self.gamma
        )
        w_pv = self.model.estimate_policy_val_w(
            dl=dl_test, pi_e_name=pi_e_name,
        )
        w_pv_norm = self.model.estimate_policy_val_w(
            dl=dl_test, pi_e_name=pi_e_name, normalize=True,
        )
        dr_pv = self.model.estimate_policy_val_dr(
            s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
            gamma=self.gamma, adversarial_lambda=self.adversarial_lambda,
            dual_cvar=False, worst_case=self.worst_case,
        )
        dr_pv_dual = self.model.estimate_policy_val_dr(
            s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
            adversarial_lambda=self.adversarial_lambda, gamma=self.gamma,
            dual_cvar=True, worst_case=self.worst_case,
        )
        dr_pv_norm = self.model.estimate_policy_val_dr(
            s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
            adversarial_lambda=self.adversarial_lambda, gamma=self.gamma,
            dual_cvar=False, normalize=True, worst_case=self.worst_case,
        )
        dr_pv_dual_norm = self.model.estimate_policy_val_dr(
            s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
            adversarial_lambda=self.adversarial_lambda, gamma=self.gamma,
            dual_cvar=True, normalize=True, worst_case=self.worst_case,
        )
        print(f"Intermediate policy value results:")
        print(f"Q-estimated v(pi_e): {q_pv}")
        print(f"W-estimated v(pi_e): {w_pv}")
        print(f"W-estimated v(pi_e) (normalized): {w_pv_norm}")
        print(f"DS/DV-estimated v(pi_e): {dr_pv}")
        print(f"DS/DV-estimated v(pi_e) (dual): {dr_pv_dual}")
        print(f"DS/DV-estimated v(pi_e) (normalized): {dr_pv_norm}")
        print(f"DS/DV-estimated v(pi_e) (normalized, dual): {dr_pv_dual_norm}")
        print("")

    def update_model(self, critic, dl, dl_2, dl_val, pi_e_name, max_num_epoch,
                     s_init, min_num_epoch, max_no_improve,
                     beta_lr, num_beta_sub_epoch, eval_freq, lr,
                     gamma_0, gamma_tik, iter_i, reg_alpha,
                     device, verbose=False, eig_threshold=1e-5):

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
        model_optim = Adam(self.model.get_parameters(), lr=lr)
        beta_optim = Adam(self.model.get_beta_parameters(), lr=beta_lr)
        if verbose:
            init_loss, init_moment_losses = self.get_mean_model_loss(
                critic=critic, s_init=s_init,
                dl=dl_val, pi_e_name=pi_e_name, omega=omega,
            )
            init_beta_loss = self.get_mean_beta_loss(
                dl=dl_val, pi_e_name=pi_e_name
            )
            print(f"MODEL: iter {iter_i}")
            print(f"Starting val loss: {init_loss},"
                    f" beta loss: {init_beta_loss}"
                    f" per moment: {init_moment_losses}")
            print("")
        best_val = float("inf")
        best_state = deepcopy(self.model.get_state())
        num_no_improve = 0

        for epoch_i in range(1, max_num_epoch+1):
            self.train_model_one_epoch(
                model_optim=model_optim, critic=critic, dl=dl, dl_2=dl_2,
                omega=omega, s_init=s_init, pi_e_name=pi_e_name,
                alpha_reg=reg_alpha,
            )
            for _ in range(num_beta_sub_epoch):
                self.train_beta_one_epoch(
                    beta_optim=beta_optim, dl=dl, pi_e_name=pi_e_name,
                    alpha_reg=reg_alpha,
                )

            if epoch_i % eval_freq == 0:
                val_loss, val_moment_losses = self.get_mean_model_loss(
                    critic=critic, s_init=s_init,
                    dl=dl_val, pi_e_name=pi_e_name, omega=omega,
                )
                if verbose:
                    train_loss, train_moment_losses = self.get_mean_model_loss(
                        critic=critic, s_init=s_init, dl=dl,
                        pi_e_name=pi_e_name, omega=omega,
                    )
                    train_beta_loss = self.get_mean_beta_loss(
                        dl=dl, pi_e_name=pi_e_name
                    ) 
                    val_beta_loss = self.get_mean_beta_loss(
                        dl=dl_val, pi_e_name=pi_e_name
                    ) 

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = deepcopy(self.model.get_state())
                    num_no_improve = 0
                else:
                    num_no_improve += 1

                if verbose:
                    print(f"MODEL: iter {iter_i}, epoch {epoch_i}")
                    print(f"mean train loss: {train_loss},"
                          f" beta loss: {train_beta_loss}"
                          f" per moment: {train_moment_losses}")
                    print(f"mean val loss: {val_loss},"
                          f" beta loss: {val_beta_loss}"
                          f" per moment: {val_moment_losses}")
                    if num_no_improve == 0:
                        print("NEW BEST")
                    print("")

                if (num_no_improve >= max_no_improve
                        and epoch_i >= min_num_epoch):
                    break

        # print("LOADING BEST PARAMS")
        # self.model.set_state(best_state)
        # print("")
        self.model.eval()
        return best_state

    def train_model_one_epoch(self, model_optim, critic, dl, dl_2, omega,
                              s_init, pi_e_name, alpha_reg):
        for batch in dl:
            dl_2_iter = iter(dl_2)
            for batch in dl:
                batch_2 = next(dl_2_iter)
                moments_1 = self.get_batch_moments(
                    batch=batch, critic=critic, s_init=s_init,
                    pi_e_name=pi_e_name, model_grad=True, basis_expansion=True
                )
                moments_2 = self.get_batch_moments( 
                    batch=batch_2, critic=critic, s_init=s_init,
                    pi_e_name=pi_e_name, model_grad=True, basis_expansion=True,
                )
                rho_f_1 = moments_1.mean(0).view(-1)
                rho_f_2 = moments_2.mean(0).view(-1)
                model_optim.zero_grad()
                m_loss = torch.einsum("xy,x,y->", omega, rho_f_1, rho_f_2)
                if alpha_reg:
                    m_reg = alpha_reg * self.get_batch_l2_reg_model(
                        batch=batch, pi_e_name=pi_e_name
                    )
                else:
                    m_reg = 0

                (m_loss + m_reg).backward()
                model_optim.step()

    def get_mean_model_loss(self, critic, dl, pi_e_name, omega, s_init,
                            batch_scale=1000.0):
        self.model.eval()
        rho_f_sum = 0
        batch_size_sum = 0
        num_m = self.get_num_moments()
        moment_rho_f_sums = [0 for _ in range(num_m)]

        for batch in dl:
            moments = self.get_batch_moments(
                batch=batch, critic=critic, s_init=s_init,
                pi_e_name=pi_e_name, basis_expansion=True,
            )
            rho_f = moments.sum(0)
            rho_f_sum = rho_f_sum + rho_f.view(-1) / batch_scale
            for i in range(num_m):
                sum_i = rho_f[:,i] / batch_scale
                moment_rho_f_sums[i] = moment_rho_f_sums[i] + sum_i
            batch_size_sum += len(batch["s"]) / batch_scale

        rho_f_mean = rho_f_sum / batch_size_sum
        m_loss = torch.einsum("xy,x,y->", omega, rho_f_mean, rho_f_mean)
        moment_losses = []
        num_k = len(rho_f_sum) // num_m
        for i in range(num_m):
            omega_i = omega.reshape(num_k, num_m, num_k, num_m)[:,i,:,i]
            rho_f_mean_i = moment_rho_f_sums[i] / batch_size_sum
            loss_i = torch.einsum("xy,x,y->", omega_i, rho_f_mean_i,
                                  rho_f_mean_i).unsqueeze(0)
            moment_losses.append(loss_i)
        self.model.train()
        return m_loss, torch.cat(moment_losses)

    def train_beta_one_epoch(self, beta_optim, dl, pi_e_name, alpha_reg):
        for batch in dl:
            beta_optim.zero_grad()
            beta_loss = self.get_batch_quantile_loss(
                batch=batch, pi_e_name=pi_e_name,
            )
            if alpha_reg:
                beta_reg = alpha_reg * self.get_batch_l2_reg_beta(batch)
            else:
                beta_reg = 0
            (beta_loss + beta_reg).backward()
            beta_optim.step()

    def get_mean_beta_loss(self, dl, pi_e_name, batch_scale=1000.0):
        self.model.eval()
        beta_loss_sum = 0
        batch_size_sum = 0

        for batch in dl:
            beta_loss = self.get_batch_quantile_loss(
                batch=batch, pi_e_name=pi_e_name,
            )
            batch_n = len(batch["s"])
            beta_loss_sum += float(beta_loss) * batch_n / batch_scale
            batch_size_sum += batch_n / batch_scale

        beta_loss = beta_loss_sum / batch_size_sum
        self.model.train()
        return beta_loss

    def get_omega_inv(self, critic, dl, pi_e_name, s_init, gamma_tik,
                      batch_scale=1000.0, device=None):
        self.model.eval()
        f_mat = 0
        batch_size_sum = 0
        gamma_mat = torch.eye(self.get_num_moments()) * gamma_tik
        if device is not None:
            gamma_mat = gamma_mat.to(device)
        for batch in dl:
            f_basis = self.get_critic_basis_expansion(
                batch=batch, critic=critic
            )
            moments_expended = self.get_batch_moments(
                batch=batch, critic=critic, s_init=s_init,
                pi_e_name=pi_e_name, basis_expansion=True,
            )
            mat_1 = torch.einsum("bkn,blm->knlm", moments_expended,
                                 moments_expended)
            mat_2 = torch.einsum("bkn,blm,nm->knlm",
                                 f_basis, f_basis, gamma_mat)
            f_mat = f_mat + (mat_1 + mat_2) / batch_scale
            batch_size_sum += len(batch["s"]) / batch_scale

        self.model.train()
        return f_mat / batch_size_sum

    def train_next_critic(self, critic, dl, dl_val, pi_e_name, s_init,
                          max_num_epoch, min_num_epoch, max_no_improve,
                          critic_reg_alpha, eval_freq, lr, iter_i,
                          verbose=False):
        self.model.train()
        critic.train()
        critic_optim = Adam(critic.parameters(), lr=lr)

        best_val = float("inf")
        num_no_improve = 0

        for epoch_i in range(1, max_num_epoch+1):
            for batch in dl:
                moments = self.get_batch_moments(
                    batch=batch, critic=critic, s_init=s_init,
                    pi_e_name=pi_e_name, critic_grad=True,
                )
                obj = moments.mean() - 0.5 * (moments ** 2).mean()
                if critic_reg_alpha:
                    reg_1 = self.get_batch_l2_reg_critic(
                        batch=batch, critic=critic,
                    )
                    reg_2 = critic.get_next_func_batch_reg(
                        batch=batch, train_q_beta=self.train_q_beta,
                        train_eta=self.train_eta, train_w=self.train_w,
                    )
                    reg = critic_reg_alpha * (reg_1 + reg_2)
                else:
                    reg = 0
                loss = (-1.0 * obj + reg)

                critic_optim.zero_grad()
                loss.backward()
                critic_optim.step()

            if epoch_i % eval_freq == 0:
                val_loss = self.get_mean_critic_loss(
                    critic=critic, dl=dl_val,
                    pi_e_name=pi_e_name, s_init=s_init,
                )

                if val_loss < best_val:
                    best_val = val_loss
                    num_no_improve = 0
                else:
                    num_no_improve += 1

                if verbose:
                    train_loss = self.get_mean_critic_loss(
                        critic=critic, dl=dl,
                        pi_e_name=pi_e_name, s_init=s_init,
                    )
                    print(f"CRITIC: iter {iter_i}, epoch {epoch_i}")
                    print(f"mean train loss: {train_loss}")
                    print(f"mean val loss: {val_loss}")
                    if num_no_improve == 0:
                        print("NEW BEST")
                    print("")

                if (num_no_improve >= max_no_improve
                        and epoch_i >= min_num_epoch):
                    break

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
            obj = moments.mean() - 0.5 * (moments ** 2).mean()
            loss = (-1.0 * obj)

            loss_sum += float(loss) / batch_scale
            batch_size_sum += len(batch["s"]) / batch_scale

        self.model.train()
        critic.train()

        return loss_sum / batch_size_sum
