from abc import ABC, abstractmethod, abstractstaticmethod
import os
import json

import torch
import torch.nn as nn


STATE_FILE_NAME = "state.pt"
KWARGS_FILE_NAME = "init_kwargs.json"


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
    def get_eta(self, s, a):
        pass

    @abstractmethod
    def get_w(self, s):
        pass

    @abstractmethod
    def get_all(self, s, a, ss, pi_ss):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def set_state(self, state):
        pass

    @abstractmethod
    def get_init_kwargs(self):
        pass

    @classmethod
    def load_model(cls, load_path):
        with open(os.path.join(load_path, KWARGS_FILE_NAME)) as f:
            init_kwargs = json.load(f)
        model = cls(**init_kwargs)
        state = torch.load(os.path.join(load_path, STATE_FILE_NAME))
        model.set_state(state)
        return model

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        state = self.get_state()
        state_path = os.path.join(save_dir, STATE_FILE_NAME)
        torch.save(state, state_path)
        init_kwargs = self.get_init_kwargs()
        init_kwargs_path = os.path.join(save_dir, KWARGS_FILE_NAME)
        with open(init_kwargs_path, "w") as f:
            json.dump(init_kwargs, f, indent=2)

        return 

    def estimate_policy_val_q(self, s_init, a_init, gamma):
        q = self.get_q(s_init.unsqueeze(0), a_init)
        return (1 - gamma) * float(q[0])

    def estimate_policy_val_w(self, dl, pi_e_name, normalize=False,
                              batch_scale=1000.0):
        w_eta_sum = 0.0
        weighted_r_sum = 0
        batch_size_sum = 0
        for batch in dl:
            pi_s = batch[f"pi_s::{pi_e_name}"]
            pi_e_match = (pi_s == batch["a"]).reshape(-1, 1) * 1.0
            eta = self.get_eta(s=batch["s"], a=batch["a"]) * pi_e_match
            w = self.get_w(s=batch["s"]) 
            w_eta_sum += float((eta * w).sum()) / batch_scale
            r = batch["r"].unsqueeze(-1)
            new_sum = float((eta * w * r).sum())
            weighted_r_sum += new_sum / batch_scale
            batch_size_sum += len(batch["s"]) / batch_scale

        if normalize:
            mean_w_eta = w_eta_sum / batch_size_sum
            weighted_r_sum = weighted_r_sum / mean_w_eta

        return weighted_r_sum / batch_size_sum

    def estimate_policy_val_dr(self, dl, s_init, a_init, pi_e_name,
                               adversarial_lambda, gamma, normalize=False,
                               batch_scale=1000.0):
        lmbda = adversarial_lambda
        inv_lmbda = lmbda ** -1
        w_eta_sum = 0.0
        weighted_pseudo_r_sum = 0
        batch_size_sum = 0
        for batch in dl:
            # compute pseudo-reward for debiasing
            s = batch["s"]
            a = batch["a"]
            ss = batch["ss"]
            pi_ss = batch[f"pi_ss::{pi_e_name}"]
            q, v, xi = self.get_q_v_xi(s, a, ss, pi_ss)
            r = batch["r"].unsqueeze(-1)
            e_cvar_v = (inv_lmbda + (1 - inv_lmbda) * (1 + lmbda) * xi) * v
            pseudo_r = r + gamma * e_cvar_v - q

            # compute weights
            pi_s = batch[f"pi_s::{pi_e_name}"]
            pi_e_match = (pi_s == batch["a"]).reshape(-1, 1) * 1.0
            eta = self.get_eta(s=batch["s"], a=batch["a"]) * pi_e_match
            w = self.get_w(s=batch["s"])
            w_eta_sum += float((eta * w).sum()) / batch_scale

            # update weighted pseudo-reward sum
            new_sum = float((eta * w * pseudo_r).sum())
            weighted_pseudo_r_sum += new_sum / batch_scale
            batch_size_sum += len(batch["s"]) / batch_scale

        debias_correction = weighted_pseudo_r_sum / batch_size_sum

        if normalize:
            mean_w_eta = w_eta_sum / batch_size_sum
            debias_correction = debias_correction / mean_w_eta

        pv_q = self.estimate_policy_val_q(s_init, a_init, gamma)

        return pv_q + debias_correction
