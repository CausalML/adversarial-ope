import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.abstract_nuisance_model import AbstractNuisanceModel
from utils.neural_nets import FFNet


class FFLargeABetaModule(nn.Module):
    def __init__(self, s_dim, num_a, gamma, config):
        super().__init__()
        self.s_dim = s_dim
        self.num_a = num_a
        self.sa_input_dim = config["s_embed_dim"] + config["a_embed_dim"]
        self.q_scale = nn.Parameter(torch.ones(1) / (1 - gamma))

        self.a_embed_net = nn.Embedding(
            num_embeddings=num_a,
            embedding_dim=config['a_embed_dim'],
        )
    
        self.s_embed_net = FFNet(
            input_dim=self.s_dim,
            output_dim=config["s_embed_dim"],
            layer_sizes=config["s_embed_layers"],
            dropout_rate=config.get("s_embed_do", 0.05),
        )

        self.beta_net = FFNet(
            input_dim=self.sa_input_dim,
            output_dim=1,
            layer_sizes=config["beta_layers"],
            dropout_rate=config.get("beta_do", 0.05)
        )
        # self.pos_head = nn.Softplus(beta=0.2)
        self.pos_head = lambda x_: x_.abs() 

    def update_q_scale(self, new_scalar):
        self.q_scale.data = self.q_scale.data * new_scalar

    def forward(self, s, a):
        s_embed = self.s_embed_net(s)
        a_embed = self.a_embed_net(a)
        sa_input = torch.concat([s_embed, a_embed], dim=1)
        beta = self.pos_head(self.beta_net(sa_input))
        return beta * self.q_scale


class FFLargeANuisanceModule(nn.Module):
    def __init__(self, s_dim, num_a, gamma, config):
        super().__init__()
        self.s_dim = s_dim
        self.sa_input_dim = config["s_embed_dim"] + config["a_embed_dim"]
        self.q_scale = nn.Parameter(torch.ones(1) / (1 - gamma))
        self.num_a = num_a

        self.emebeds_frozen = False

        self.a_embed_net = nn.Embedding(
            num_embeddings=num_a,
            embedding_dim=config['a_embed_dim'],
        )
    
        self.s_embed_net = FFNet(
            input_dim=self.s_dim,
            output_dim=config["s_embed_dim"],
            layer_sizes=config["s_embed_layers"],
            dropout_rate=config.get("s_embed_do", 0.05),
        )

        self.q_net = FFNet(
            input_dim=self.sa_input_dim,
            output_dim=1,
            layer_sizes=config["q_layers"],
            dropout_rate=config.get("q_do", 0.05)
        )
        self.eta_net = FFNet(
            input_dim=self.sa_input_dim,
            output_dim=1,
            layer_sizes=config["eta_layers"],
            dropout_rate=config.get("eta_do", 0.05)
        )
        self.w_net = FFNet(
            input_dim=config["s_embed_dim"],
            output_dim=1,
            layer_sizes=config["w_layers"],
            dropout_rate=config.get("w_do", 0.05)
        )
        # self.pos_head = nn.Softplus(beta=0.2)
        self.pos_head = lambda x_: x_.abs()

    def freeze_embeds(self):
        self.emebeds_frozen = True

    def update_q_scale(self, new_scalar):
        self.q_scale.data = self.q_scale.data * new_scalar

    def get_sa_input(self, s, a):
        s_embed = self.s_embed_net(s)
        a_embed = self.a_embed_net(a)
        embeds = torch.concat([s_embed, a_embed], dim=1)
        if self.emebeds_frozen:
            embeds = embeds.detach()
        return embeds

    def forward(self, s, a=None, ss=None, pi_ss=None, calc_q=False,
                calc_v=False, calc_eta=False, calc_w=False):

        if calc_q or calc_eta:
            sa_input = self.get_sa_input(s, a)
        else:
            sa_input = None

        if calc_q:
            assert a is not None
            q = self.pos_head(self.q_net(sa_input)) * self.q_scale
        else:
            q = None

        if calc_v:
            ss_pi_input = self.get_sa_input(ss, pi_ss)
            v = self.pos_head(self.q_net(ss_pi_input)) * self.q_scale
        else:
            v = None

        if calc_eta:
            eta = self.pos_head(self.eta_net(sa_input))
        else:
            eta = None

        if calc_w:
            s_embed = self.s_embed_net(s)
            w = self.pos_head(self.w_net(s_embed))
        else:
            w = None

        return q, v, eta, w


class LargeAFeedForwardNuisanceModel(AbstractNuisanceModel):
    def __init__(self, s_dim, num_a, gamma, config, device=None):
        super().__init__(s_dim, num_a)
        self.gamma = gamma
        self.config = config
        self.device = device
        self.reset_networks()

    def reset_networks(self):
        self.net = FFLargeANuisanceModule(
            s_dim=self.s_dim, num_a=self.num_a,
            gamma=self.gamma, config=self.config,
        )
        self.beta_net = FFLargeABetaModule(
            s_dim=self.s_dim, num_a=self.num_a, gamma=self.gamma,
            config=self.config,
        )   
        if self.device is not None:
            self.net.to(self.device)
            self.beta_net.to(self.device)
        self.net.eval()
        self.beta_net.eval()

    def to(self, device):
        if device is not None:
            self.net.to(device)
            self.beta_net.to(device)
            self.device = device

    def get_q(self, s, a):
        q, _, _, _ = self.net(s, a, calc_q=True)
        return q

    def get_q_v_beta(self, s, a, ss, pi_ss):
        q, v, _, _ = self.net(
            s, a, ss, pi_ss,
            calc_q=True, calc_v=True, 
        )
        beta = self.beta_net(s, a)
        return q, v, beta

    def get_v_beta(self, s, a, ss, pi_ss):
        _, v, _, _ = self.net(
            s, a, ss, pi_ss,
            calc_v=True, 
        )
        beta = self.beta_net(s, a)
        return v, beta

    def get_eta(self, s, a):
        _, _, eta, _ = self.net(s, a, calc_eta=True)
        return eta

    def get_w(self, s):
        _, _, _, w = self.net(s, calc_w=True)
        return w

    def get_beta(self, s, a):
        return self.beta_net(s, a)

    def get_all(self, s, a, ss, pi_ss):
        beta = self.beta_net(s, a)
        q, v, eta, w = self.net(
            s, a, ss, pi_ss, calc_q=True, calc_v=True, 
            calc_eta=True, calc_w=True,
        )
        return q, v, beta, eta, w

    def get_init_kwargs(self):
        return {
            "s_dim": self.s_dim, "num_a": self.num_a,
            "gamma": self.gamma, "config": self.config,
        }

    def get_parameters(self):
        return self.net.parameters()

    def get_beta_parameters(self):
        return self.beta_net.parameters()

    def get_model_state(self):
        return self.net.state_dict()

    def get_beta_state(self):
        return self.beta_net.state_dict()

    def set_model_state(self, state_dict):
        self.net.load_state_dict(state_dict)

    def set_beta_state(self, state_dict):
        self.beta_net.load_state_dict(state_dict)

    def freeze_embeds(self):
        self.net.freeze_embeds()

    def add_q_multiplier_correction(
            self, dl, lmbda, pi_e_name, use_dual_cvar=True,
            worst_case=True, batch_scale=1000.0, verbose=False):

        mean_e_cvar_sum = 0
        mean_q_sum = 0
        mean_r_sum = 0

        batch_sum = 0

        for batch in dl:
            s = batch["s"]
            a = batch["a"]
            ss = batch["ss"]
            pi_ss = batch[f"pi_ss::{pi_e_name}"]

            q, v, beta = self.get_q_v_beta(s, a, ss, pi_ss)
            r = batch["r"]

            inv_lmbda = lmbda ** -1
            if use_dual_cvar:
                if worst_case:
                    cvar_v = beta - (1 + lmbda) * F.relu(beta - v)
                else:
                    cvar_v = beta + (1 + lmbda) * F.relu(v - beta)
            else:
                if worst_case:
                    cvar_v = (1 + lmbda) * (beta > v) * v
                else:
                    cvar_v = (1 + lmbda) * (v > beta) * v
            e_cvar_v = inv_lmbda * v + (1 - inv_lmbda) * cvar_v

            batch_weight = len(s) / batch_scale

            mean_e_cvar_sum += batch_weight * float(e_cvar_v.mean().detach())
            mean_q_sum += batch_weight * float(q.mean().detach())
            mean_r_sum += batch_weight * float(r.mean())
            batch_sum += batch_weight

        mean_e_cvar = mean_e_cvar_sum / batch_sum
        mean_q = mean_q_sum / batch_sum
        mean_r = mean_r_sum / batch_sum

        correction_mul = mean_r / (mean_q - self.gamma * mean_e_cvar)
        if correction_mul > 0:
            self.net.update_q_scale(correction_mul)
            self.beta_net.update_q_scale(correction_mul)
            if verbose:
                q_scalar_new = float(self.net.q_scale.data[0])
                print(f'new Q scalar: {q_scalar_new}')
        else:
            if verbose:
                print('WARNING: invalid Q correction multiplier, not doing correction')

    def train(self):
        self.net.train()
        self.beta_net.train()

    def eval(self):
        self.net.eval()
        self.beta_net.eval()