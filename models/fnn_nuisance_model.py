import torch
import torch.nn as nn

from models.abstract_nuisance_model import AbstractNuisanceModel
from utils.neural_nets import FFNet

class FFNuisanceModule(nn.Module):
    def __init__(self, s_dim, num_a, gamma, config):
        self.s_dim = s_dim
        self.num_a = num_a
        self.scale = 1 / (1 - gamma)
        super().__init__()
        s_embed_size = config["s_embed_dim"]
        a_embed_size = config["a_embed_dim"]
        self.s_embed_net = FFNet(
            input_dim=s_dim,
            output_dim=s_embed_size,
            layer_sizes=config["s_embed_layers"],
        )
        self.a_embed_net = nn.Embedding(
            num_embeddings=num_a,
            embedding_dim=a_embed_size,
        )
        sa_embed_size = s_embed_size + a_embed_size
        self.sa_feature_net = FFNet(
            input_dim=sa_embed_size,
            output_dim=config["sa_feature_dim"],
            layer_sizes=config["sa_feature_layers"],
        )
        self.q_net = FFNet(
            input_dim=config["sa_feature_dim"],
            output_dim=1,
            layer_sizes=config["q_layers"],
        )
        self.beta_net = FFNet(
            input_dim=config["sa_feature_dim"],
            output_dim=1,
            layer_sizes=config["beta_layers"],
        )
        self.w_net = FFNet(
            input_dim=config["sa_feature_dim"],
            output_dim=1,
            layer_sizes=config["w_layers"],
        )
        self.xi_temp = nn.Parameter(torch.FloatTensor([0.1]))
        self.max_temp = (torch.ones_like(self.xi_temp)
                         * config.get("max_temp", 10.0))
        self.zero = torch.zeros_like(self.xi_temp)
        self.temp_clip_alpha = config.get("max_temp_clip_alpha", 1e-3)
        # self.temp_relu = nn.LeakyReLU(negative_slope=temp_clip_alpha)

    def forward(self, s, a, ss=None, pi_ss=None,
                calc_q=True, calc_v=True, calc_xi=True, calc_w=True):
        assert calc_q or calc_w or calc_w

        s_embed = self.s_embed_net(s)
        a_embed = self.a_embed_net(a)
        sa_concat = torch.cat([s_embed, a_embed], dim=1)
        sa_features = self.sa_feature_net(sa_concat)

        if calc_q or calc_xi:
            q = self.scale * torch.sigmoid(self.q_net(sa_features) / 10.0) 
        else:
            q = None

        if calc_v or calc_xi:
            assert ss is not None
            assert pi_ss is not None
            ss_embed = self.s_embed_net(ss)
            pi_ss_embed = self.a_embed_net(pi_ss)
            ss_pi_ss_concat = torch.cat([ss_embed, pi_ss_embed], dim=1)
            ss_pi_ss_features = self.sa_feature_net(ss_pi_ss_concat)
            v = self.scale * torch.sigmoid(self.q_net(ss_pi_ss_features) / 10.0)
        else:
            v = None

        if calc_xi:
            beta = self.scale * torch.sigmoid(self.beta_net(sa_features) / 10.0)
            # temp = self._soft_temp_clip(self.xi_temp)
            temp = 1.0
            xi = torch.sigmoid(temp * (beta - v))
        else:
            xi = None

        if calc_w:
            w = self.w_net(sa_features)
        else:
            w = None

        return q, v, xi, w

    def _soft_temp_clip(self, temp):
        abs_temp = temp.abs()
        alpha = self.temp_clip_alpha
        soft_excess = alpha * torch.max(abs_temp - self.max_temp, self.zero)
        hard_clip = torch.min(abs_temp, self.max_temp)
        return hard_clip + soft_excess


class FeedForwardNuisanceModel(AbstractNuisanceModel):
    def __init__(self, s_dim, num_a, gamma, config):
        super().__init__(s_dim, num_a)
        self.net = FFNuisanceModule(s_dim=s_dim, num_a=num_a, config=config,
                                    gamma=gamma)

    def get_q(self, s, a):
        q, _, _, _ = self.net(
            s, a, calc_v=False, calc_xi=False, calc_w=False)
        return q

    def get_q_v_xi(self, s, a, ss, pi_ss):
        q, v, xi, _ = self.net(
            s, a, ss, pi_ss, calc_w=False)
        return q, v, xi

    def get_w(self, s, a):
        _, _, _, w = self.net(
            s, a, calc_q=False, calc_v=False, calc_xi=False)
        return w

    def get_all(self, s, a, ss, pi_ss):
        return self.net(s, a, ss, pi_ss)

    def get_parameters(self):
        return self.net.parameters()