import torch
import torch.nn as nn

from models.abstract_critic import AbstractCritic
from utils.neural_nets import FFNet


class FeedForwardCritic(AbstractCritic):
    def __init__(self, num_out, s_dim, num_a, config):
        super().__init__(num_out, s_dim, num_a)
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
        self.critic_net = FFNet(
            input_dim=sa_embed_size,
            output_dim=self.num_out,
            layer_sizes=config["critic_layers"],
        )

    def forward(self, s, a):
        s_embed = self.s_embed_net(s)
        a_embed = self.a_embed_net(a)
        sa_concat = torch.cat([s_embed, a_embed], dim=1)
        return self.critic_net(sa_concat)
