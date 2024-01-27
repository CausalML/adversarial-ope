import gymnasium as gym
import numpy as np
import torch

from environments.toy_env import ToyEnv
from utils.policy_evaluation import evaluate_policy
from policies.generic_policies import UniformPolicy, EpsilonSmoothPolicy
from policies.toy_env_policies import ThresholdPolicy
from utils.offline_dataset import OfflineRLDataset
from models.fnn_nuisance_model import FeedForwardNuisanceModel
from models.fnn_critic import FeedForwardCritic
from learners.min_max_learner import MinMaxLearner

def main():
    init_s = 2.0
    env = ToyEnv(init_s=init_s, adversarial=False, adversarial_lambda=2)
    gamma = 0.9
    pi_base = ThresholdPolicy(env, s_threshold=1.5)
    pi_b = EpsilonSmoothPolicy(env, pi_base=pi_base, epsilon=0.1)

    dataset = OfflineRLDataset(env=env, pi=pi_b, burn_in=1000, num_sample=10000, thin=10)
    pi_e = ThresholdPolicy(env, s_threshold=init_s)
    pi_e_name = "pi_e"
    dataset.apply_eval_policy(pi_e_name, pi_e)
    # dl = dataset.get_batch_loader(batch_size=10)
    # for batch in dl:
    #     for k, v in batch.items():
    #         print(k, v.shape)
    #         print(v)
    #     print("")

    adversarial_lambda = 3
    env_eval = ToyEnv(init_s=init_s, adversarial=True,
                      adversarial_lambda=adversarial_lambda)

    model_config = {
        "s_embed_dim": 16,
        "s_embed_layers": [16],
        "a_embed_dim": 16,
        "sa_feature_dim": 32,
        "sa_feature_layers": [32],
        "q_layers": [32, 16],
        "beta_layers": [32, 16],
        "w_layers": [32, 16],

    }
    s_dim = env.get_s_dim()
    num_a = env.get_num_a()
    model = FeedForwardNuisanceModel(s_dim=s_dim, num_a=num_a,
                                     config=model_config)
    critic_class = FeedForwardCritic
    critic_config = {
        "s_embed_dim": 8,
        "a_embed_dim": 8,
        "s_embed_layers": [8],
        "critic_layers": [16],
    }
    critic_kwargs = {
        "s_dim": s_dim,
        "num_a": num_a,
        "config": critic_config
    }
    learner = MinMaxLearner(
        nuisance_model=model,
        gamma=gamma, 
        adversarial_lambda=adversarial_lambda
    )
    learner.train(
        dataset, pi_e_name=pi_e_name,
        critic_class=critic_class,
        critic_kwargs=critic_kwargs
    )
    init_s, init_a = env_eval.get_init_s_a(pi_e)
    pi_e_val_est = learner.estimate_policy_val_q(init_s, init_a)
    print(f"estimated v(pi_e): {pi_e_val_est}")
    print("")

    pi_e_val = evaluate_policy(env_eval, pi_e, gamma, min_prec=1e-4)
    print(f"true v(pi_e): {pi_e_val}")
    print("")


if __name__ == "__main__":
    main()