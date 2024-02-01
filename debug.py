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
from learners.iterative_sieve_critic import IterativeSieveLearner

def main():
    s_threshold = 2.0
    adversarial_lambda = 4.0
    env = ToyEnv(s_init=s_threshold, adversarial=False)
    gamma = 0.9
    pi_base = ThresholdPolicy(env, s_threshold=1.5)
    pi_b = EpsilonSmoothPolicy(env, pi_base=pi_base, epsilon=0.1)
    pi_e = ThresholdPolicy(env, s_threshold=s_threshold)
    pi_e_name = "pi_e"

    dataset = OfflineRLDataset()
    num_sample = 10000
    dataset.sample_new_trajectory(env=env, pi=pi_b, burn_in=1000,
                                  num_sample=num_sample, thin=10)
    dataset.apply_eval_policy(pi_e_name, pi_e)
    # dl = dataset.get_batch_loader(batch_size=10)
    # for batch in dl:
    #     for k, v in batch.items():
    #         print(k, v.shape)
    #         print(v)
    #     print("")


    model_do = 0.05
    model_config = {
        "s_embed_dim": 16,
        "s_embed_layers": [16],
        "s_embed_do": model_do,
        "a_embed_dim": 16,
        "sa_feature_dim": 32,
        "sa_feature_layers": [32],
        "sa_feature_do": model_do,
        "q_layers": [32, 16],
        "q_do": model_do,
        "beta_layers": [32, 16],
        "beta_do": model_do,
        "w_layers": [32, 16],
        "w_do": model_do,
        "eta_layers": [32, 16],
        "eta_do": model_do,
    }
    s_dim = env.get_s_dim()
    num_a = env.get_num_a()
    model = FeedForwardNuisanceModel(s_dim=s_dim, num_a=num_a, gamma=gamma,
                                     config=model_config)
    critic_class = FeedForwardCritic
    critic_do = 0.05
    critic_config = {
        "s_embed_dim": 8,
        "s_embed_layers": [8],
        "s_embed_do": critic_do,
        "a_embed_dim": 8,
        "critic_layers": [16],
        "critic_do": critic_do,
    }
    critic_kwargs = {
        "s_dim": s_dim,
        "num_a": num_a,
        "config": critic_config
    }
    # learner = MinMaxLearner(
    #     nuisance_model=model,
    #     gamma=gamma, 
    #     adversarial_lambda=adversarial_lambda
    # )
    learner_1 = IterativeSieveLearner(
        nuisance_model=model,
        gamma=gamma,
        adversarial_lambda=adversarial_lambda,
        train_q_xi=False, train_eta=True, train_w=True,
    )
    test_dataset = OfflineRLDataset()
    test_dataset.sample_new_trajectory(env=env, pi=pi_b, burn_in=1000,
                                       num_sample=num_sample, thin=10)
    test_dataset.apply_eval_policy(pi_e_name, pi_e)
    dl_test = test_dataset.get_batch_loader(batch_size=1024)
    s_init, a_init = env.get_s_a_init(pi_e)
    evaluate_pv_kwargs = {
        "s_init": s_init, "a_init": a_init,
        "dl_test": dl_test, "pi_e_name": pi_e_name,
    }
    learner_1.train(
        dataset, init_basis_func=env.init_basis_func,
        num_init_basis=env.get_num_init_basis_func(),
        pi_e_name=pi_e_name, verbose=True,
        evaluate_pv_kwargs=evaluate_pv_kwargs, critic_class=critic_class,
        s_init=s_init, critic_kwargs=critic_kwargs,
    )
    model.save_model("tmp_model")

    model = FeedForwardNuisanceModel.load_model("tmp_model")

    q_pv = model.estimate_policy_val_q(
        s_init=s_init, a_init=a_init, gamma=gamma
    )
    w_pv = model.estimate_policy_val_w(dl=dl_test)
    dr_pv = model.estimate_policy_val_dr(
        s_init=s_init, a_init=a_init, pi_e_name=pi_e_name, dl=dl_test,
        adversarial_lambda=adversarial_lambda, gamma=gamma
    )
    print(f"EVALUATING FINAL BEST MODEL:")
    print(f"Q-estimated v(pi_e): {q_pv}")
    print(f"W-estimated v(pi_e): {w_pv}")
    print(f"DS/DV-estimated v(pi_e): {dr_pv}")
    print("")

    env_eval = ToyEnv(s_init=s_threshold, adversarial=True,
                      adversarial_lambda=adversarial_lambda)
    pi_e_val = evaluate_policy(env_eval, pi_e, gamma, min_prec=1e-3)
    print(f"true v(pi_e): {pi_e_val}")
    print("")


if __name__ == "__main__":
    main()