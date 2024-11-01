import json

import pandas
from stable_baselines3 import DQN

from gym_sepsis.envs.sepsis_env import SepsisEnv
from environments.sepsis_env_wrapper import SepsisEnvWrapper
from policies.sb3_policy import SB3Policy
from utils.policy_evaluation import evaluate_policy


def main(config):
    gamma = config["gamma"]
    base_env = SepsisEnv()
    env_eval = SepsisEnvWrapper(base_env=base_env, s_init_idx=0)
    pi_e = SB3Policy(env_eval, model=DQN.load(config['dqn_model_path']))
    pi_e_val = evaluate_policy(env_eval, pi_e, gamma, min_prec=5e-5, verbose=True)
    print(f'policy value: {pi_e_val}')

if __name__ == "__main__":
    with open("configs/sepsis_config.json") as f:
        config = json.load(f)
    main(config)