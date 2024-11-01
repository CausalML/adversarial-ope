import torch
import torch.nn.functional as F
import numpy as np

import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
from stable_baselines3.common.callbacks import BaseCallback

from gym_sepsis.envs.sepsis_env import NUM_FEATURES, NUM_ACTIONS

register(
    id = "ToyEnv",
    entry_point = "environments.toy_env:ToyEnv"
)

class SepsisEnvWrapper(gym.Env):
    '''
    This a wrapper around the SepsisEnv environment, which:

    (1) Implements same additional methods as ToyEnv, which are required by
        this codebase
    (2) Ensures rewards are in ranage [0, 1], to be in-line with theory
    (3) Ensures deterministic starting state on call to .reset()
    (4) States are flattened (rank 1 tensor, rather than rank 3), which
        matches assumptions of this codebase
    (5) number of outputs from .step() and .reset() match the standard assumd
        by gymnasium (5 and 2 outputs respectively)
    (6) Changes environment to be infinite-horizon, by resetting the
        base environment to a random new starting state whenever it terminates

    '''
    def __init__(self, base_env, s_init_idx=0):
        super(SepsisEnvWrapper, self).__init__()
        self.base_env = base_env
        self.avail_start_states = self.base_env.starting_states
        self.num_start_states = len(self.avail_start_states)
        self.s_init_idx = s_init_idx

        self.s = None

        self.observation_space = spaces.Box(
            low=-100,
            high=100,
            shape=(NUM_FEATURES-2,),
            # dtype=np.float64,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.logging = False
        self.reward_buffer = []

    def reset(self, seed=None, options=None):
        # use fixed starting state

        start_state = self._get_starting_state_from_idx(self.s_init_idx)
        obs = self.base_env.reset(starting_state=start_state)
        self.s = SepsisEnvWrapper._convert_base_env_obs(obs)

        return self.s, {}

    def step(self, action):
        obs, base_reward, done, _ = self.base_env.step(action)
        # base reward is either:
        # a) -15 (if terminated, and patient died),
        # b) 15 (if terminated, and patient survived),
        # c) 0 (if not terminated yet)
        
        reward = (base_reward + 15.0) / 30.0
        if done:
            # do an internal reset of base env, from a random staring state
            start_idx = np.random.randint(0, self.num_start_states)
            start_state = self._get_starting_state_from_idx(start_idx)
            obs = self.base_env.reset(starting_state=start_state)

        self.s = SepsisEnvWrapper._convert_base_env_obs(obs)

        if self.logging:
            self.reward_buffer.append(reward)

        return self.s, reward, False, False, {}
    
    @staticmethod
    def _convert_base_env_obs(obs):
        return obs.flatten().astype(np.float32)
        # return obs.flatten()

    def _get_starting_state_from_idx(self, idx):
        start_states = self.base_env.starting_states
        max_idx = len(start_states) - 1
        assert isinstance(idx, int), \
            'Starting state index must be an integer'
        assert idx >= 0, \
            'Starting state index must be non-negative'
        assert idx <= max_idx, \
            f'Starting state index cannot exceed {idx}'
        return self.base_env.starting_states[idx][:-1]

    def get_s_dim(self):
        return NUM_FEATURES - 2

    def get_num_a(self):
        return NUM_ACTIONS

    def get_s_a_init(self, pi_e, device=None):
        obs_init = self._get_starting_state_from_idx(self.s_init_idx)
        s_init = SepsisEnvWrapper._convert_base_env_obs(obs_init)
        a_init = pi_e(s_init)
        s_init_torch = torch.from_numpy(s_init)
        a_init_torch = torch.LongTensor([a_init])
        if device is None:
            return s_init_torch, a_init_torch
        else:
            return s_init_torch.to(device), a_init_torch.to(device)

    def bias_basis_func(self, s, a=None):
        assert len(s.shape) == 2, \
            'Expected batch of states (rank 2 input)'
        bias = torch.ones(len(s), 1)
        return bias.to(s.device)

    def get_callback(self, log_freq=200):
        self.logging = True
        return SepsisEnvCallback(self, log_freq)


class SepsisEnvCallback(BaseCallback):
    def __init__(self, env, log_freq):
        BaseCallback.__init__(self, verbose=0)
        self.env = env
        self.log_freq = log_freq
        self.t = 0

    def _on_step(self) -> bool:
        self.t += 1
        buffer_size = len(self.env.reward_buffer)
        if buffer_size >= self.log_freq:
            reward_list = self.env.reward_buffer
            # do log, then clear buffer
            mean_reward = np.mean(reward_list)
            num_dead = len([r_ for r_ in reward_list if r_ == 0])
            num_survived = len([r_ for r_ in reward_list if r_ == 1])
            num_done = num_dead + num_survived
            if num_done > 0:
                survival_rate = num_survived / num_done
                mean_time = (len(reward_list) - num_done) / num_done
            else:
                survival_rate = 0
                mean_time = self.log_freq
            self.logger.record('performance/mean_reward', mean_reward)
            self.logger.record('performance/survival_rate', survival_rate)
            self.logger.record('performance/mean_time_in_ICU', mean_time) 
            self.logger.record('performance/num_deaths_in_batch', num_dead) 
            self.logger.record('performance/num_survived_in_batch', num_survived) 
            self.logger.record('performance/num_done_in_batch', num_done) 
            self.logger.dump(step=self.t)
            self.env.reward_buffer = []
        return True