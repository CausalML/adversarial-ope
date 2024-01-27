


import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
import torch
import numpy as np

register(
    id = "ToyEnv",
    entry_point = "environments.toy_env:ToyEnv"
)

class ToyEnv(gym.Env):
    def __init__(self, init_s=1.0, adversarial=True, adversarial_lambda=2.0):
        super(ToyEnv, self).__init__()
        self.s = None

        self.init_s = init_s
        self.s_max_val = 5.0
        self.adversarial = adversarial
        self.adversarial_lambda = adversarial_lambda

        self.control_frac= 0.2

        self.drift_min = -0.2
        self.drift_max = 1.0

        self.control_drift_min = -0.1
        self.control_drift_max = 0.5

        self.max_risk = self.s_max_val ** 2
        # self.max_risk = self.s_max_val

        self.observation_space = spaces.Box(low=0, high=self.s_max_val)
        self.action_space = spaces.Discrete(2)

        self.PASS_ACTION = 0
        self.CONTROL_ACTION = 1
        self.ACTION_IDX_TO_NAME = {
            self.PASS_ACTION: "pass",
            self.CONTROL_ACTION: "control",
        }

    def reset(self, seed=None, options=None):
        self.s = self.init_s
        return np.array([self.s], dtype="float32"), {}

    def step(self, action):
        # first decide if we are doing regular or adversarial transition

        if not self.adversarial:
            s_updated = self._transition_regular(action)
        else:
            prob_adversarial = 1 - 1 / self.adversarial_lambda
            if np.random.rand() > prob_adversarial:
                s_updated = self._transition_regular(action)
            else:
                s_updated = self._transition_adversarial(action)

        risk = (self.s ** 2 + s_updated ** 2 + self.s * s_updated) / 3.0
        # risk = (self.s + s_updated) / 2.0
        reward = (self.max_risk - risk) / self.max_risk
        self.s = s_updated
        return np.array([s_updated], dtype="float32"), reward, False, False, {}

    def _get_transition_min_max(self, action):
        assert action in self.ACTION_IDX_TO_NAME
        a_name = self.ACTION_IDX_TO_NAME[action]

        # get normal range for non-adversarial transition
        if a_name == "control":
            s_min = self.control_frac * (self.s + self.control_drift_min)
            s_max = self.s + self.control_drift_max
        elif a_name == "pass":
            s_min = self.s + self.drift_min
            s_max = self.s + self.drift_max
        else:
            raise ValueError(f"invalid action name: {a_name}")
        s_min_fixed = max(0, s_min)
        s_max_fixed = min(self.s_max_val, s_max) 
        assert s_min_fixed < s_max_fixed
        return s_min_fixed, s_max_fixed

    def _transition_regular(self, action):
        s_min, s_max = self._get_transition_min_max(action)
        return np.random.uniform(s_min, s_max)

    def _transition_adversarial(self, action):
        s_min_raw, s_max = self._get_transition_min_max(action)
        adversarial_alpha = 1 / (1 + self.adversarial_lambda)
        s_min = s_max - adversarial_alpha * (s_max - s_min_raw)
        return np.random.uniform(s_min, s_max)
 
    def get_s_dim(self):
        return 1

    def get_num_a(self):
        return 2

    def get_init_s_a(self, pi_e, device=None):
        init_s = np.array([self.init_s], dtype="float32")
        init_a = pi_e(init_s)
        init_s_torch = torch.from_numpy(init_s)
        init_a_torch = torch.LongTensor([init_a])
        if device is None:
            return init_s_torch, init_a_torch
        else:
            return init_s_torch.to(device), init_a_torch.to(device)
