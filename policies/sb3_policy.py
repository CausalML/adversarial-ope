from policies.abstract_policy import AbstractPolicy


class SB3Policy(AbstractPolicy):
    def __init__(self, env, model, deterministic=False):
        super().__init__(env)
        self.model = model
        self.deterministic = deterministic

    def sample_action(self, s):
        a, _ = self.model.predict(s, deterministic=self.deterministic)
        return int(a)
