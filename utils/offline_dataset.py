import torch
from torch.utils.data import Dataset, DataLoader
from policies.abstract_policy import AbstractPolicy

class OfflineRLDataset(Dataset):
    def __init__(self, env, pi, burn_in, num_sample, thin=1):
        super().__init__()
        assert isinstance(pi, AbstractPolicy)

        # first, run some burn-in iterations
        s, _ = env.reset()
        for _ in range(burn_in):
            a = pi(s)
            s, _, _, _, _ = env.step(a)

        # now, sample data from stationary distribution
        s_list = []
        ss_list = []
        a_list = []
        r_list = []
        for i_ in range(num_sample * thin):
            a = pi(s)
            ss, r, _, _, _ = env.step(a)
            if i_ % thin == 0:
                s_list.append(torch.from_numpy(s))
                a_list.append(a)
                ss_list.append(torch.from_numpy(ss))
                r_list.append(r)
            s = ss

        # finally, convert sampled data into tensors
        self.s = torch.stack(s_list)
        self.a = torch.LongTensor(a_list)
        self.ss = torch.stack(ss_list)
        self.r = torch.FloatTensor(r_list)
        self.ss_list = ss_list
        self.pi_ss = {}

    def get_batch_loader(self, batch_size):
        return DataLoader(self, batch_size=batch_size)

    def apply_eval_policy(self, pi_eval_name, pi_eval):
        assert pi_eval_name not in self.pi_ss
        pi_ss_list = [pi_eval(ss_) for ss_ in self.ss_list]
        self.pi_ss[pi_eval_name] = torch.LongTensor(pi_ss_list)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        x = {
            "s": self.s[i],
            "a": self.a[i],
            "ss": self.ss[i],
            "r": self.r[i],
        }
        for pi_eval_name, pi_ss in self.pi_ss.items():
            x[pi_eval_name] = pi_ss[i]
        return x