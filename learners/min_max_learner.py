import torch
from torch.optim import Adam
import numpy as np

from learners.abstract_learner import AbstractLearner
from utils.oadam import OAdam


class MinMaxLearner(AbstractLearner):
    def __init__(self, nuisance_model, gamma, adversarial_lambda):
        super().__init__(nuisance_model, gamma, adversarial_lambda)

    def train(self, dataset, pi_e_name, critic_class, critic_kwargs,
              batch_size=128, num_epoch=1000, model_lr=1e-3, critic_lr=5e-3,
              eval_freq=5):
        model = self.model
        critic = critic_class(num_out=2, **critic_kwargs)
        model_optim = OAdam(model.get_parameters(), lr=model_lr)
        critic_optim = OAdam(critic.parameters(), lr=critic_lr)

        dl = dataset.get_batch_loader(batch_size=batch_size)
        torch.autograd.set_detect_anomaly(True)
        for epoch_i in range(num_epoch):
            if epoch_i % eval_freq == 0:
                model_loss_list = []
                critic_loss_list = []
            for batch in dl:
                rho = self.rho_q_xi(
                    s=batch["s"],
                    a=batch["a"],
                    ss=batch["ss"],
                    r=batch["r"],
                    pi_ss=batch[pi_e_name],
                )
                f = critic(
                    s=batch["s"],
                    a=batch["a"]
                )
                moments = (rho * f).sum(1)
                model_loss = moments.mean()
                critic_loss = -moments.mean() + 0.5 * (moments ** 2).mean()
                # print(model_loss)
                # print(critic_loss)

                model_optim.zero_grad()
                model_loss.backward(retain_graph=True)
                model_optim.step()

                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()

                model_loss_list.append(float(model_loss))
                critic_loss_list.append(float(-critic_loss))

            if (epoch_i + 1) % eval_freq == 0:
                print(f"epoch {epoch_i + 1}")
                print(f"mean model loss: {np.mean(model_loss_list)}")
                print(f"mean critic loss: {np.mean(critic_loss_list)}")
                print("")
