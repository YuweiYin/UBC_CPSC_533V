import numpy as np
import torch
from torch import nn
from torch import optim

from .base_critic import BaseCritic
from ..infrastructure import pytorch_util as ptu
from ..infrastructure import sac_utils


class SACCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """

    def __init__(self, hparams):
        super(SACCritic, self).__init__()
        self.ob_dim = hparams["ob_dim"]
        self.ac_dim = hparams["ac_dim"]
        self.discrete = hparams["discrete"]
        self.size = hparams["size"]
        self.n_layers = hparams["n_layers"]
        self.learning_rate = hparams["learning_rate"]

        # critic parameters
        self.gamma = hparams["gamma"]
        self.Q1 = ptu.build_mlp(
            self.ob_dim + self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            activation="relu",
        ).to(ptu.device)
        self.Q2 = ptu.build_mlp(
            self.ob_dim + self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
            activation="relu",
        ).to(ptu.device)

        self.loss_func = nn.MSELoss()

        self.optimizer = optim.Adam(
            self.parameters(),
            self.learning_rate,
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        # Return the two q values
        inputs = torch.cat((obs, action), 1)
        q1 = self.Q1(inputs)
        q2 = self.Q2(inputs)

        return q1, q2

    def update(self, ob_no, ac_na, target_q):
        inputs = torch.cat((ob_no, ac_na), 1)
        target_q = target_q.unsqueeze(1).detach()
        q1_pred = self.Q1(inputs)
        q2_pred = self.Q2(inputs)

        loss_q1 = self.loss_func(q1_pred, target_q)
        loss_q2 = self.loss_func(q2_pred, target_q)
        loss = loss_q1 + loss_q2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
