"""Model for PPO on holodeck Env

Note: rlpyt models should use infer_leading_dims and _restore_leading_dims
to handle variable tensor shapes. The PPO algo passes state, prev_action, 
and prev_reward into step and expects policy and value vectors in return.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class PpoHolodeckModel(nn.Module):

    def __init__(self, img_size, lin_size, action_size, is_continuous, 
                 hidden_size=1024):
        super().__init__()

        self.is_continuous = is_continuous
        in_channel = img_size[0]
        conv_out = int(img_size[1] * img_size[2])
        linear_in = conv_out + lin_size

        self._conv_layers = nn.Sequential(
            nn.Conv2d(in_channel, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, stride=2),
            nn.ReLU())

        self._linear_layers = nn.Sequential(
            nn.Linear(linear_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())

        self._policy = nn.Linear(hidden_size, action_size)

        self._value = nn.Linear(hidden_size, 1)

        self._log_std = torch.nn.Parameter(torch.zeros(action_size))

    def forward(self, state, prev_action, prev_reward):
        img, lin = state

        # T and B are rlpyt's iterations per sampling period and num envs
        lead_dim, T, B, _ = infer_leading_dims(img, 3)
        img = img.view(T * B, img.shape[-3], img.shape[-2], img.shape[-1])
        lin = lin.view(T * B, lin.shape[-1])

        img_rep = self._conv_layers(img)
        img_flat = torch.flatten(img_rep, start_dim=1)
        linear_inp = torch.cat((img_flat, lin), dim=-1)
        linear_out = self._linear_layers(linear_inp)

        pi, v = self._policy(linear_out), self._value(linear_out).squeeze(-1)
        if self.is_continuous:
            log_std = self._log_std.repeat(T * B, 1)
            mu, log_std, v = restore_leading_dims((pi, log_std, v), lead_dim, T, B)
            return mu, log_std, v
        else:
            pi = F.softmax(pi, dim=-1)
            pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
            return pi, v


