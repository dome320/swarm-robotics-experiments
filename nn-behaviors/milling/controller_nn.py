import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from swarmsim.agent.control.Controller import Controller


class MillingNNController(Controller, nn.Module):
    v_max = 0.27
    w_max = 0.60

    def __init__(self, agent, genome=None, in_features=8, h1=8, h2=9):
        Controller.__init__(self, agent)
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, 2)

        #binary sensor history for each agent
        self._h_hist = deque([0.0] * 7, maxlen=7)

        if genome is not None:
            self.set_params_vector(genome)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def get_actions(self, agent): #Called automatically by MazeAgent.step()
        
        #BinaryFOVSensor must already on
        h = self.agent.sensors[0].current_state
        h = 1.0 if h else 0.0

        #update history
        self._h_hist.appendleft(h)

        #construct an oberservation 
        obs = torch.tensor(
            list(self._h_hist) + [1.0],
            dtype=torch.float32
        )

        raw_action = self.forward(obs)

        v = torch.sigmoid(raw_action[0]) * self.v_max
        w = torch.tanh(raw_action[1]) * self.w_max

        return [float(v.item()), float(w.item())]

    #evolution helper methods
    def get_params_vector(self):
        return torch.cat([p.detach().view(-1) for p in self.parameters()])

    def set_params_vector(self, vec):
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec, dtype=torch.float32)

        idx = 0
        for p in self.parameters():
            n = p.numel()
            with torch.no_grad():
                p.copy_(vec[idx:idx+n].view_as(p))
            idx += n

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
