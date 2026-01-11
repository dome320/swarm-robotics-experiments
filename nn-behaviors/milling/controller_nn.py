import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from collections import deque

class MillingNNController(nn.Module):
    v_max = 0.27
    w_max = 0.60

    def __init__(self, in_features=8, h1=8, h2=9, out_features=2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2) 
        self.out = nn.Linear(h2, out_features)   

        #Using an agent history of binary sensor readings
        self._h_hist = {} # agent_id -> deque(maxlen=7)

    def forward(self, obs):
        x = F.relu(self.fc1(obs)) 
        x = F.relu(self.fc2(x)) 
        raw_action = self.out(x)
        return raw_action        

    def act(self, agent_id: int, h):
        obs = self.build_observation(agent_id, h)
        raw_action = self.forward(obs)

        raw_v = raw_action[..., 0]
        raw_w = raw_action[..., 1]

        v = torch.sigmoid(raw_v) * self.v_max
        w = torch.tanh(raw_w) * self.w_max

        return v, w
    
    def get_params_vector(self) -> torch.Tensor:
        parts = []

        for p in self.parameters():
            parts.append(p.detach().view(-1))
        return torch.cat(parts, dim=0) 

    def set_params_vector(self, vec:  torch.Tensor) -> None:
        # Ensuring that the vec passsed in is Tensor 
        if not isinstance(vec, torch.Tensor):
            vec = torch.tensor(vec,dtype=torch.float32)

        vec = vec.to(next(self.parameters()).device).float() #Just ensuring params are on same device

        idx = 0
        for p in self.parameters():
            n = p.numel() #how many numbers this param needs
            new_vals = vec[idx:idx+n].view_as(p) #slicing and reshaping parameters 
            with torch.no_grad():
                p.copy_(new_vals) # Overwriting parameter values 
            idx += n
        
        if idx != vec.numel(): 
            raise ValueError(f"Vector has {vec.numel()} elems, but model uses {idx}") #Catching mismatched lengths early 

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters()) #helper method to allocate mutation vectors
    
    def build_observation(self, agent_id: int, h) -> torch.Tensor:
        #7-step history of binary sensor h (0 or 1) + bias which returns tensor shape 8 

        #normalize h to exactly 0.0 or 1.0 
        h = 1.0 if float(h) > 0.5 else 0.0 

        #initialize history for the agent if needed
        if agent_id not in self._h_hist:
            self._h_hist[agent_id] = deque([0.0] *7, maxlen=7)
        
        hist = self._h_hist[agent_id] 
        hist.appendleft(h) #keeping newest index at 0 for this implementation

        obs = list(hist) + [1.0] 
        return torch.tensor(obs, dtype=torch.float32)


        
