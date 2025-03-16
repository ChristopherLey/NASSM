import pickle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from datetime import datetime
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ParticleFilter.Tools import resample

batterySet = pickle.load(open("./Battery_Data/new_battery_cycles/new_battery_v2.p", 'rb'))


class Observation(nn.Module):
    def __init__(self, layers=(1024, 512)):
        super(Observation, self).__init__()
        self.fc1 = nn.Linear(2, layers[0])
        self.depth = len(layers)
        if len(layers) >= 2:
            self.fc2 = nn.Linear(layers[0], layers[1])
        if len(layers) == 3:
            self.fc3 = nn.Linear(layers[1], layers[2])
        self.out = nn.Linear(layers[-1], 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        if self.depth >= 2:
            x = torch.sigmoid(self.fc2(x))
        if self.depth == 3:
            x = torch.sigmoid(self.fc3(x))
        x = self.out(x)
        return x


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
observation = Observation()
observation.load_state_dict(torch.load("./Battery_Data/new_battery_cycles/Observation_v2.mdl"))
observation.to(device)

Characterisation_Set = pickle.load(open("Battery_Data/new_battery_cycles/Characterisation_Set.p", 'rb'))

SoC_max, SoC_min = Characterisation_Set['preprocessing']['SoC']
Current_max, Current_min = Characterisation_Set['preprocessing']['Current']
charSet = batterySet[5]

g_std = 0.01
f_std = 0.01


class VPF(nn.Module):
    def __init__(self, layers=(1024, 512)):
        super(VPF, self).__init__()
        self.fc1 = nn.Linear(2, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.out = nn.Linear(layers[-1], 1)
        self.nu = torch.Tensor([1.0 / (g_std * np.sqrt(2 * np.pi))])

    def scale(self, state):
        state[0, 0:1] = (state[0, 0:1] - SoC_min) / (SoC_max - SoC_min)
        state[0, 1:2] = (state[0, 1:2] - Current_min) / (Current_max - Current_min)
        return state

    def forward(self, sensor, state, observer_model):
        # State evolution
        state[:, 0:1] = state[:, 0:1] - self.state_transition(state)
        state[:, 0:1] = state[:, 0:1] + \
                        torch.normal(torch.zeros([state.shape[0], 1]), torch.ones([state.shape[0], 1]) * f_std)

        # Bounds
        max_test = state[:, 0] > 1.0
        state[max_test, 0] = 1.0
        min_test = state[:, 0] < 0.0
        state[min_test, 0] = 0.0000000001

        V = observer_model(self.scale(state).to(device)).to("cpu")
        W = self.nu * torch.exp(-0.5 * torch.pow((V - sensor) / g_std, 2.0))

        logW = torch.log(self.nu) - 0.5 * torch.pow((V - sensor) / g_std, 2.0)

        return W, logW, state

    def state_transition(self, state):
        state = self.scale(state)
        state = state.to(device)
        F.dropout()
        state = torch.sigmoid(self.fc1(state))
        state = torch.sigmoid(self.fc2(state))
        state = self.out(state)
        return state.to("cpu")


vpf = VPF()
for p in vpf.parameters():
    nn.init.normal_(p)

vpf.to(device)
optimiser = optim.Adam(vpf.parameters())


def train(N, start, stop, state):
    first = True
    vpf.zero_grad()
    for t in range(start, stop):

        V_measured = torch.Tensor([charSet['Voltage'][0, t]])

        state[:, 1] = charSet['Current'][0, t]

        W, logW, state = vpf(V_measured, state, observation)

        max_logW = logW.max()
        loss_W = torch.exp(logW - max_logW)

        if not first:
            loss = loss + max_logW + torch.log(torch.sum(loss_W)) - torch.Tensor([np.log(N)])
        else:
            loss = max_logW + torch.log(torch.sum(loss_W)) - torch.Tensor([np.log(N)])
            first = False

        # Resampling
        resampled, W = resample(state, loss_W)
        state[:, 0:1] = resampled[:, 0:1]

    loss = -1*loss  # We are **maximising** the ELBO => minimising KL divergence
    loss.backward()

    optimiser.step()
    print("partial loss", loss.item())
    return state, loss


def linspace_iter(data, num):
    iterable = np.linspace(0, data.shape[1], num + 1, dtype=np.int)
    for i in range((len(iterable) - 1)):
        yield iterable[i], iterable[i + 1]


filename = datetime.now().isoformat()

loss_history = []
import time
accum_time = 0
epochs = 2000
partial = 0
N = 100
torch.Tensor()

for epoch in range(epochs):
    print("epoch", epoch)
    start_time = time.time()
    state = torch.ones(N, 2) * 1.0

    min_loss = 1e30
    accum_loss = 0
    inter_loss = []
    iterator = linspace_iter(charSet['Voltage'], 1)
    for start, stop in iterator:
        state, loss = train(N, start, stop, state)
        state.detach_()
        inter_loss.append([loss.item()])
        accum_loss += loss.item()
    if epoch % 100 == 0:
        partial += 1
    if accum_loss < min_loss:
        torch.save(vpf.state_dict(), "./Trained_Models/{}_part_{}.mdl".format(filename, partial))
        min_loss = accum_loss
    inter_time = (time.time() - start_time)/60.0
    print("Total loss: {:.4e}".format(accum_loss))
    print("exec time:", inter_time, "min")
    accum_time += inter_time
    print("time_remaining:", accum_time/(epoch + 1)*epochs - accum_time, "min")
    loss_history.append(inter_loss)
