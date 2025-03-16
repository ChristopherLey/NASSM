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


class StateTransition(nn.Module):
    def __init__(self, layers=(1024, 512)):
        super(StateTransition, self).__init__()
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


state_transition = StateTransition()
for W in state_transition.parameters():
    nn.init.normal_(W)
state_transition.to(device)

mu = torch.autograd.Variable(torch.ones(1, requires_grad=True)*0.0)
sigma = torch.autograd.Variable(torch.ones(1, requires_grad=True)*0.01)


def transition_kernel(x):
    result = torch.normal(x + mu, torch.ones_like(x)*sigma)
    return result


# Optimisers
expectation = optim.Adam([mu, sigma])
maximisation = optim.Adam(state_transition.parameters())

Characterisation_Set = pickle.load(open("Battery_Data/new_battery_cycles/Characterisation_Set.p", 'rb'))

SoC_max, SoC_min = Characterisation_Set['preprocessing']['SoC']
Current_max, Current_min = Characterisation_Set['preprocessing']['Current']
charSet = batterySet[5]


def train(N, start, stop, state, optimiser, g_std=0.02):
    first = True
    scaled_state = torch.empty_like(state)
    nu = torch.Tensor([1.0 / (0.2 * np.sqrt(2 * np.pi))])
    optimiser.zero_grad()
    print(mu.grad)
    print(sigma.grad)
    for p in state_transition.parameters():
        print(p.grad)

    for t in range(start, stop):

        V_measured = torch.Tensor([charSet['Voltage'][0, t]])

        state[:, 1:2] = (charSet['Current'][0, t] - Current_min)/(Current_max - Current_min)

        # State Transition
        transition = state_transition(state.to(device)).to("cpu")
        kernel_result = transition_kernel(transition)
        state[:, 0:1] = kernel_result[:, 0:1]

        # Bounds
        max_test = state[:, 0] > 1.0
        state[max_test, 0] = 1.0
        min_test = state[:, 0] < 0.0
        state[min_test, 0] = 0.0000000001

        # Observation
        scaled_SoC = (state - SoC_min)/(SoC_max - SoC_min)
        scaled_state[:, 0:1] = scaled_SoC[:, 0:1]
        scaled_current = (charSet['Current'][0, t] - Current_min)/(Current_max - Current_min)
        scaled_state[:, 1] = scaled_current
        V = observation(scaled_state.to(device)).to("cpu")
        V_est = V.clone()

        # Likelihood
        logW = torch.log(nu) - 0.5*torch.pow((V_est - V_measured)/g_std, 2.0)
        max_logW = logW.max()
        loss_W = torch.exp(logW - max_logW)

        if not first:
            loss = loss + max_logW + torch.log(torch.sum(loss_W)) - torch.Tensor([np.log(N)])
        else:
            loss = max_logW + torch.log(torch.sum(loss_W)) - torch.Tensor([np.log(N)])
            first = False

        # resampled, W = resample(state, loss_W)
        # state[:, 0:1] = resampled[:, 0:1]

    loss = -1*loss  # We are **maximising** the ELBO => minimising KL divergence
    print(loss)
    loss.backwards()
    optimiser.step()

    print(loss.grad)
    print(mu.grad)
    print(sigma.grad)
    for p in state_transition.parameters():
        print(p.grad)
    return state, loss


def linspace_iter(data, num):
    iterable = np.linspace(0, data.shape[1], num + 1, dtype=np.int)
    for i in range((len(iterable) - 1)):
        yield iterable[i], iterable[i + 1]


filename = datetime.now().isoformat()

E_loss_history = []
M_loss_history = []

accum_time = 0
epochs = 2000
partial = 0
N = 100
min_loss = 1e30

for epoch in range(epochs):
    print("epoch", epoch)
    start_time = time.time()
    state = torch.ones(N, 2, requires_grad=True) * 1.0

    # # Expectation
    E_accum_loss = 0
    # iterator = linspace_iter(charSet['Voltage'][0, 0:10], 1)
    # for start, stop in iterator:
    #     state, loss = train(N, start, stop, state, expectation)
    #     state.detach_()
    #     E_accum_loss += loss.item()
    # E_loss_history.append(E_accum_loss)
    # print("Variational parameters", {"mu": mu, "sigma": sigma})
    # if E_accum_loss < min_loss:
    #     torch.save(state_transition.state_dict(), "./Trained_Models/{}.mdl".format(filename))
    #
    #     pickle.dump({"mu": mu, "sigma": sigma}, open("./Trained_Models/{}.p".format(filename), 'wb'))
    #     min_loss = E_accum_loss

    # Maximisation
    M_accum_loss = 0
    iterator = linspace_iter(charSet['Voltage'][0, 0:1000], 1)
    for start, stop in iterator:
        state, loss = train(N, start, stop, state, maximisation)
        state.detach_()
        M_accum_loss += loss.item()
    M_loss_history.append(M_accum_loss)

    if M_accum_loss < min_loss:
        torch.save(state_transition.state_dict(), "./Trained_Models/{}.mdl".format(filename))
        print("New Model minimum", {"mu": mu, "sigma": sigma})
        pickle.dump({"mu": mu, "sigma": sigma}, open("./Trained_Models/{}.p".format(filename), 'wb'))
        min_loss = M_accum_loss

    print("Epoch:", epoch, "Expectation loss:", E_accum_loss, "Maximisation loss:", M_accum_loss)
    inter_time = (time.time() - start_time) / 60.0
    print("exec time:", inter_time, "min")
    accum_time += inter_time
    print("time_remaining:", accum_time / (epoch + 1) * epochs - accum_time, "min")

