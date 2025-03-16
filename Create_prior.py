#!/usr/bin/python
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ParticleFilter.Tools import resample

Characterisation_Set = pickle.load(open("Battery_Data/new_battery_cycles/Characterisation_Set_Complete.p", 'rb'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def scale(X):
    """

    :param X:
    :return: Normalised array like X, mean, std
    """
    return (X - X.min())/(X.max() - X.min()), X.min(), X.max()


def apply_scale(X, X_min, X_max):
    """

    :param X:
    :return: Normalised array like X, mean, std
    """
    return (X - X_min)/(X_max - X_min)


SoC, SoC_min, SoC_max = scale(Characterisation_Set["SoC"].T)
Current, Current_min, Current_max = scale(Characterisation_Set["Current"].T)
# Voltage, Voltage_min, Voltage_max = scale(Characterisation_Set["Voltage"].T)
Voltage = Characterisation_Set["Voltage"].T
Characterisation_Set["preprocessing"] = {
    "SoC": (SoC_max, SoC_min),
    "Current": (Current_max, Current_min)
}


class VoCNetwork(nn.Module):
    def __init__(self):
        super(VoCNetwork, self).__init__()
        self.voc_hl1 = nn.Linear(1, 512)
        self.voc_hl2 = nn.Linear(512, 256)
        self.voc_out = nn.Linear(256, 1)

    def forward(self, SoC):
        voc = torch.sigmoid(self.voc_hl1(SoC))
        voc = torch.sigmoid(self.voc_hl2(voc))
        voc = self.voc_out(voc)
        return voc


voc = VoCNetwork()
voc.to(device)
voc_network = "./Trained_Models/VoC_network_small_v1.mdl"
voc.load_state_dict(torch.load(voc_network))


class RNNetwork(nn.Module):
    def __init__(self):
        super(RNNetwork, self).__init__()
        self.Z_hl1 = nn.Linear(2, 1024)
        self.Z_hl2 = nn.Linear(1024, 512)
        self.Z_p = nn.Linear(512, 1)
        # SMC params
        self.f_std = torch.Tensor([0.005])
        self.g_std = 0.01
        self.nu = torch.Tensor([1.0 / (self.g_std * np.sqrt(2 * np.pi))])

    def VoC(self, SoC):
        return voc(SoC)

    def forward(self, soc_gt, current, voltage_measured):
        first = True
        voltage = torch.empty((1, current.shape[1]), dtype=torch.float)
        soc_hist = torch.empty((1, current.shape[1]), dtype=torch.float)
        soc = torch.Tensor([[1.0]]).to(device, torch.float)
        N = 1

        for t in range(current.shape[1]):

            # A prior estimate
            if first:
                I = torch.ones(N, 1) * current[0, t]
            else:
                I = torch.ones(N, 1) * current[0, t-1]
            I = I.to(device, torch.float)
            scaled_I = apply_scale(I, Current_min, Current_max)
            # Estimate Z_p
            combined = torch.cat((soc, scaled_I), 1)
            Z = torch.sigmoid(self.Z_hl1(combined))
            Z = torch.sigmoid(self.Z_hl2(Z))
            Z = self.Z_p(Z)
            V = self.VoC(soc) - I * Z

            # Predict SoC(t-1) -> SoC(t)
            soc = soc - I*V/Characterisation_Set['E_crit']

            # Bounds
            max_test = soc[:, 0] > 1.0
            soc[max_test, 0] = 1.0
            min_test = soc[:, 0] < 0.0
            soc[min_test, 0] = 0.0000000001

            # Posterior Evidence
            I = torch.ones(N, 1) * current[0, t]
            I = I.to(device, torch.float)
            scaled_I = apply_scale(I, Current_min, Current_max)
            # Estimate Z_p
            combined = torch.cat((soc, scaled_I), 1)
            Z = torch.sigmoid(self.Z_hl1(combined))
            Z = torch.sigmoid(self.Z_hl2(Z))
            Z = self.Z_p(Z)

            # a Priori evidence
            V = self.VoC(soc) - I*Z

            mse = torch.sum(torch.pow(V.to("cpu") - voltage_measured[0, t], 2.0)) / N

            if not first:
                loss = loss + (mse / current.shape[1])
            else:
                loss = (mse / current.shape[1])
                first = False

            voltage[:, t] = V[:, 0]
            soc_hist[:, t] = soc[:, 0]
            soc = torch.Tensor([[soc_gt[0, t]]]).to(device, torch.float)


        return loss, voltage, soc_hist


prior_set = Characterisation_Set['Sets'][0]

# load_prior_network = "./Battery_Data/new_battery_cycles/Battery_RNN_prior_v3.mdl"
load_prior_network = None
save_network = "./Battery_Data/new_battery_cycles/Battery_RNN_prior_small_VoCNN.mdl"

print("Loading prior from:\n\t", load_prior_network)
print("Saving graph \"n\" to:\n\t", save_network)

vsmc = RNNetwork()
if load_prior_network:
    vsmc.load_state_dict(torch.load(load_prior_network))
else:
    for W in vsmc.parameters():
        nn.init.normal_(W)
vsmc.to(device)
optimiser = optim.Adam(vsmc.parameters())

loss_history = []
import time
accum_time = 0
epochs = 3000
parts = 10
partial = 0
N = 100


for epoch in range(epochs):
    print("epoch", epoch)
    start_time = time.time()
    min_loss = 1e30
    accum_loss = 0
    inter_loss = []

    optimiser.zero_grad()
    loss, voltage, soc_hist = vsmc(prior_set['SoC'], prior_set['Current'], prior_set['Voltage'])
    loss.backward()
    optimiser.step()

    inter_loss.append([loss.item()])
    accum_loss += loss.item()
    if accum_loss < min_loss:
        torch.save(vsmc.state_dict(), save_network)
        min_loss = accum_loss
    inter_time = (time.time() - start_time) / 60.0
    print("Total loss: {:.4e}".format(accum_loss))
    print("exec time:", inter_time, "min")
    accum_time += inter_time
    remaining = accum_time / (epoch + 1) * epochs - accum_time
    hours_remaining = int(remaining // 60.0)
    min_remaining = remaining % 60.0
    print("time_remaining:", hours_remaining, "hours", min_remaining, "min")
    loss_history.append(inter_loss)
