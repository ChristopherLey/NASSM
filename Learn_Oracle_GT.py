#!/usr/bin/python
import pickle
import numpy as np
import time
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ParticleFilter.Tools import resample

parameters = pickle.load(open("Battery_Data/Oracle/oracle_random_VPF.p", 'rb'))
t = np.array(parameters['t'], ndmin=2)
V = np.array(parameters["V"], ndmin=2)
I = np.array(parameters["I"], ndmin=2)
SoC = np.array(parameters["SoC"], ndmin=2)
SoC_prior = np.concatenate(([[1.0]], SoC[np.newaxis, 0, 0:-1]), axis=1)
I_prior = np.concatenate(([[I[0, 0]]], I[np.newaxis, 0, 0:-1]), axis=1)

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


Current, Current_min, Current_max = scale(I)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_prior = True
uniform_prior = "Battery_Data/Oracle/Uniform_prior.mdl"

if train_prior:
    # Create Prior
    class Polarising_Impedance_Map(nn.Module):
        def __init__(self):
            super(Polarising_Impedance_Map, self).__init__()
            self.Z_hl1 = nn.Linear(2, 1024)
            self.Z_hl2 = nn.Linear(1024, 512)
            self.Z_p = nn.Linear(512, 1)

        def forward(self, soc_prior, current_prior):

            if soc_prior.shape[1] == 1:
                soc = torch.from_numpy(soc_prior).to(device, torch.float)
            else:
                soc = torch.from_numpy(soc_prior.T).to(device, torch.float)
            # A prior estimate
            if current_prior.shape[1] == 1:
                I = torch.from_numpy(current_prior).to(device, torch.float)
            else:
                I = torch.from_numpy(current_prior.T).to(device, torch.float)
            scaled_I = apply_scale(I, Current_min, Current_max)
            # Estimate Z_p
            combined = torch.cat((soc, scaled_I), 1)
            Z = torch.sigmoid(self.Z_hl1(combined))
            Z = torch.sigmoid(self.Z_hl2(Z))
            Z = self.Z_p(Z)
            return Z




    Z_pol = Polarising_Impedance_Map()
    for W in Z_pol.parameters():
        nn.init.normal_(W)
    Z_pol.to(device)
    optimiser = optim.Adam(Z_pol.parameters())
    criterion = nn.MSELoss() # Mean Squared Loss


    epochs = 50000
    running_loss = 0.0
    loss_min = 1e15
    loss_hist = []

    start = time.time()

    target = torch.Tensor([[0.12]]).to(device)


    def timeSince(since):
        now = time.time()
        interval = now - since
        m = math.floor(interval / 60)
        s = interval - m * 60
        return '%dm %ds' % (m, s), now, interval


    for epoch in range(epochs):
        count = -1
        avg_loss = 0

        optimiser.zero_grad()
        Z_est = Z_pol(SoC, I)
        loss = criterion(Z_est, target)
        loss.backward()
        optimiser.step()
        avg_loss += loss.item()
        loss_hist.append(loss.item())

        if avg_loss < loss_min:
            torch.save(Z_pol.state_dict(), uniform_prior)
            loss_min = avg_loss

        if epoch % 100 == 0:
            now_string, now, interval = timeSince(start)
            remaining_epochs = epochs - (epoch + 1)
            remaining_time = interval * remaining_epochs / (epoch + 1)
            h_f = remaining_time / 60.0 / 60.0
            h = math.floor(h_f)
            m_f = (h_f - h) * 60.0
            m = math.floor(m_f)
            s = (m_f - m) * 60.0
            # sys.stdout.write("\033[F")  # Cursor up one line
            # sys.stdout.write("\033[K")  # Clear to the end of line
            # sys.stdout.write("\033[F")  # Cursor up one line
            # sys.stdout.write("\033[K")  # Clear to the end of line
            remaining_string = '%dh %dm %ds' % (h, m, s)
            time_string = "epoch {}, time since start: {}, estimated remaining time: {}".format(epoch, now_string,
                                                                                                remaining_string)
            print(time_string)
            loss_string = "New average minimum: {}".format(loss_min)
            print(loss_string)


class OracleNetwork(nn.Module):
    def __init__(self):
        super(OracleNetwork, self).__init__()
        self.Z_hl1 = nn.Linear(2, 1024)
        self.Z_hl2 = nn.Linear(1024, 512)
        self.Z_p = nn.Linear(512, 1)

    def VoC(self, SoC):
        v_L = torch.Tensor([[parameters["v_L"]]]).to(device)
        v_0 = torch.Tensor([[parameters["v_0"]]]).to(device)
        gamma = torch.Tensor([[parameters["gamma"]]]).to(device)
        alpha = torch.Tensor([[parameters["alpha"]]]).to(device)
        beta = torch.Tensor([[parameters["beta"]]]).to(device)
        return v_L + (v_0 - v_L)*torch.exp(gamma*(SoC - 1)) + alpha*v_L*(SoC - 1) \
            + (1 - alpha)*v_L*(torch.exp(-beta) - torch.exp(-beta*torch.sqrt(SoC)))

    def forward(self, soc_prior, current_prior, current_posterior):

        if soc_prior.shape[1] == 1:
            soc = torch.from_numpy(soc_prior).to(device, torch.float)
        else:
            soc = torch.from_numpy(soc_prior.T).to(device, torch.float)
        # A prior estimate
        if current_prior.shape[1] == 1:
            I = torch.from_numpy(current_prior).to(device, torch.float)
        else:
            I = torch.from_numpy(current_prior.T).to(device, torch.float)

        scaled_I = apply_scale(I, Current_min, Current_max)
        combined = torch.cat((soc, scaled_I), 1)
        Z = torch.sigmoid(self.Z_hl1(combined))
        Z = torch.sigmoid(self.Z_hl2(Z))
        Z = self.Z_p(Z)
        V = self.VoC(soc) + I * Z

        soc = soc - I*V*parameters['E_crit_inv']

        # Bounds
        max_test = soc[:, 0] > 1.0
        soc[max_test, 0] = 1.0
        min_test = soc[:, 0] < 0.0
        soc[min_test, 0] = 0.0000000001

        # posterior
        if current_posterior.shape[1] == 1:
            I_post = torch.from_numpy(current_posterior).to(device, torch.float)
        else:
            I_post = torch.from_numpy(current_posterior.T).to(device, torch.float)
        scaled_I = apply_scale(I_post, Current_min, Current_max)
        # Estimate Z_p
        combined = torch.cat((soc, scaled_I), 1)
        Z = torch.sigmoid(self.Z_hl1(combined))
        Z = torch.sigmoid(self.Z_hl2(Z))
        Z = self.Z_p(Z)
        V = self.VoC(soc) - I_post*Z
        return V


load_prior_network = uniform_prior
save_network = "Battery_Data/Oracle/Learnt_Oracle_gt_with_prior_v1.mdl"

print("Loading prior from:\n\t", load_prior_network)
print("Saving graph \"n\" to:\n\t", save_network)

vsmc = OracleNetwork()
if load_prior_network:
    vsmc.load_state_dict(torch.load(load_prior_network))
else:
    for W in vsmc.parameters():
        nn.init.normal_(W)
vsmc.to(device)
optimiser = optim.Adam(vsmc.parameters())
criterion = nn.MSELoss()

loss_history = []
accum_time = 0
epochs = 100000
parts = 10
partial = 0
N = 100

if V.shape[1] == 1:
    V_gt = torch.from_numpy(V).to(device, torch.float)
else:
    V_gt = torch.from_numpy(V.T).to(device, torch.float)

for epoch in range(epochs):
    print("epoch", epoch)
    start_time = time.time()
    min_loss = 1e30
    accum_loss = 0
    inter_loss = []

    optimiser.zero_grad()
    V_est = vsmc(SoC, I, I)
    loss = criterion(V_est, V_gt)
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
with open(save_network + "_meta.p", 'wb') as f:
    meta_data = {
        "Loss": loss_history,
    }
    pickle.dump(meta_data, f)
