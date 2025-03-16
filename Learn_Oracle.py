import pickle
import numpy as np
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from ParticleFilter.Tools import resample

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns


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


parameters = pickle.load(open("Battery_Data/Oracle/oracle_random_VPF.p", 'rb'))
t = np.array(parameters['t'], ndmin=2)
V = np.array(parameters["V"], ndmin=2)
I = np.array(parameters["I"], ndmin=2)
plt.figure()
plt.plot(t.T, V.T)
plt.show()
SoC = np.array(parameters["SoC"], ndmin=2)
Current, Current_min, Current_max = scale(I)
train_prior = False
uniform_prior = "Battery_Data/Oracle/Uniform_prior.mdl"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


    epochs = 200000
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

        if epoch % 10000 == 0:
            now_string, now, interval = timeSince(start)
            remaining_epochs = epochs - (epoch + 1)
            remaining_time = interval * remaining_epochs / (epoch + 1)
            h_f = remaining_time / 60.0 / 60.0
            h = math.floor(h_f)
            m_f = (h_f - h) * 60.0
            m = math.floor(m_f)
            s = (m_f - m) * 60.0
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
        # SMC params
        self.f_mean = torch.Tensor([0.0])
        self.f_std = torch.Tensor([0.05])
        # self.f_std = torch.Tensor([0.0002551068601502507])
        # self.g_std = 0.011334095
        self.g_std = 0.02
        self.nu = torch.Tensor([1.0 / (self.g_std * np.sqrt(2 * np.pi))])
        self.w_hist = None
        self.voltage_expected_hist = None
        self.soc_expected_hist = None

    def VoC(self, SoC):
        v_L = torch.Tensor([[parameters["v_L"]]]).to(device)
        v_0 = torch.Tensor([[parameters["v_0"]]]).to(device)
        gamma = torch.Tensor([[parameters["gamma"]]]).to(device)
        alpha = torch.Tensor([[parameters["alpha"]]]).to(device)
        beta = torch.Tensor([[parameters["beta"]]]).to(device)
        return v_L + (v_0 - v_L)*torch.exp(gamma*(SoC - 1)) + alpha*v_L*(SoC - 1) \
            + (1 - alpha)*v_L*(torch.exp(-beta) - torch.exp(-beta*torch.sqrt(SoC)))

    def forward(self, soc_init, current, voltage_measured):
        first = True
        voltage = torch.empty((soc_init.shape[0], current.shape[1]), dtype=torch.float)
        soc_hist = torch.empty((soc_init.shape[0], current.shape[1]), dtype=torch.float)
        self.w_hist = torch.empty((soc_init.shape[0], current.shape[1]), dtype=torch.float)
        self.voltage_expected_hist = torch.empty((1, current.shape[1]), dtype=torch.float)
        self.soc_expected_hist = torch.empty((1, current.shape[1]), dtype=torch.float)
        soc = soc_init.to(device, torch.float)
        N = soc_init.shape[0]

        I = torch.ones(N, 1) * current[0, 0]
        I = I.to(device, torch.float)
        scaled_I = apply_scale(I, Current_min, Current_max)
        scaled_soc = apply_scale(soc, 0.0, 1.0)
        # Estimate Z_p
        combined = torch.cat((scaled_soc, scaled_I), 1)
        Z = torch.sigmoid(self.Z_hl1(combined))
        Z = torch.sigmoid(self.Z_hl2(Z))
        Z = self.Z_p(Z)

        # Estimate prior V
        V = self.VoC(soc) - I * Z

        for t in range(current.shape[1]):

            # Predict SoC
            soc = soc - I*V*parameters["E_crit_inv"]
            # Add model uncertainty
            soc = soc + torch.normal(torch.ones([N, 1]) * self.f_mean, torch.ones([N, 1]) * self.f_std).to(device)

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

            # Estimate posterior V
            V = self.VoC(soc) - I*Z

            #SMC
            W = self.nu * torch.exp(-0.5 * torch.pow((V.to("cpu") - voltage_measured[0, t]) / self.g_std, 2.0))
            logW = torch.log(self.nu) - 0.5 * torch.pow((V.to("cpu") - voltage_measured[0, t]) / self.g_std, 2.0)

            max_logW = logW.max()
            loss_W = torch.exp(logW - max_logW)

            if not first:
                loss = loss + max_logW + torch.log(torch.sum(loss_W)) - torch.Tensor([np.log(N)])
            else:
                loss = max_logW + torch.log(torch.sum(loss_W)) - torch.Tensor([np.log(N)])
                first = False

            # Resampling
            soc, W = resample(soc, loss_W)

            self.w_hist[:, t] = W[:, 0]
            voltage[:, t] = V[:, 0]
            soc_hist[:, t] = soc[:, 0]
            self.voltage_expected_hist[0, t] = V.transpose(0, 1).mm(W.to(device))
            self.soc_expected_hist[0, t] = soc.transpose(0, 1).mm(W.to(device))

        return loss, voltage, soc_hist


load_prior_network = uniform_prior
save_network = "Battery_Data/Oracle/Learnt_Oracle_retrain_v1"
retrain = True
if retrain:
    load_prior_network = "{0}_part_{1}.mdl".format("Battery_Data/Oracle/Learnt_Oracle_v1", 20)

print("Loading prior from:\n\t", load_prior_network)
print("Saving graph \"n\" to:\n\t", save_network + "_part_{n}.mdl")

vsmc = OracleNetwork()
vsmc.load_state_dict(torch.load(load_prior_network))
vsmc.to(device)
optimiser = optim.Adam(vsmc.parameters())

loss_history = []

accum_time = 0
epochs = 1000
parts = 10
partial = 0
N = 100


for epoch in range(epochs):
    print("epoch", epoch)
    start_time = time.time()
    state = torch.ones(N, 1) * 1.0
    min_loss = 1e30
    accum_loss = 0

    optimiser.zero_grad()
    loss, _, _ = vsmc(state, I, V)
    loss = -1 * loss  # We are **maximising** the ELBO => minimising KL divergence
    loss.backward()
    optimiser.step()

    loss_history.append(loss.item())
    accum_loss += loss.item()
    if epoch % (epochs/parts) == 0:
        partial += 1

    if accum_loss < min_loss:
        torch.save(vsmc.state_dict(), "{0}_part_{1}.mdl".format(save_network, partial))
        min_loss = accum_loss
    with open(save_network + "_meta.p", 'wb') as f:
        meta_data = {
            "Loss": loss_history,
        }
        pickle.dump(meta_data, f)
    inter_time = (time.time() - start_time) / 60.0
    print("Total loss: {:.4e}".format(accum_loss))
    print("exec time:", inter_time, "min")
    accum_time += inter_time
    remaining = accum_time / (epoch + 1) * epochs - accum_time
    hours_remaining = int(remaining // 60.0)
    min_remaining = remaining % 60.0
    print("time_remaining:", hours_remaining, "hours", min_remaining, "min")
