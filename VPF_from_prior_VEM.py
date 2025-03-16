import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from ParticleFilter.Tools import resample


Characterisation_Set = pickle.load(open("Battery_Data/new_battery_cycles/Characterisation_Set_Complete.p", 'rb'))


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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNNetwork(nn.Module):
    def __init__(self):
        super(RNNetwork, self).__init__()
        self.Z_hl1 = nn.Linear(2, 1024)
        self.Z_hl2 = nn.Linear(1024, 512)
        self.Z_p = nn.Linear(512, 1)
        # SMC params
        self.f_mean = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.f_std = nn.Parameter(torch.Tensor([0.005]), requires_grad=True)
        # self.f_std = torch.Tensor([0.0002551068601502507])
        self.g_std = 0.01
        self.nu = torch.Tensor([1.0 / (self.g_std * np.sqrt(2 * np.pi))])
        self.w_hist = None
        self.voltage_expected_hist = None
        self.soc_expected_hist = None

    def parameters(self, divide=None):
        if divide == "Expectation":
            for name, param in self.named_parameters():
                if name == "f_mean" or name == "f_std":
                    yield param
        elif divide == "Maximisation":
            for name, param in self.named_parameters():
                if name != "f_mean" and name != "f_std":
                    yield param
        else:
            for name, param in self.named_parameters():
                yield param

    def VoC(self, SoC):
        v_L = torch.Tensor([[-1.59614486]]).to(device)
        v_0 = torch.Tensor([[4.13646328]]).to(device)
        gamma = torch.Tensor([[0.63726463]]).to(device)
        alpha = torch.Tensor([[1.40174122]]).to(device)
        beta = torch.Tensor([[2.54478965]]).to(device)
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
        scaled_soc = apply_scale(soc, SoC_min, SoC_max)
        # Estimate Z_p
        combined = torch.cat((scaled_soc, scaled_I), 1)
        Z = torch.sigmoid(self.Z_hl1(combined))
        Z = torch.sigmoid(self.Z_hl2(Z))
        Z = self.Z_p(Z)

        # Estimate prior V
        V = self.VoC(soc) - I * Z

        variational_distribution = torch.distributions.Normal(loc=self.f_mean, scale=self.f_std)

        for t in range(current.shape[1]):

            # Predict SoC
            soc = soc - I*V/Characterisation_Set['E_crit']
            # Add model uncertainty
            soc = soc + variational_distribution.rsample((N,))

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


load_prior_network = "./Battery_Data/new_battery_cycles/Battery_RNN_prior_v1.mdl"
save_network = "./Battery_Data/new_battery_cycles/Battery_RNN_vpf_from_prior_vem_v2"


print("Loading prior from:\n\t", load_prior_network)
print("Saving graph \"n\" to:\n\t", save_network + "_part_{n}.mdl")

vsmc = RNNetwork()
pretrained_dict = torch.load(load_prior_network)
model_dict = vsmc.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
vsmc.load_state_dict(model_dict)
vsmc.to(device)
Expectation_optimiser = optim.Adam(vsmc.parameters("Expectation"))
Maximisation_optimiser = optim.Adam(vsmc.parameters("Maximisation"))

expectation_loss_history = []
maximisation_loss_history = []
f_std_hist = []
f_mean_hist = []

import time
accum_time = 0
epochs = 500
parts = 20
partial = 0
N = 100


for epoch in range(epochs):
    print("epoch", epoch)
    start_time = time.time()
    state = torch.ones(N, 1) * 1.0
    min_loss = 1e30
    accum_loss = 0
    inter_expectation_loss = []
    inter_maximisation_loss = []
    inter_f_std = []
    inter_f_mean = []

    for i, set_dict in enumerate(Characterisation_Set['Sets']):
        """ Expectation """
        Expectation_optimiser.zero_grad()
        loss, _, _ = vsmc(state, set_dict['Current'], set_dict['Voltage'])
        loss = -1 * loss  # We are **maximising** the ELBO => minimising KL divergence
        loss.backward()
        Expectation_optimiser.step()

        inter_f_std.append([vsmc.f_std.item()])
        inter_f_mean.append([vsmc.f_mean.item()])
        inter_expectation_loss.append([loss.item()])
        accum_loss += loss.item()

        """ Maximisation """
        Maximisation_optimiser.zero_grad()
        loss, _, _ = vsmc(state, set_dict['Current'], set_dict['Voltage'])
        loss = -1 * loss  # We are **maximising** the ELBO => minimising KL divergence
        loss.backward()
        Maximisation_optimiser.step()

        inter_maximisation_loss.append([loss.item()])
        accum_loss += loss.item()
    if epoch % (epochs/parts) == 0:
        partial += 1
    expectation_loss_history.append(inter_expectation_loss)
    maximisation_loss_history.append(inter_maximisation_loss)
    f_mean_hist.append(inter_f_mean)
    f_std_hist.append(inter_f_std)
    if accum_loss < min_loss:
        torch.save(vsmc.state_dict(), "{0}_part_{1}.mdl".format(save_network, partial))
        min_loss = accum_loss
        with open("./Battery_Data/new_battery_cycles/Battery_RNN_vpf_from_prior_vem_v2_meta.p", 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            meta_data = {
                "Expectation": expectation_loss_history,
                "maximisation": maximisation_loss_history,
                "f_std": f_std_hist,
                "f_mean": f_mean_hist
            }
            pickle.dump(meta_data, f)
    inter_time = (time.time() - start_time) / 60.0
    print("Total loss: {:.4e}".format(accum_loss))
    print("Current Variational Distribution: N({}, {})".format(vsmc.f_mean.item(), vsmc.f_std.item()))
    print("exec time:", inter_time, "min")
    accum_time += inter_time
    remaining = accum_time / (epoch + 1) * epochs - accum_time
    hours_remaining = int(remaining // 60.0)
    min_remaining = remaining % 60.0
    print("time_remaining:", hours_remaining, "hours", min_remaining, "min")