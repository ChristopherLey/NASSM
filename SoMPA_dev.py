import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ParticleFilter.Tools import resample
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
import seaborn as sns
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# plt.style.use('ggplot')
sns.set()
import time
execution_metrics = []

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

E_crit_new = 26267.160775850585
E_crit_old = 21879.133773481735
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(device)


class Polarising_Impedance_Map(nn.Module):
    def __init__(self, E_crit=False):
        super(Polarising_Impedance_Map, self).__init__()
        self.Z_hl1 = nn.Linear(2, 1024)
        self.Z_hl2 = nn.Linear(1024, 512)
        self.Z_p = nn.Linear(512, 1)
        if E_crit:
            self.E_crit = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)


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


def VoC(SoC):
    v_L = torch.Tensor([[-1.59614486]]).to(device)
    v_0 = torch.Tensor([[4.13646328]]).to(device)
    gamma = torch.Tensor([[0.63726463]]).to(device)
    alpha = torch.Tensor([[1.40174122]]).to(device)
    beta = torch.Tensor([[2.54478965]]).to(device)
    return v_L + (v_0 - v_L) * torch.exp(gamma * (SoC - 1)) + alpha * v_L * (SoC - 1) \
           + (1 - alpha) * v_L * (torch.exp(-beta) - torch.exp(-beta * torch.sqrt(SoC)))


saved_network_degraded = "./Battery_Data/degraded_battery_cycles/Battery_RNN_from_new_vpf_learn_Ecrit_v1"
parts = 4
N = 1000
mc_samples = 10000
estimation_stop, cut_off_voltage = 400, 3.1
saved = "{0}_part_{1}.mdl".format(saved_network_degraded, parts)
Z_pol = Polarising_Impedance_Map(E_crit=True)
Z_pol.load_state_dict(torch.load(saved))
Z_pol.to(device)

soc_map = np.array(np.linspace(0, 1.0, 1000), ndmin=2).T

# matplotlib.rcParams['figure.figsize'] = (30.0, 10.0)
font = {"figure.titlesize": 20,
        "figure.titleweight": 'normal',
        "axes.titlesize" : 20,
        "axes.labelsize" : 20,
        "lines.linewidth" : 3,
        "lines.markersize" : 10,
        "xtick.labelsize" : 16,
        "ytick.labelsize" : 16,
        'axes.labelweight': 'bold',
        'legend.fontsize': 20.0,}
for key in font:
    matplotlib.rcParams[key] = font[key]



# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_title("Polarising Impedance Surface [Degraded Battery]")
# Make data.
Current_map = np.array(np.linspace(0.001, Current_max, 10000), ndmin=2).T
Z_surf = []
V_surf = []
P_surf = []
V_bounded_surf = []
P_bounded_surf = []
I_bounded_surf = []
X, Y = np.meshgrid(soc_map, Current_map)
for i in range(Current_map.shape[0]):
    with torch.no_grad():
        marginal_current = np.ones(soc_map.shape)*Current_map[i]
        Z = Z_pol(soc_map, marginal_current)
        Z_surf.append(Z.cpu().numpy())
        V = VoC(torch.from_numpy(soc_map).to(device)) - torch.from_numpy(marginal_current).to(device)*Z
        V_surf.append(V.cpu().numpy())
        P = torch.from_numpy(marginal_current).to(device)*V
        P_surf.append(P.cpu().numpy())
        V_bounds = V <= cut_off_voltage
        V[V_bounds] = 0.0
        V_bounded_surf.append(V.cpu().numpy())
        P = torch.from_numpy(marginal_current).to(device)*V
        P_bounded_surf.append(P.cpu().numpy())
        marginal_current[V_bounds.cpu().numpy()] = 0.0
        I_bounded_surf.append(marginal_current)

Z = np.concatenate(Z_surf, axis=1).T
V_surf = np.concatenate(V_surf, axis=1).T
P_surf = np.concatenate(P_surf, axis=1).T
V_bounded_surf = np.concatenate(V_bounded_surf, axis=1).T
P_bounded_surf = np.concatenate(P_bounded_surf, axis=1).T

SoMPA_degraded = np.max(P_bounded_surf, 0)
I_SoMPA_degraded = np.max(I_bounded_surf, 0)

# plt.figure()
# plt.subplot(211)
# plt.title("Degraded SoC v SoMPA")
# plt.plot(soc_map, np.max(P_bounded_surf, 0), 'b')
# plt.gca().invert_xaxis()
#
# plt.subplot(212)
# plt.title("Degraded SoC v I_max(SoMPA)")
# plt.plot(soc_map, np.max(I_bounded_surf, 0), 'r')
# plt.gca().invert_xaxis()

# Plot the surface.
# surf = ax.plot_surface(X, Y, P_bounded_surf, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.xlabel("\nSoC")
# # plt.xlim(1.1, 0.0)  # decreasing soc
# plt.ylabel("\nCurrent")
# # plt.ylim(0.1, 7.5)  # decreasing soc
# ax.set_zlabel('\nSoMPA')

saved_network_degraded = "./Battery_Data/new_battery_cycles/Battery_RNN_from_prior_v4"
parts = 10
N = 1000
mc_samples = 10000
estimation_stop, cut_off_voltage = 400, 3.1
saved = "{0}_part_{1}.mdl".format(saved_network_degraded, parts)
Z_pol = Polarising_Impedance_Map(E_crit=False)
Z_pol.load_state_dict(torch.load(saved))
Z_pol.to(device)

soc_map = np.array(np.linspace(0, 1.0, 1000), ndmin=2).T

# matplotlib.rcParams['figure.figsize'] = (30.0, 10.0)
# font = {"figure.titlesize": 20,
#         "figure.titleweight": 'normal',
#         "axes.titlesize" : 20,
#         "axes.labelsize" : 20,
#         "lines.linewidth" : 3,
#         "lines.markersize" : 10,
#         "xtick.labelsize" : 16,
#         "ytick.labelsize" : 16,
#         'axes.labelweight': 'bold',
#         'legend.fontsize': 20.0,}
# for key in font:
#     matplotlib.rcParams[key] = font[key]



# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_title("Polarising Impedance Surface [Degraded Battery]")
# Make data.
Current_map = np.array(np.linspace(0.001, Current_max, 10000), ndmin=2).T
Z_surf = []
V_surf = []
P_surf = []
V_bounded_surf = []
P_bounded_surf = []
I_bounded_surf = []
X, Y = np.meshgrid(soc_map, Current_map)
for i in range(Current_map.shape[0]):
    with torch.no_grad():
        marginal_current = np.ones(soc_map.shape)*Current_map[i]
        Z = Z_pol(soc_map, marginal_current)
        Z_surf.append(Z.cpu().numpy())
        V = VoC(torch.from_numpy(soc_map).to(device)) - torch.from_numpy(marginal_current).to(device)*Z
        V_surf.append(V.cpu().numpy())
        P = torch.from_numpy(marginal_current).to(device)*V
        P_surf.append(P.cpu().numpy())
        V_bounds = V <= cut_off_voltage
        V[V_bounds] = 0.0
        V_bounded_surf.append(V.cpu().numpy())
        P = torch.from_numpy(marginal_current).to(device)*V
        P_bounded_surf.append(P.cpu().numpy())
        marginal_current[V_bounds.cpu().numpy()] = 0.0
        I_bounded_surf.append(marginal_current)

Z = np.concatenate(Z_surf, axis=1).T
V_surf = np.concatenate(V_surf, axis=1).T
P_surf = np.concatenate(P_surf, axis=1).T
V_bounded_surf = np.concatenate(V_bounded_surf, axis=1).T
P_bounded_surf = np.concatenate(P_bounded_surf, axis=1).T

SoMPA_new = np.max(P_bounded_surf, 0)
I_SoMPA_new = np.max(I_bounded_surf, 0)

plt.figure()
plt.subplot(211)
# plt.title("SoC v SoMPA")
plt.plot(soc_map, SoMPA_new, 'g', label="$SoMPA_{new}$")
plt.plot(soc_map, SoMPA_degraded, 'c', label="$SoMPA_{degraded}$")
plt.gca().invert_xaxis()
plt.xlabel("State of Charge [SoC]")
plt.ylabel("State of Maximum Power [SoMPA]")
plt.legend()

plt.subplot(212)
# plt.title("SoC v I_max(SoMPA)")
plt.plot(soc_map, I_SoMPA_new, 'r', label="$I_{max}(SoMPA_{new})$")
plt.plot(soc_map, I_SoMPA_degraded, 'purple', label="$I_{max}(SoMPA_{degraded})$")
plt.gca().invert_xaxis()
plt.xlabel("State of Charge [SoC]")
plt.ylabel("Maximum Current @ SoMPA")
plt.legend()

# Plot the surface.
# surf = ax.plot_surface(X, Y, P_bounded_surf, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# # ax.zaxis.set_major_locator(LinearLocator(10))
# # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.xlabel("\nSoC")
# # plt.xlim(1.1, 0.0)  # decreasing soc
# plt.ylabel("\nCurrent")
# # plt.ylim(0.1, 7.5)  # decreasing soc
# ax.set_zlabel('\nSoMPA')


plt.show()
