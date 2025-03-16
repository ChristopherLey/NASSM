# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import pickle
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

Ecrit_changed = False

new_battery_Z = "./Battery_Data/new_battery_cycles/Battery_RNN_from_prior_v4_part_10.mdl"
if Ecrit_changed:
    degraded_battery_Z = "./Battery_Data/degraded_battery_cycles/Battery_RNN_from_new_vpf_change_Ecrit_v2_part_10.mdl"
else:
    degraded_battery_Z = "./Battery_Data/degraded_battery_cycles/Battery_RNN_from_new_vpf_v1_part_7.mdl"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def apply_scale(X, X_min, X_max):
    """

    :param X:
    :return: Normalised array like X, mean, std
    """
    return (X - X_min)/(X_max - X_min)


Current_min = -0.000520787
Current_max = 6.74372

E_crit_new = 26267.160775850585
E_crit_old = 21879.133773481735


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Z_pol_new = Polarising_Impedance_Map()
Z_pol_new.load_state_dict(torch.load(new_battery_Z))
Z_pol_new.to(device)
Z_pol_old = Polarising_Impedance_Map()
Z_pol_old.load_state_dict(torch.load(degraded_battery_Z))
Z_pol_old.to(device)
soc_map = np.array(np.linspace(0, 1.0, 1000), ndmin=2)

Z_map = []
plt.figure()
plt.title("New Level Curves")
for i in range(1, 16):
    with torch.no_grad():
        Z = Z_pol_new(soc_map.T, np.ones((1000, 1))*float(i/2.0))
        Z_map.append(Z.cpu().numpy())
        plt.plot(soc_map.T, Z.cpu().numpy(), label="I = {}".format(i/2.0))
plt.legend()

Z_map = []
plt.figure()
plt.title("Degraded Level Curves")
for i in range(1, 16):
    with torch.no_grad():
        Z = Z_pol_old(soc_map.T, np.ones((1000, 1))*float(i/2.0))
        Z_map.append(Z.cpu().numpy())
        plt.plot(soc_map.T, Z.cpu().numpy(), label="I = {}".format(i/2.0))
plt.legend()

Z_map = []
plt.figure()
plt.title("Delta Level Curves")
for i in range(1, 16):
    with torch.no_grad():
        Z_new = Z_pol_new(soc_map.T, np.ones((1000, 1))*float(i/2.0))
        Z_old = Z_pol_old(soc_map.T, np.ones((1000, 1)) * float(i / 2.0))
        Z = torch.abs(Z_old - Z_new)
        Z_map.append(Z.cpu().numpy())
        plt.plot(soc_map.T, Z.cpu().numpy(), label="I = {}".format(i/2.0))
plt.legend()

Z_map = []
plt.figure()
plt.title("proportion change Level Curves")
for i in range(1, 16):
    with torch.no_grad():
        Z_new = Z_pol_new(soc_map.T, np.ones((1000, 1))*float(i/2.0))
        Z_old = Z_pol_old(soc_map.T, np.ones((1000, 1)) * float(i / 2.0))
        Z = Z_old/Z_new
        Z_map.append(Z.cpu().numpy())
        plt.plot(soc_map.T, Z.cpu().numpy(), label="I = {}".format(i/2.0))
plt.legend()
#
# fig = plt.figure("New")
# ax = fig.gca(projection='3d')
# ax.set_title("New Surface")
#
# # Make data.
# X = soc_map
# Y = np.linspace(0.1, 7.5, 100)
# Z_surf = []
# for i in range(Y.shape[0]):
#     with torch.no_grad():
#         Z = Z_pol_new(soc_map.T, np.ones((1000, 1))*Y[i])
#         Z_surf.append(Z.cpu().numpy())
# X, Y = np.meshgrid(X, Y)
# Z = np.concatenate(Z_surf, axis=1).T
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.xlabel("SoC")
# plt.ylabel("Current")
#
# fig = plt.figure("Old")
# ax = fig.gca(projection='3d')
# ax.set_title("Degraded Surface")
#
# # Make data.
# X = soc_map
# Y = np.linspace(0.1, 7.5, 100)
# Z_surf = []
# for i in range(Y.shape[0]):
#     with torch.no_grad():
#         Z = Z_pol_old(soc_map.T, np.ones((1000, 1))*Y[i])
#         Z_surf.append(Z.cpu().numpy())
# X, Y = np.meshgrid(X, Y)
# Z = np.concatenate(Z_surf, axis=1).T
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.xlabel("SoC")
# plt.ylabel("Current")
#
# fig = plt.figure("Delta")
# ax = fig.gca(projection='3d')
# ax.set_title("Delta(new/old) Surface")
#
# # Make data.
# X = soc_map
# Y = np.linspace(0.1, 7.5, 100)
# Z_surf = []
# for i in range(Y.shape[0]):
#     with torch.no_grad():
#         Z_old = Z_pol_old(soc_map.T, np.ones((1000, 1))*Y[i])
#         Z_new = Z_pol_new(soc_map.T, np.ones((1000, 1)) * Y[i])
#         Z = Z_old - Z_new
#         Z_surf.append(Z.cpu().numpy())
# X, Y = np.meshgrid(X, Y)
# Z = np.concatenate(Z_surf, axis=1).T
#
# # Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)
#
# # Customize the z axis.
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=10)
# plt.xlabel("SoC")
# plt.ylabel("Current")

plt.show()