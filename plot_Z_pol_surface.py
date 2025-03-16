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

save_network = "./Battery_Data/new_battery_cycles/Battery_RNN_GT_new_v1.mdl"


def apply_scale(X, X_min, X_max):
    """

    :param X:
    :return: Normalised array like X, mean, std
    """
    return (X - X_min)/(X_max - X_min)


Current_min = -0.000520787
Current_max = 6.74372


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
Z_pol = Polarising_Impedance_Map()
Z_pol.load_state_dict(torch.load(save_network))
Z_pol.to(device)
soc_map = np.array(np.linspace(0, 1.0, 1000), ndmin=2)

Z_map = []
plt.figure()
for i in range(1, 16):
    with torch.no_grad():
        Z = Z_pol(soc_map.T, np.ones((1000, 1))*float(i/2.0))
        Z_map.append(Z.cpu().numpy())
#         plt.subplot(2, 1, 1)
        plt.plot(soc_map.T, Z.cpu().numpy(), label="I = {}".format(i/2.0))
#         plt.subplot(2, 1, 2)
#         plt.semilogy(soc_map.T, Z.cpu().numpy(), label="I = {}".format(i))
# plt.axis([-0.1, 1.1, 0.0, 0.1])
plt.legend()

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = soc_map
Y = np.linspace(0.1, 7.5, 100)
Z_surf = []
for i in range(Y.shape[0]):
    with torch.no_grad():
        Z = Z_pol(soc_map.T, np.ones((1000, 1))*Y[i])
        Z_surf.append(Z.cpu().numpy())
X, Y = np.meshgrid(X, Y)
Z = np.concatenate(Z_surf, axis=1).T
print(Z.shape)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.xlabel("SoC")
plt.ylabel("Current")

plt.show()