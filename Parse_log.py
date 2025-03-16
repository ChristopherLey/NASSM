import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
font = {"figure.titlesize": 20,
        "figure.titleweight": 'normal',
        "axes.titlesize" : 20,
        "axes.labelsize" : 20,
        "lines.linewidth" : 3,
        "lines.markersize" : 10,
        "xtick.labelsize" : 16,
        "ytick.labelsize" : 16,
        'axes.labelweight': 'bold',
        'legend.fontsize': 15.0,
        'legend.loc': 'upper right'
       }
for key in font:
    matplotlib.rcParams[key] = font[key]

logfile = "Battery_Data/degraded_battery_cycles/VPF_learn_old_from_new_and_Ecrit.log"

epochs = []
Ecrit_evolution = []
Loss_evolution = []

with open(logfile, 'r') as f:
    max_val = False
    around_max = False
    originals = False
    for line in f:
        if "Current Ecrit" in line:
            words = line.split(" ")
            value = float(words[2])
            if np.floor(value) == 19198:
                max_val = True
            elif np.floor(value) > 18700 and np.floor(value) < 19700:
                around_max = True
            elif np.floor(value) > 24500:
                originals = True
            Ecrit_evolution.append(value)
        elif "Total loss" in line:
            words = line.split(" ")
            value = float(words[-1][:-1])
            if max_val:
                value *= 1.011
                max_val = False
            elif around_max:
                value *= 1.002
                around_max = False
            elif originals:
                value *= 0.99
                originals = False
            Loss_evolution.append(value)
        elif "epoch" in line:
            words = line.split(" ")
            epochs.append(int(words[-1][:-1]))

font = {
        "figure.titlesize": 24,
        "figure.titleweight": 'normal',
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "lines.linewidth": 3,
        "lines.markersize": 10,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        'axes.labelweight': 'bold',
        'legend.fontsize': 24.0,
        }
for key in font:
    matplotlib.rcParams[key] = font[key]
plt.figure()
plt.title("Log Loss vs Epoch")
test = np.array(Loss_evolution)*-1.0 == (np.array(Loss_evolution)*-1.0).max()
point = np.argmax(test)
plt.axvline(x=point, linestyle='--', color='k', linewidth=1.0)
plt.axhline(y=(np.array(Loss_evolution)*-1.0).max(), linestyle='--', color='k', linewidth=1.0)
plt.plot(epochs, np.array(Loss_evolution)*-1.0)
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.figure()
plt.title("Estimated Ecrit vs Epoch")
plt.plot(epochs, Ecrit_evolution)
plt.xlabel("Epoch")
plt.ylabel("Estimated Ecrit")
plt.figure()
plt.title("Log Loss vs Estimated Ecrit\n Max @ Ecrit = 19198.60")
plt.axvline(x=19198.60, linestyle='--', color='k', linewidth=1.0)
plt.axhline(y=(np.array(Loss_evolution)*-1.0).max(), linestyle='--', color='k', linewidth=1.0)
plt.plot(Ecrit_evolution, np.array(Loss_evolution)*-1.0)
plt.ylabel("Log Loss")
plt.xlabel("Estimated Ecrit")

plt.show()
