import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ParticleFilter.Tools import resample
import matplotlib
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt

saved_state = pickle.load(open("state_save_event.p", 'rb'))

plt.figure()
plt.plot()