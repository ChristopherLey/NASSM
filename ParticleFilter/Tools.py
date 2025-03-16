import torch
import numpy as np


def resample(state, W, sampleLimit=0.85):
    length = W.shape[0]
    P = torch.empty(length, 1)
    u = torch.rand(length, 1)
    index = torch.empty_like(state)
    temp_cumsum = 0.0
    W = W/torch.sum(W)
    N_eff = 1.0 / torch.sum(torch.pow(W, 2.0))
    if N_eff < sampleLimit*length:     # Resample
        for j in range(length):
            temp_cumsum += W[j, 0]
            P[j, 0] = temp_cumsum
            u[j, 0] = torch.pow(u[j, 0], (1.0/(length - j)))
        ut = torch.cumprod(u, 0)
        u = torch.flip(ut, (0, 1))
        k = 0
        try:
            for i in range(length):
                while P[k, 0] <= u[i, 0]:
                    k += 1
                    if k > (length - 1):
                        k = length - 1
                        break
                index[i, :] = state[k, :]
                W[i, 0] = 1.0/float(length)
            return index.to(state.device), W
        except IndexError:
            print("IndexError:", P, u)
            return state, W
    else:
        return state, W


def run_test(voltage, current, soc):
    return (
        voltage,
        np.random.normal(voltage, np.ones_like(voltage)*0.01),
        current,
        soc,
        np.random.normal(voltage, np.ones_like(voltage)*0.01))


if __name__ == "__main__":
    soc = torch.normal(1, 0.1, (20, 1))
    R = torch.normal(0, 0.1, (20, 1))
    state = torch.cat([soc, R], 1)
    W = torch.ones((20, 1))*0.001
    W[0:2] = 1.0
    print(f'most likely to be sampled: {state[0:2, :]}, with weights {W[0:2, :]}')
    print(resample(state, W))
