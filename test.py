from ParticleFilter.Tools import resample

if __name__ == "__main__":
    import torch
    import numpy as np
    import time

    for i in range(1000):
        N = 100
        groundTruth = 0.85
        g_std = 0.05
        state = torch.ones(N, 1, requires_grad=True) * groundTruth
        state[:, 0:1] = state[:, 0:1] + torch.normal(torch.zeros([state.shape[0], 1]),
                                                     torch.ones([state.shape[0], 1]) * g_std)
        nu = torch.Tensor([1.0 / (0.2 * np.sqrt(2 * np.pi))])
        logW = torch.log(nu) - 0.5*torch.pow((state - groundTruth)/g_std, 2.0)
        max_logW = logW.max()
        loss_W = torch.exp(logW - max_logW)
        start_time = time.time()
        resampled, W = resample(state, loss_W)
        print("--- %s seconds ---" % (time.time() - start_time))
        state[:, 0:1] = resampled[:, 0:1]
        loss = state.std()
        loss.backward()

