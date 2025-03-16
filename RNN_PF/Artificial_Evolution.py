import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ParticleFilter.Tools import resample

Characterisation_Set = pickle.load(open("Battery_Data/new_battery_cycles/Characterisation_Set_Complete.p", 'rb'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class AE_PF(nn.Module):
    def __init__(self):
        super(AE_PF, self).__init__()
        self.E_crit = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        # SMC params
        self.f_mean = torch.Tensor([0.0])
        self.f_std = torch.Tensor([0.001])
        self.R_std = torch.Tensor([0.01])
        self.R_init = torch.Tensor([0.08076263685971334])
        self.g_std = 0.01
        self.nu = torch.Tensor([1.0 / (self.g_std * np.sqrt(2 * np.pi))])
        self.voltage_expected_hist = None
        self.soc_expected_hist = None
        self.last_soc = None
        self.last_R = None
        self.R_expected_hist = None
        self.R_hist = None

    def VoC(self, SoC):
        v_L = torch.Tensor([[-1.59614486]]).to(device)
        v_0 = torch.Tensor([[4.13646328]]).to(device)
        gamma = torch.Tensor([[0.63726463]]).to(device)
        alpha = torch.Tensor([[1.40174122]]).to(device)
        beta = torch.Tensor([[2.54478965]]).to(device)
        return v_L + (v_0 - v_L) * torch.exp(gamma * (SoC - 1)) + alpha * v_L * (SoC - 1) \
               + (1 - alpha) * v_L * (torch.exp(-beta) - torch.exp(-beta * torch.sqrt(SoC)))

    def forward(self, soc_init, current, voltage_measured,  estimation_stop=None, R_init=None):
        first = True
        set_size = current.shape[1]
        if estimation_stop is not None and estimation_stop <= set_size:
            set_size = estimation_stop
        voltage = torch.empty((soc_init.shape[0], set_size), dtype=torch.float)
        soc_hist = torch.empty((soc_init.shape[0], set_size), dtype=torch.float)
        self.w_hist = torch.empty((soc_init.shape[0], set_size), dtype=torch.float)
        self.voltage_expected_hist = torch.empty((1, set_size), dtype=torch.float)
        self.soc_expected_hist = torch.empty((1, set_size), dtype=torch.float)
        self.R_expected_hist = torch.empty((1, set_size), dtype=torch.float)
        self.R_hist = torch.empty((soc_init.shape[0], set_size), dtype=torch.float)
        soc = soc_init.to(device, torch.float)
        if R_init is None:
            R = torch.ones_like(soc_init)*self.R_init
        else:
            R = torch.ones_like(soc_init) * R_init
        R = R.to(device, torch.float)
        N = soc_init.shape[0]
        I = torch.ones(N, 1) * current[0, 0]
        I = I.to(device, torch.float)

        for t in range(set_size):

            # Estimate posterior V
            R_std = self.R_std * np.exp(-t / 100)
            R = R + torch.normal(torch.zeros([N, 1]), torch.ones([N, 1]) * R_std).to(device)
            V = self.VoC(soc) - I * R

            # Predict SoC
            soc = soc - I*V/E_crit_new*self.E_crit
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

            # Estimate posterior V
            V = self.VoC(soc) - I*R

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

            state = torch.cat((soc, R, V), 1)
            # Resampling
            state, W = resample(state, loss_W)
            soc = state[:, 0:1]
            R = state[:, 1:2]
            V = state[:, 2:3]

            self.w_hist[:, t] = W[:, 0]
            self.R_hist[:, t] = R[:, 0]
            voltage[:, t] = V[:, 0]
            soc_hist[:, t] = soc[:, 0]
            self.voltage_expected_hist[0, t] = V.transpose(0, 1).mm(W.to(device))
            self.soc_expected_hist[0, t] = soc.transpose(0, 1).mm(W.to(device))
            self.R_expected_hist[0, t] = R.transpose(0, 1).mm(W.to(device))

        self.last_soc = soc.transpose(0, 1).mm(W.to(device))
        self.last_R = R.transpose(0, 1).mm(W.to(device))

        return loss, voltage, soc_hist

    def SoMPA(self, soc_init, current, voltage_measured, estimation_stop, cut_off_voltage, mc_samples=10000):
        loss, voltage, soc_hist = self.forward(soc_init, current, voltage_measured, estimation_stop=estimation_stop)
        current = np.concatenate((current, current[0:1, 0:200]), axis=1)
        set_size = current.shape[1] - soc_hist.shape[1]
        N = mc_samples
        soc = torch.ones((N, 1), dtype=torch.float).to(device)*self.last_soc
        voltage_prediction = torch.empty((soc.shape[0], set_size), dtype=torch.float)
        soc_prediction = torch.empty((soc.shape[0], set_size), dtype=torch.float)

        I = torch.ones(N, 1) * current[0, estimation_stop-1]
        I = I.to(device, torch.float)

        V = self.VoC(soc) - I * self.last_R
        for t in range(0, set_size):
            i = soc_hist.shape[1] + t
            # Predict SoC
            soc = soc - I * V / E_crit_new * self.E_crit
            # Add model uncertainty
            soc = soc + torch.normal(torch.ones([N, 1]) * self.f_mean, torch.ones([N, 1]) * self.f_std).to(device)

            # Bounds
            max_test = soc[:, 0] > 1.0
            soc[max_test, 0] = 1.0
            min_test = soc[:, 0] < 0.0
            soc[min_test, 0] = 0.0000000001

            # Posterior Evidence
            I = torch.ones(N, 1) * current[0, i]
            I = I.to(device, torch.float)

            # Estimate posterior V
            V = self.VoC(soc) - I * self.last_R
            voltage_prediction[:, t] = V[:, 0]
            soc_prediction[:, t] = soc[:, 0]
        # Generate SoMPA KDE
        from sklearn.neighbors import KernelDensity
        test_V = voltage_prediction.numpy().T < cut_off_voltage
        first_past_threshold = np.argmax(test_V, axis=0)[:, np.newaxis]
        min_test = first_past_threshold[:, 0] == 0.0
        first_past_threshold[min_test, 0] = first_past_threshold.max()
        first_past_threshold += estimation_stop
        std_samples = np.std(first_past_threshold)
        SoMPA_base = np.arange(0, current.shape[1]+200)[:, np.newaxis]
        log_dens = KernelDensity(kernel='gaussian', bandwidth=1.06*std_samples*np.power(mc_samples, -1/5.0)
                                 ).fit(first_past_threshold).score_samples(SoMPA_base)
        SoMPA_pdf = np.exp(log_dens)

        return loss, voltage, soc_hist, voltage_prediction, soc_prediction, SoMPA_pdf, first_past_threshold, current[0, soc_hist.shape[1]:]