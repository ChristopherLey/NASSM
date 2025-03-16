import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ParticleFilter.Tools import resample
import time
execution_metrics = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


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

Characterisation_Set = pickle.load(open("Battery_Data/new_battery_cycles/Characterisation_Set_Complete.p", 'rb'))

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

class RNNetwork(nn.Module):
    def __init__(self):
        super(RNNetwork, self).__init__()
        self.Z_hl1 = nn.Linear(2, 1024)
        self.Z_hl2 = nn.Linear(1024, 512)
        self.Z_p = nn.Linear(512, 1)
        self.E_crit = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        # SMC params
        self.f_mean = torch.Tensor([0.0])
        self.f_std = torch.Tensor([0.001])
        self.g_std = 0.01
        self.nu = torch.Tensor([1.0 / (self.g_std * np.sqrt(2 * np.pi))])
        self.voltage_expected_hist = None
        self.soc_expected_hist = None
        self.last_soc = None

    def VoC(self, SoC):
        v_L = torch.Tensor([[-1.59614486]]).to(device)
        v_0 = torch.Tensor([[4.13646328]]).to(device)
        gamma = torch.Tensor([[0.63726463]]).to(device)
        alpha = torch.Tensor([[1.40174122]]).to(device)
        beta = torch.Tensor([[2.54478965]]).to(device)
        return v_L + (v_0 - v_L) * torch.exp(gamma * (SoC - 1)) + alpha * v_L * (SoC - 1) \
               + (1 - alpha) * v_L * (torch.exp(-beta) - torch.exp(-beta * torch.sqrt(SoC)))

    def forward(self, soc_init, current, voltage_measured,  estimation_stop=None):
        first = True
        set_size = current.shape[1]
        if estimation_stop is not None and estimation_stop <= set_size:
            set_size = estimation_stop
        voltage = torch.empty((soc_init.shape[0], set_size), dtype=torch.float)
        soc_hist = torch.empty((soc_init.shape[0], set_size), dtype=torch.float)
        self.w_hist = torch.empty((soc_init.shape[0], set_size), dtype=torch.float)
        self.voltage_expected_hist = torch.empty((1, set_size), dtype=torch.float)
        self.soc_expected_hist = torch.empty((1, set_size), dtype=torch.float)
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
        for t in range(set_size):
            start_time = time.time()
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
            exec_speed = time.time() - start_time
            print(f'Inference execution speed: {exec_speed}')
            execution_metrics.append(exec_speed)

            self.w_hist[:, t] = W[:, 0]
            voltage[:, t] = V[:, 0]
            soc_hist[:, t] = soc[:, 0]
            self.voltage_expected_hist[0, t] = V.transpose(0, 1).mm(W.to(device))
            self.soc_expected_hist[0, t] = soc.transpose(0, 1).mm(W.to(device))

            self.last_soc = soc.transpose(0, 1).mm(W.to(device))

        return loss, voltage, soc_hist

    def SoMPA(self, soc_init, current, voltage_measured, estimation_stop, cut_off_voltage, mc_samples=10000):
        loss, voltage, soc_hist = self.forward(soc_init, current, voltage_measured, estimation_stop=estimation_stop)
        set_size = current.shape[1] - soc_hist.shape[1]
        N = mc_samples
        soc = torch.ones((N, 1), dtype=torch.float).to(device)*self.last_soc
        voltage_prediction = torch.empty((soc.shape[0], set_size), dtype=torch.float)
        soc_prediction = torch.empty((soc.shape[0], set_size), dtype=torch.float)

        I = torch.ones(N, 1) * current[0, estimation_stop-1]
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
        start_time = time.time()
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
            scaled_I = apply_scale(I, Current_min, Current_max)
            # Estimate Z_p
            combined = torch.cat((soc, scaled_I), 1)
            Z = torch.sigmoid(self.Z_hl1(combined))
            Z = torch.sigmoid(self.Z_hl2(Z))
            Z = self.Z_p(Z)

            # Estimate posterior V
            V = self.VoC(soc) - I * Z
            voltage_prediction[:, t] = V[:, 0]
            soc_prediction[:, t] = soc[:, 0]
        print(f'Prognosis execution time:{time.time()-start_time}')
        # Generate SoMPA KDE
        from sklearn.neighbors import KernelDensity
        test_V = voltage_prediction.numpy().T < cut_off_voltage
        first_past_threshold = np.argmax(test_V, axis=0)[:, np.newaxis]
        min_test = first_past_threshold[:, 0] == 0.0
        first_past_threshold[min_test, 0] = first_past_threshold.max()
        first_past_threshold += estimation_stop
        std_samples = np.std(first_past_threshold)
        SoMPA_base = np.arange(0, current.shape[1])[:, np.newaxis]
        log_dens = KernelDensity(kernel='gaussian', bandwidth=1.06*std_samples*np.power(mc_samples, -1/5.0)
                                 ).fit(first_past_threshold).score_samples(SoMPA_base)
        SoMPA_pdf = np.exp(log_dens)

        return loss, voltage, soc_hist, voltage_prediction, soc_prediction, SoMPA_pdf, first_past_threshold