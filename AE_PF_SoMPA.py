import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ParticleFilter.Tools import resample
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
sns.set()

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
            R_std = self.R_std*np.exp(-t/100)
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

        return loss, voltage, soc_hist, voltage_prediction, soc_prediction, SoMPA_pdf, first_past_threshold


vsmc = AE_PF()

Training_Set = pickle.load(open("Battery_Data/degraded_battery_cycles/Test_Degraded_Battery_Set.p", 'rb'))
parts = 4
N = 100
mc_samples = 10000
estimation_stop, cut_off_voltage = 400, 3.1
with torch.no_grad():
    vsmc.to(device)
    title = ["Test Discharge (not used in training)",
             "Random Discharge 1", "Random Discharge 2", "Random Discharge 3"]
    for j, set_dict in enumerate(Training_Set):
        if j == 1:
            test = set_dict['Voltage'] < cut_off_voltage
            gt_cuttoff = np.argmax(test, axis=1)[0, 0]
            state = torch.ones(N, 1) * 1.0
            loss, voltage, soc_hist, voltage_prediction, soc_prediction, \
            SoMPA, pass_threshold = vsmc.SoMPA(state,
                                               set_dict['Current'],
                                               set_dict['Voltage'],
                                               estimation_stop,
                                               cut_off_voltage,
                                               mc_samples=mc_samples)
            seconds = np.ones_like(set_dict['Current'])
            voltage_expected_hist = vsmc.voltage_expected_hist.numpy()
            soc_expected_hist = vsmc.soc_expected_hist.numpy()
            current = np.array(set_dict['Current'])
            data_set = {
                "Model_Details": "SoC discharge with NASSM, with known future discharge profile",
                "terminal_voltage": set_dict['Voltage'].T,
                "measured_current": set_dict['Current'][0, :estimation_stop].T,
                "current_prediction": set_dict['Current'][0, estimation_stop:].T,
                "estimation_stop": estimation_stop,
                "gt_cutoff_at": gt_cuttoff,
                "cut_off_voltage": cut_off_voltage,
                "state_init": np.ones((N, 1)),
                "P(disconnection)": SoMPA,
                "voltage_estimate_particles": voltage,
                "voltage_expectation": voltage_expected_hist.T,
                "voltage_prediction": voltage_prediction.numpy(),
                "SoC_estimate_particles": soc_hist.numpy().T,
                "SoC_expectation": soc_expected_hist.T,
                "SoC_prediction": soc_prediction.numpy(),
                "first_instance_beyond_threshold": pass_threshold,
            }

            matplotlib.rcParams['figure.figsize'] = (20.0, 20.0)
            font = {"figure.titlesize": 20,
                    "figure.titleweight": 'normal',
                    "axes.titlesize": 20,
                    "axes.labelsize": 20,
                    "lines.linewidth": 3,
                    "lines.markersize": 10,
                    "xtick.labelsize": 16,
                    "ytick.labelsize": 16,
                    'axes.labelweight': 'bold',
                    'legend.fontsize': 20.0, }
            for key in font:
                matplotlib.rcParams[key] = font[key]

            # FIGURE 1a:
            title = ["SoC Discharge"]
            fig, ax1 = plt.subplots(nrows=1)
            ax1.set_xlabel("Seconds")
            ax1.axvline(x=data_set["estimation_stop"], color='g', linestyle='--', label="Prognosis Starts", linewidth=2.0)
            color = 'k'
            ax1.set_ylabel("Volts", color=color)
            downsample = np.random.randint(mc_samples, size=100)
            for j in range(downsample.shape[0]):
                i = downsample[j]
                ax1.plot(np.arange(data_set["estimation_stop"], data_set["first_instance_beyond_threshold"][i, 0]),
                         data_set['voltage_prediction'][i, :(data_set["first_instance_beyond_threshold"][i, 0]-data_set["estimation_stop"])], '-c')
            ax1.plot([], '-c', label="Predicted Voltage")
            ax1.plot(data_set["terminal_voltage"][:data_set['gt_cutoff_at']+1],
                     linestyle='-', color=color, label="Terminal Voltage (pre-cutoff)")
            ax1.plot(data_set["voltage_expectation"], '-g', label="Estimated Voltage")
            ax1.axhline(y=data_set["cut_off_voltage"], color='r', linestyle='--', label="Cut-off Voltage", linewidth=2.0)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_ylim(2.5, 4.3)

            # FIGURE 1b:
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'b'
            ax2.set_ylabel('Disconnection probabilty', color=color)  # we already handled the x-label with ax1
            ax2.fill(np.arange(0, data_set["P(disconnection)"].shape[0]), data_set["P(disconnection)"], fc=color, alpha=0.9, label='P(disconnection)')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim([0, data_set["P(disconnection)"].max()*5.0])
            ax2.grid(None)
            fig.legend(bbox_to_anchor=(0.8, 0.80))

            # FIGURE 2:
            fig, ax3 = plt.subplots(nrows=1)
            ax3.set_xlabel("Seconds")
            ax3.plot(np.arange(0, data_set["estimation_stop"]), data_set["measured_current"], '-r', label="Current Profile")
            ax3.plot(np.arange(data_set["estimation_stop"], data_set["terminal_voltage"].shape[0]),
                     data_set['current_prediction'], '-b')
            ax3.plot([], '-b', label="Predicted Profile")
            ax3.axvline(x=data_set["estimation_stop"], color='g', linestyle='--', label="Prognosis Starts", linewidth=2.0)
            ax3.set_ylabel("Current")
            ax3.legend()

            # FIGURE 3:
            fig, ax4 = plt.subplots(nrows=1)
            ax4.set_xlabel("Seconds")
            ax4.plot(data_set['SoC_estimate_particles'], '.b')
            ax4.plot([],'.b', label="E particles")
            ax4.plot(data_set["SoC_expectation"],'-g', label="Estimated Energy")
            downsample = np.random.randint(mc_samples, size=100)
            for j in range(downsample.shape[0]):
                i = downsample[j]
                ax4.plot(np.arange(data_set["estimation_stop"], data_set["first_instance_beyond_threshold"][i, 0]),
                         data_set['SoC_prediction'][i, :(data_set["first_instance_beyond_threshold"][i, 0]-data_set["estimation_stop"])], '.c')
            ax4.plot([], '.c', label="E prognosis")
            ax4.axvline(x=data_set["estimation_stop"], color='g', linestyle='--', label="Prognosis Starts", linewidth=2.0)
            ax4.legend(loc='best',)
            ax4.set_ylabel("SoC")
            ax4.set_ylim([0.0, 1.0])

plt.show()