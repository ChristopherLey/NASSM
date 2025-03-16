import torch
import pickle
import numpy as np
import time
from RNN_PF.NASSM import NASSM_new, NASSM_Degraded, device, Characterisation_Set
from RNN_PF.Artificial_Evolution import AE_PF

root_dir = "/home/chris/Dropbox/University/Publications/Where did all my energy go? Informed Maximum Power " \
           "Prediction using a Hybrid Neural Adaptive State Space Model/IEEE Access/Simulation_Results/JITP_sensitivity"
sets = ["new", "degraded"]
# models = ["NASSM", "AE"]
models = ["AE",]
estimation_stop = 400
sensitivity_steps = 100
mc_samples = 10000
cut_off_voltage = 3.1
N = 100

for set in sets:
    if set == "new":
        set_dict = Characterisation_Set['Sets'][2]
        saved_network = "./Battery_Data/new_battery_cycles/Battery_RNN_from_prior_v4"
        parts = 10
        saved = "{0}_part_{1}.mdl".format(saved_network, parts)
        vsmc = NASSM_new()
        nn_state = torch.load(saved)
    else:   # degraded
        set_dict = pickle.load(open("Battery_Data/degraded_battery_cycles/Test_Degraded_Battery_Set.p", 'rb'))[1]
        saved_network = "./Battery_Data/degraded_battery_cycles/Battery_RNN_from_new_vpf_learn_Ecrit_v1"
        parts = 4
        saved = "{0}_part_{1}.mdl".format(saved_network, parts)
        vsmc = NASSM_Degraded()
        nn_state = torch.load(saved)
    test = set_dict['Voltage'].T < cut_off_voltage
    gt_cuttoff = np.argmax(test, axis=0) + 1
    for model in models:
        with torch.no_grad():
            if model == "NASSM":
                vsmc.load_state_dict(nn_state)
            else:   # AE
                vsmc = AE_PF()
            vsmc.to(device)
            start_time = time.time()
            state = torch.ones(N, 1) * 1.0
            for k in range(sensitivity_steps):
                loss, voltage, soc_hist, voltage_prediction, \
                soc_prediction, SoMPA, pass_threshold, current_prediction = vsmc.SoMPA(state,
                                                                                       set_dict['Current'],
                                                                                       set_dict['Voltage'],
                                                                                       estimation_stop,
                                                                                       cut_off_voltage,
                                                                                       mc_samples=mc_samples)
                now = time.time()
                print(f"{k}; Execution Time: {(now - start_time) / 60.0}m, Estimated finish in: "
                      f"{((now - start_time) / (k + 1) * (sensitivity_steps - k)) / 60.0}m")
                save_to = f"{root_dir}/{set}/{model}_jitp_{k}.p"
                print(f"Save to: {save_to}")
                voltage_expected_hist = vsmc.voltage_expected_hist.numpy()
                soc_expected_hist = vsmc.soc_expected_hist.numpy()
                current = np.array(set_dict['Current'])
                save_data = {
                    "P(disconnection)": SoMPA,
                    "first_instance_beyond_threshold": pass_threshold,
                    "voltage_prediction_pdf": np.empty((pass_threshold.shape[0], 1)),
                    "soc_prediction_pdf": np.empty((pass_threshold.shape[0], 1)),
                    "voltage_expectation_end": voltage_expected_hist[0, -1],
                    "soc_expectation_end": soc_expected_hist[0, -1],
                    "gt_cutoff_at": gt_cuttoff[0, 0],
                }

                for i in range(pass_threshold.shape[0]):
                    save_data["voltage_prediction_pdf"][i, 0] = voltage_prediction[i, pass_threshold[i, 0]-estimation_stop]
                    save_data["soc_prediction_pdf"][i, 0] = soc_prediction[i, pass_threshold[i, 0]-estimation_stop]
                with open(save_to, 'wb') as f:
                    pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)
