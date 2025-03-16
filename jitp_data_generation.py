import torch
import pickle
import numpy as np
import time
from RNN_PF.NASSM import NASSM_new, NASSM_Degraded, device, Characterisation_Set
from RNN_PF.Artificial_Evolution import AE_PF

root_dir = "/home/chris/Dropbox/University/Publications/Where did all my energy go? Informed Maximum Power " \
           "Prediction using a Hybrid Neural Adaptive State Space Model/IEEE Access/Simulation_Results/JITP"
sets = ["new", "degraded"]
model = ["NASSM", "AE"]
choose = (1, 1)
start = 400

if choose[0] == 0:       # new
    set_dict = Characterisation_Set['Sets'][2]
    estimation_stop, cut_off_voltage = 400, 3.1
    mc_samples = 10000
    test = set_dict['Voltage'].T < cut_off_voltage
    gt_cuttoff = np.argmax(test, axis=0) + 1
    stop = gt_cuttoff[0, 0]-50
    N = 100
    # stop = 200
    with torch.no_grad():
        if choose[1] == 0:      # NASSM
            saved_network = "./Battery_Data/new_battery_cycles/Battery_RNN_from_prior_v4"
            parts = 10


            vsmc = NASSM_new()
            saved = "{0}_part_{1}.mdl".format(saved_network, parts)
            nn_state = torch.load(saved)
            vsmc.load_state_dict(nn_state)

        elif choose[1] == 1:    # AE
            vsmc = AE_PF()

        vsmc.to(device)
        start_time = time.time()
        for i in range(start, stop):
            state = torch.ones(N, 1) * 1.0
            loss, voltage, soc_hist, voltage_prediction, \
            soc_prediction, SoMPA, pass_threshold, current_prediction = vsmc.SoMPA(state,
                                                                                   set_dict['Current'],
                                                                                   set_dict['Voltage'],
                                                                                   estimation_stop,
                                                                                   cut_off_voltage,
                                                                                   mc_samples=mc_samples)
            save_to = f"{root_dir}/{sets[choose[0]]}/{model[choose[1]]}_jitp_{i}.p"
            with open(save_to, 'wb') as f:
                pickle.dump(SoMPA, f, pickle.HIGHEST_PROTOCOL)
            now = time.time()
            print(f"{i}; Execution Time: {(now - start_time)/60.0}m, Estimated finish in: "
                  f"{((now-start_time)/(i-start+1)*(stop - start - i))/60.0}m")


elif choose[0] == 1:     # degraded
    save_to = f"{root_dir}/{sets[choose[0]]}/{model[choose[1]]}_jitp"