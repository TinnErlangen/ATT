import mne
import matplotlib.pyplot as plt
plt.ion()
from scipy.spatial import distance_matrix
import numpy as np

proc_dir = "../proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29", "ATT_29",
         "ATT_30", "ATT_31", "ATT_32", "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
#subjs = ["ATT_10", "ATT_11", "ATT_12"]
runs = ["rest","audio","visselten","visual","zaehlen"]

pos = np.zeros((len(subjs),len(runs),3))
dist_mat = np.zeros((len(subjs),len(runs),len(runs)))
for sub_idx,sub in enumerate(subjs):
    for run_idx,run in enumerate(runs):
        epo_name = "{dir}nc_{sub}_{run}_hand-epo.fif".format(dir=proc_dir, sub=sub, run=run)
        epo = mne.read_epochs(epo_name)
        dev_head_t = epo.info["dev_head_t"]
        pos[sub_idx,run_idx] = mne.transforms.apply_trans(dev_head_t,
                                                          np.array([0,0,0]))
    dist_mat[sub_idx,] = distance_matrix(pos[sub_idx,],pos[sub_idx,])
    if np.max(dist_mat>0.005):
        print("Warning: Subject {} run {} produced a distance of more than 5mm".
              format(sub,run))
