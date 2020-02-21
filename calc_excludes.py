import mne
import numpy as np

spacing = "ico5"
thresh = 0.25
proc_dir = "../proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29", "ATT_31",
         "ATT_33", "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

sens = []
for sub in subjs:
    sens.append(mne.read_source_estimate(
      "{dir}nc_{sub}_{sp}_sens".format(dir=proc_dir,sub=sub,sp=spacing)))

sen = sens[0].copy()
for s in sens[1:]:
    sen.data += s.data
sen.data /= len(sens)
sen.data[sen.data<thresh] = 0
sen.data[sen.data>=thresh] = 1
exclude = np.where(sen.data==0)[0]
np.save("{}fsaverage_exclude.npy".format(proc_dir),exclude)
