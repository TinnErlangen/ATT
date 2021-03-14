import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import pickle
import mne
from mayavi import mlab

indep_var = "Angenehm"
band = "alpha_0"
proc_dir = "/home/jeff/ATT_dat/proc/"
threshold = 0.05
perm_num = 200

with open("{}dics_{}_main_result".format(proc_dir, band), "rb") as f:
    main_result = pickle.load(f)
perms = np.load("{}dics_{}_perm_{}.npy".format(proc_dir, band, perm_num))
perm_thresh = np.quantile(perms.max(axis=1),1-threshold)
cs_main = main_result["cluster_stats"]
cs_perm = perms.flatten()

stc = mne.read_source_estimate("{}fsaverage_blank-lh.stc".format(proc_dir))

stc_clust = stc.copy()
stc_clust.data[:,0] = cs_main
stc_clust.data[stc_clust.data<perm_thresh] = 0
stc_clust_min = stc_clust.data[stc_clust.data>0].min()
stc_clust_mid = stc_clust_min + (stc_clust.data.max()-stc_clust_min)/2

stc_coef = stc.copy()
stc_coef.data[:,0] = main_result["coeffs"]
stc_coef.data[stc_clust.data<perm_thresh] = 0
stc_coef_min = abs(stc_coef.data[abs(stc_coef.data)>0].min())
stc_coef_mid = stc_coef_min + (abs(stc_coef.data).max()-stc_coef_min)/2

stc_clust.plot(hemi="both", subject="fsaverage", clim={"kind":"values",
                                                 "lims":[stc_clust_min,
                                                 stc_clust_mid,
                                                 stc_clust.data.max()]})
stc_coef.plot(hemi="both", subject="fsaverage", clim={"kind":"values",
                                                 "pos_lims":[stc_coef_min,
                                                 stc_coef_mid,
                                                 abs(stc_coef.data).max()]})
