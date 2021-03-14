import mne
from mayavi import mlab
import pickle
import scipy.sparse
from scipy import stats
import numpy as np
from mne.stats import f_mway_rm,summarize_clusters_stc,f_threshold_mway_rm
import matplotlib.pyplot as plt
plt.ion()

mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "DEN59":"ATT_26","WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31",
           "Mun79":"ATT_35","ATT_37_fsaverage":"ATT_37",
           "ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}
sub_key = {v: k for k,v in mri_key.items()}
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31",
         "ATT_33", "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

subjects_dir = "/home/jeff/hdd/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = "ico5"
n_jobs=16
f_ranges = [[3,7],[8,12],[13,15],[15,24],[24,48]]
f_ranges = [[3,7],[13,15],[15,24],[24,48]]
f_ranges = [[7,14]]
conds = ["visselten","visual"]
perm_num = 512
return_pvals=False
p = 0.05
threshold = stats.distributions.t.ppf(1-p, len(subjs)-1)
threshold = dict(start=0, step=0.2)
cond_str = conds[0]
for c in conds[1:]:
    cond_str += "_" + c
thresh_str = "tfce" if isinstance(threshold,dict) else p

# get connectivity
fs_src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,"fsaverage",                                                         spacing))
cnx = mne.spatial_src_connectivity(fs_src)
del fs_src
exclude = np.load("{}fsaverage_exclude.npy".format(proc_dir))

for fr in f_ranges:
    X = [[] for cond in conds]
    for sub_idx,sub in enumerate(subjs):
        src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,sub,spacing))
        vertnos=[s["vertno"] for s in src]
        morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                         subject_to="fsaverage",
                                         spacing=5,
                                         subjects_dir=subjects_dir,
                                         smooth=None)

        idx = 0
        for cond_idx,cond in enumerate(conds):
            X_temp = []
            stc_temp = mne.read_source_estimate(
                    "{dir}stcs/nc_{a}_{b}_{f0}-{f1}Hz_{d}-lh.stc".format(
                      dir=proc_dir,a=sub,b=cond,f0=fr[0],f1=fr[1],
                      d=spacing))
            stc_temp = morph.apply(stc_temp)
            X_temp.append(stc_temp.data.transpose(1,0))
            X[idx].append(np.vstack(X_temp))
            idx += 1
    X = [(np.array(x)*1e+26).astype(np.float32) for x in X]
    del X_temp, morph, src
    #XX = (X[1] - X[2]) / (X[0] + 1e-8)
    XX = X[0] - X[1]

    t_obs, clusters, cluster_pv, H0 = clu = \
      mne.stats.spatio_temporal_cluster_1samp_test(XX,connectivity=cnx,n_jobs=n_jobs,
                                                   threshold=threshold,
                                                   n_permutations=perm_num,
                                                   spatial_exclude=exclude)
    raw_t = stc_temp.copy()
    sign = np.sign(XX[:,:].mean(axis=0))
    raw_t.data = (t_obs * sign).T
    raw_t.save("{}rawstat_{}-{}Hz_{}".format(proc_dir, fr[0], fr[1], cond_str))
    with open("{}clu_{}-{}Hz_{}_{}".format(proc_dir, fr[0], fr[1], cond_str, thresh_str),"wb") as f:
        pickle.dump(clu,f)
    try:
        stc_clu = mne.stats.summarize_clusters_stc(clu,subject="fsaverage",p_thresh=0.05)
        stc_clu.save("{}stc_clu_{}-{}Hz_{}_{}".format(proc_dir, fr[0], fr[1],
                                                         cond_str,thresh_str))
    except:
        print("No significant results")
