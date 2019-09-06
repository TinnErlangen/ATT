import mne
from mayavi import mlab
import pickle
import scipy.sparse
from scipy import stats
import numpy as np
from mne.stats import permutation_cluster_1samp_test,summarize_clusters_stc

mri_key = {"ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18","NAG_83":"ATT_36",
           "PAG48":"ATT_21","SAG13":"ATT_20"}
sub_key = {v: k for k,v in mri_key.items()}

subjs = ["ATT_19"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
#all_conds = [["audio","rest"],["visual","rest"],["audio","visual"],["visselten","audio"],["zaehlen","rest"]]
all_conds = [["visselten","audio"]]
threshold = 0.99
avg_clim = {"kind":"percent","pos_lims":[30,75,100]}

for sub in subjs:
    src = mne.read_source_spaces(proc_dir+sub+"-src.fif")
    filename = proc_dir+"stcs/"+sub
    for conds in all_conds:
        stc_a = mne.read_source_estimate("{dir}stcs/nc_{a}_{b}_mean-lh.stc".format(dir=proc_dir,a=sub,b=conds[0]))
        stc_b = mne.read_source_estimate("{dir}stcs/nc_{a}_{b}_mean-lh.stc".format(dir=proc_dir,a=sub,b=conds[1]))
        stc_c = stc_a - stc_b
        favg = mlab.figure()
        stc_c.plot(sub_key[sub],hemi="both",figure=favg,clim=avg_clim)
        cnx = mne.spatial_src_connectivity(src)
        X = [np.load("{dir}stcs/nc_{a}_{b}_stc.npy".format(dir=proc_dir,a=sub,b=conds[0])),
        np.load("{dir}stcs/nc_{a}_{b}_stc.npy".format(dir=proc_dir,a=sub,b=conds[1]))]
        X = [x.transpose(0,2,1) for x in X]
        p_threshold = 0.0001
        f_threshold = stats.distributions.f.ppf(1. - p_threshold / 2.,
                                            X[0].shape[0]-1, X[1].shape[0]-1)
        f_obs, clusters, cluster_pv, H0 = clu = mne.stats.spatio_temporal_cluster_test(
                X,connectivity=cnx,n_jobs=8,threshold=f_threshold)
        with open(proc_dir+"stcs/"+sub+"_clust","wb") as f:
            pickle.dump(clu,f)

        stc_clu = mne.stats.summarize_clusters_stc(clu,subject=sub,vertices=stc_c.vertices)
        fclu = mlab.figure()
        stc_clu.plot(sub_key[sub],hemi="both",clim=dict(kind='value', lims=[0, 1, 40]), time_viewer=True,figure=fclu)
