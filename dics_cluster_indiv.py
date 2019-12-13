import mne
from mayavi import mlab
import pickle
import scipy.sparse
from scipy import stats
import numpy as np
from mne.stats import permutation_cluster_1samp_test,summarize_clusters_stc

mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "DEN59":"ATT_26","WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31",
           "Mun79":"ATT_35","ATT_37_fsaverage":"ATT_37",
           "ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}
sub_key = {v: k for k,v in mri_key.items()}

sub = ["ATT_10"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
all_conds = [["audio","rest"],["visual","rest"],["audio","visual"],["visselten","audio"],["zaehlen","rest"]]
all_conds = [["visselten","audio"]]
threshold = 0.99
lower = 3e-27
upper = 3e-26
avg_clim = {"kind":"value","lims":[lower,(upper-lower)/2,upper]}
avg_clim = "auto"

src = mne.read_source_spaces(proc_dir+sub+"-src.fif")
filename = proc_dir+"stcs/"+sub
for conds in all_conds:
    stc_a = mne.read_source_estimate("{dir}stcs/nc_{a}_{b}_mean-lh.stc".format(
                                     dir=proc_dir,a=sub,b=conds[0]))
    stc_b = mne.read_source_estimate("{dir}stcs/nc_{a}_{b}_mean-lh.stc".format(
                                     dir=proc_dir,a=sub,b=conds[1]))
    stc_c = stc_a - stc_b
    favg = mlab.figure()
    stc_c.plot(sub_key[sub],hemi="both",figure=favg,clim=avg_clim)
    cnx = mne.spatial_src_connectivity(src)
    X = [np.load("{dir}stcs/nc_{a}_{b}_stc.npy".format(dir=proc_dir,a=sub,
                                                       b=conds[0])),
    np.load("{dir}stcs/nc_{a}_{b}_stc.npy".format(dir=proc_dir,a=sub,
                                                  b=conds[1]))]
    X = [x.transpose(0,2,1) for x in X]
    p_threshold = 0.0001
    f_threshold = stats.distributions.f.ppf(1. - p_threshold / 2.,
                                        X[0].shape[0]-1, X[1].shape[0]-1)
    f_obs, clusters, cluster_pv, H0 = clu = \
      mne.stats.spatio_temporal_cluster_test(X,connectivity=cnx,n_jobs=8,
                                             threshold=f_threshold)
    with open(proc_dir+"stcs/"+sub+"_clust","wb") as f:
        pickle.dump(clu,f)

    stc_clu = mne.stats.summarize_clusters_stc(clu,subject=sub,
                                               vertices=stc_c.vertices)
    fclu = mlab.figure()
    stc_clu.plot(sub_key[sub],hemi="both",clim=dict(kind='value',
                 lims=[0, 1, 7]), time_viewer=True,figure=fclu)
