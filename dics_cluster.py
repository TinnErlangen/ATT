import mne
from mayavi import mlab
import pickle
import scipy.sparse
from scipy import stats
import numpy as np
from mne.stats import permutation_cluster_1samp_test,summarize_clusters_stc


subjs = ["ATT_21"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
conds = ["audio","visual"]
threshold = 0.99

for sub in subjs:
    filename = proc_dir+"stcs/"+sub
    src = mne.read_source_spaces(proc_dir+sub+"-src.fif")
    stc = mne.read_source_estimate("{dir}stcs/nc_{a}_{b}_mean-lh.stc".format(dir=proc_dir,a=sub,b="rest"))
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

    stc_c = mne.stats.summarize_clusters_stc(clu,subject=sub,vertices=stc.vertices)
    stc_c.plot(sub,hemi="both",clim=dict(kind='value', lims=[0, 1, 40]), time_viewer=True)
