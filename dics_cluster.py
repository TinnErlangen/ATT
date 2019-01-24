import mne
from mayavi import mlab
import pickle
import scipy.sparse
from scipy import stats
import numpy as np
from mne.stats import permutation_cluster_1samp_test,summarize_clusters_stc

subjs = ["ATT_10"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
conds = ["audio","rest"]
threshold = 0.99

for sub in subjs:
    src = mne.read_source_spaces(proc_dir+"nc_"+sub+"-src.fif")
    cnx = mne.spatial_src_connectivity(src)
    filename = proc_dir+"stcs/"+sub
    X = [np.load("{dir}stcs/nc_{a}_{b}_stc.npy".format(dir=proc_dir,a=sub,b=conds[0])),
    np.load("{dir}stcs/nc_{a}_{b}_stc.npy".format(dir=proc_dir,a=sub,b=conds[1]))]
    XX = X[0]-X[1]
    #XX = np.transpose(XX, [0,2,1])
    p_threshold = 0.001
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., XX.shape[0] - 1)
    t_obs, clusters, cluster_pv, H0 = clu = permutation_cluster_1samp_test(
            XX,connectivity=cnx,n_jobs=8,threshold=t_threshold)
    with open(proc_dir+"stcs/"+sub+"_clust","wb") as f:
        pickle.dump(clu,f)

    filename = "{dir}stcs/nc_{a}_{b}_mean-lh.stc".format(dir=proc_dir,a=sub,b=conds[0])
    stc_clu = mne.read_source_estimate(filename)
    stc_mean = stc_clu.copy()
    t_obs_new = np.zeros(stc_clu.data.shape)
    for c_idx,c in enumerate(clusters):
        if cluster_pv[c_idx] > 0.05:
            continue
        t_obs_new += t_obs * c.astype(np.float)
    stc_clu.data = t_obs_new
    XX_mean = np.mean(XX,axis=0)
    stc_mean.data = XX_mean
    stc_clu.plot(figure=0,subject="nc_"+sub,subjects_dir=subjects_dir,hemi="both",
    colormap="mne")
    mlab.title("Sig. clusters")
    stc_mean.plot(figure=1,subject="nc_"+sub,subjects_dir=subjects_dir,hemi="both",
    colormap="mne",clim={"kind":"values","lims":[np.min(XX_mean),0,np.max(XX_mean)]})
    mlab.title("Subtraction")
