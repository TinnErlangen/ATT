import mne
from mayavi import mlab
import pickle
import scipy.sparse
from scipy import stats
import numpy as np
from mne.stats import f_mway_rm,summarize_clusters_stc
import matplotlib.pyplot as plt
plt.ion()

def stat_fun(*args):
    # get f-values only.
    return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                     effects=effects, return_pvals=return_pvals)[0]

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
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
conds = ["rest","audio","visual","visselten","zaehlen"]
conds = ["audio","visual","visselten","zaehlen"]
#conds = ["audio","zaehlen"]

X = [[] for cond in conds]
fs_src = mne.read_source_spaces("{}{}-src.fif".format(proc_dir,"fsaverage"))
cnx = mne.spatial_src_connectivity(fs_src)
del fs_src
for sub_idx,sub in enumerate(subjs):
    src = mne.read_source_spaces("{}{}-src.fif".format(proc_dir,sub))
    morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                     subjects_dir=subjects_dir,
                                     spacing=[s["vertno"] for s in src])
    for cond_idx,cond in enumerate(conds):
        stc_temp = mne.read_source_estimate(
                "{dir}stcs/nc_{a}_{b}_mean-lh.stc".format(dir=proc_dir,a=sub,
                                                          b=cond))
        stc_temp = morph.apply(stc_temp)
        X[cond_idx].append(stc_temp.data)

X = [np.array(x).transpose(0,2,1) for x in X]
p_threshold = 0.01
f_threshold = stats.distributions.f.ppf(1. - p_threshold / 2.,
                                    X[0].shape[0]-1, X[1].shape[0]-1)
factor_levels = [len(conds)]
effects = ["A"]
return_pvals=False
masks = [np.ones(stc_temp.data[:,0].shape,dtype="bool")]
try:
    f_obs, clusters, cluster_pv, H0 = clu = \
      mne.stats.spatio_temporal_cluster_test(X,connectivity=cnx,n_jobs=8,
                                             threshold=f_threshold,
                                             stat_fun=stat_fun)
    stc_clu = mne.stats.summarize_clusters_stc(clu,subject="fsaverage",
                                           vertices=stc_temp.vertices)
    fclu = mlab.figure()
    stc_clu.plot(sub_key[sub],hemi="both",clim=dict(kind='value',
                 lims=[0,1,7]), time_viewer=True,figure=fclu)
    with open(proc_dir+"stcs/"+sub+"_clust","wb") as f:
        pickle.dump(clu,f)
    masks = [stc_clu.data[:,x] for x in range(1,stc_clu.data.shape[1])]
except:
    print("No significant results.")

min_disp_thresh=0.3
max_disp_thresh = 0.8
for mask_idx,mask in enumerate(masks):
    XX = [x.mean(axis=1)*mask for x in X]
    XXX = np.array(XX)
    XXX = np.sort(np.array(list(set(XXX.flatten()))))
    min_disp_thresh_idx = int(np.round(XXX.size*min_disp_thresh))
    max_disp_thresh_idx = int(np.round(XXX.size*max_disp_thresh))
    min_disp_thresh = XXX[min_disp_thresh_idx]
    max_disp_thresh = XXX[max_disp_thresh_idx]
    stc_conds = []
    for cond_idx,cond in enumerate(conds):
        stc_ttemp = stc_temp.copy()
        stc_ttemp.data[:,0] = XX[cond_idx].mean(axis=0)
        stc_ttemp.save("{}{}_avg_stc".format(proc_dir,cond))
        stc_conds.append(stc_ttemp)
        mfig = mlab.figure()
        mlab.title("Cluster {}, condition {}".format(mask_idx,cond))
        stc_ttemp.plot(sub_key[sub],hemi="both",figure=mfig,clim={"kind":"value",
                       "lims":[min_disp_thresh,
                              (max_disp_thresh-min_disp_thresh)/2+min_disp_thresh,
                              max_disp_thresh]})
    avgs = []
    for xx in XX:
        avgs.append([])
        inds = xx>0
        for sub_idx in range(inds.shape[0]):
            avgs[-1].append(xx[sub_idx,inds[sub_idx,]].mean())
    avgs = np.array(avgs)
    avg = avgs.mean(axis=1)
    sem = stats.sem(avgs,axis=1)
    plt.bar(np.arange(len(conds)),avg,yerr=sem,tick_label=conds)
