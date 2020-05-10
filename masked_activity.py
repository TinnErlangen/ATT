import mne
from mayavi import mlab
import pickle
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from mne.stats.cluster_level import _find_clusters
from surfer import Brain
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
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]

subjects_dir = "/home/jeff/hdd/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
stat_dir = "superanova_1/"
spacing = "ico4"
conds = ["audio", "visual", "visselten"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
#conds = ["visual","visselten"]
cond_str = conds[0]
for c in conds[1:]:
    cond_str += "_" + c
thresh_str = "tfce"

# get connectivity
fs_src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir, "fsaverage",
                                                         spacing))
cnx = mne.spatial_src_connectivity(fs_src)


fr = [4,30]
frange = list(np.arange(fr[0],fr[1]+1))
f_subs = [[0,3],[7,26]]
###### first get our clusters/masks
stc_clu = mne.read_source_estimate("{}{}stc_clu_{}-{}Hz_{}_{}_A-lh.stc".format(
                                   proc_dir, stat_dir, fr[0], fr[1], cond_str,
                                   thresh_str))
stc_clu.subject = "fsaverage"
with open("{}{}clu_{}-{}Hz_{}_{}_A".format(proc_dir, stat_dir, fr[0], fr[1], cond_str,
                                         thresh_str),"rb") as f:
    f_obs, clusters, cluster_pv, H0 = pickle.load(f)
f_thresh = np.quantile(H0,0.95)
# stc_clu = mne.stats.summarize_clusters_stc(clu,subject="fsaverage",
#                                            p_thresh=0.05,
#                                            vertices=stc_clu.vertices)
meta_clusts = _find_clusters(stc_clu.data[:,0],1e-8,connectivity=cnx)[0]
clust_labels = []
for mc in meta_clusts:
    temp_stc = stc_clu.copy()
    temp_stc.data[:] = np.zeros((temp_stc.data.shape[0],1))
    temp_stc.data[mc,0] = 1
    lab = [x for x in mne.stc_to_label(temp_stc,src=fs_src) if x][0]
    clust_labels.append(lab)

X = np.zeros((len(subjs),len(clust_labels),fr[1]-fr[0]+1,len(conds),len(wavs)))
for sub_idx,sub in enumerate(subjs):
    src = mne.read_source_spaces("{}{}_{}-src.fif".format(proc_dir,sub,spacing))
    vertnos=[s["vertno"] for s in src]
    morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                     subject_to="fsaverage",
                                     spacing=int(spacing[-1]),
                                     subjects_dir=subjects_dir,
                                     smooth=None)


    for cond_idx, cond in enumerate(conds):
        for wav_idx, wav in enumerate(wavs):
            X_temp = []
            stc_temp = mne.read_source_estimate(
                    "{dir}stcs/nc_{a}_{b}_{c}_{f0}-{f1}Hz_{d}-lh.stc".format(
                      dir=proc_dir,a=sub,b=cond,c=wav,f0=fr[0],f1=fr[1],
                      d=spacing))
            stc_temp = morph.apply(stc_temp)
            X[sub_idx,:,:,cond_idx,wav_idx] = \
              mne.extract_label_time_course(stc_temp,clust_labels,
                                            fs_src,mode="mean")

for fs in f_subs:
    temp_frange = list(np.arange(frange[fs[0]],frange[fs[1]])+1)
    XX = X[:,:,fs[0]:fs[1],]
    XX = XX.mean(axis=-1) # average over tone
    XX_t = XX/sem(XX,axis=0) # t by subject
    XXX_t = XX_t.mean(axis=0)
    these_data = XXX_t
    for cl_idx,cl in enumerate(clust_labels):
        hemi = cl.hemi
        fig, axes = plt.subplots(4,1)
        mfig = mlab.figure()
        brain = Brain("fsaverage", hemi, "inflated",
                      subjects_dir=subjects_dir, figure=mfig)
        brain.add_label(cl,color="blue",alpha=0.7)
        axes[0].imshow(mlab.screenshot(figure=mfig))
        axes[0].axis("off")
        mlab.close()
        mfig = mlab.figure()
        brain = Brain("fsaverage", hemi, "inflated",
                      subjects_dir=subjects_dir, figure=mfig,
                      views=["med"])
        brain.add_label(cl,color="blue",alpha=0.7)
        axes[1].imshow(mlab.screenshot(figure=mfig))
        axes[1].axis("off")
        mlab.close()
        temp_mean = these_data.mean(axis=0)
        temp_sem = sem(these_data,axis=0)
        axes[2].imshow(these_data[cl_idx,].T)
        axes[2].set_xticks(np.arange(these_data.shape[1]))
        axes[2].set_xticklabels([str(t) for t in temp_frange])
        axes[2].set_yticks(np.arange(3))
        axes[2].set_yticklabels(conds)
        axes[3].imshow((f_obs[:,meta_clusts[cl_idx]]>f_thresh))
        axes[3].set_yticks(np.arange(X.shape[2]))
        axes[3].set_yticklabels([str(t) for t in list(np.arange(fr[0],fr[1]+1))])
        plt.suptitle("{}-{}Hz, Cluster {}".format(temp_frange[0],temp_frange[-1],cl_idx))
