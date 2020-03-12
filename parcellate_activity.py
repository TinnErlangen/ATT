import mne
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.stats import sem
from mayavi import mlab
plt.ion()

subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = "ico5"
clust_thresh = 0.25
fuse_regions=True

# draw overall anova with parcellations, lateral and medial views
labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub', 'lh', subjects_dir=subjects_dir)
names = [label.name for label in labels if "lh" in label.name]

if fuse_regions:
    fused_labels = {}
    for l in labels:
        root_name = re.split("_",l.name)[0]
        if root_name in fused_labels:
            fused_labels[root_name] += l
        else:
            fused_labels[root_name] = l
    labels = []
    for k,v in fused_labels.items():
        v.name = k
        labels.append(v)
names = [label.name for label in labels]
colors = [label.color for label in labels if "lh" in label.name]
stc_stat = mne.read_source_estimate("../proc/stc_clu_tfce-lh.stc")
stc_stat.data = np.expand_dims(stc_stat.data[:,0],1)

parc_verts = [l.get_vertices_used() for l in labels]
parc_bools = [stc_stat.data[pv,0]>0 for pv in parc_verts]
perc_used = np.array([np.sum(pb)/len(pb) for pb in parc_bools])
parc_inds = list(np.where(perc_used>clust_thresh)[0])

# mfig = mlab.figure(size=(1200,600))
# brain = stc_stat.plot(subjects_dir=subjects_dir, colorbar=False,
#                       subject="fsaverage",figure=mfig,
#                       clim={"kind":"values","lims":[1,2,32]})
# brain.add_annotation("aparc_sub")
# brain_lat = mlab.screenshot(figure=mfig)
# mlab.close()
# mfig = mlab.figure(size=(1200,600))
# brain = stc_stat.plot(views=["med"],subjects_dir=subjects_dir,
#                       subject="fsaverage",figure=mfig,
#                       clim={"kind":"values","lims":[1,2,32]})
# brain.add_annotation("aparc_sub")
# brain_med = mlab.screenshot(figure=mfig)
# mlab.close()

# fig, axes = plt.subplots(2,1)
# axes[0].imshow(brain_lat)
# axes[0].axis("off")
# axes[1].imshow(brain_med)
# axes[1].axis("off")
# plt.tight_layout()

# now get the raw DICS
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
#subjs = ["ATT_10", "ATT_11"]
f_ranges = [[7,14],[15,22],[23,30],[31,38]]
all_f = [item for sublist in [list(np.arange(f[0],f[1]+1)) for f in f_ranges] for item in sublist]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = "ico5"
conds = ["rest","audio","visual","visselten","zaehlen"]
conds = ["audio","visual","visselten"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
#conds = ["audio","zaehlen"]

X = [[] for wav in wavs for cond in conds]
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
        for wav_idx,wav in enumerate(wavs):
            X_temp = []
            for fr in f_ranges:
                stc_temp = mne.read_source_estimate(
                        "{dir}stcs/nc_{a}_{b}_{c}_{f0}-{f1}Hz_{d}-lh.stc".format(
                          dir=proc_dir,a=sub,b=cond,c=wav,f0=fr[0],f1=fr[1],
                          d=spacing))
                stc_temp = morph.apply(stc_temp)
                X_temp.append(stc_temp.data.transpose(1,0))
            X_temp = np.array(X_temp)
            X[idx].append(X_temp.reshape(X_temp.shape[0]*X_temp.shape[1],
                                         X_temp.shape[2]))
            idx += 1
X = [(np.array(x)*1e+26).astype(np.float32) for x in X]
X = np.array(X)
XX = np.array([np.mean(X[i:i+4,],axis=0) for i in range(0,12,4)])

for idx,parc in enumerate(parc_inds):
    fig, axes = plt.subplots(3,1)
    plt.suptitle(names[parc])

    # first depict the ROI on the brain
    mfig = mlab.figure(size=(1200,600))
    brain = stc_stat.plot(subjects_dir=subjects_dir, colorbar=False,
                          subject="fsaverage",figure=mfig,
                          clim={"kind":"values","lims":[1,2,32]})
    brain.add_label(labels[parc],color="blue",alpha=0.7)
    axes[0].imshow(mlab.screenshot(figure=mfig))
    axes[0].axis("off")
    mlab.close()
    mfig = mlab.figure(size=(1200,600))
    brain = stc_stat.plot(views=["med"],subjects_dir=subjects_dir,
                          subject="fsaverage",figure=mfig,
                          clim={"kind":"values","lims":[1,2,32]})
    brain.add_label(labels[parc],color="blue",alpha=0.7)
    axes[1].imshow(mlab.screenshot(figure=mfig))
    axes[1].axis("off")
    mlab.close()

    temp_dat = XX[:,:,:,parc_verts[parc]].mean(axis=-1)
    t_temp_dat = temp_dat.mean(axis=1)/sem(temp_dat,axis=1)
    axes[2].imshow(t_temp_dat)
    plt.xticks(range(len(all_f)),[str(f) for f in all_f])
    plt.xlabel("Hz")
    plt.yticks(range(3),conds)
