from os import listdir
import numpy as np
from cnx_utils import plot_rgba, make_brain_image, plot_parc_compare
import mne
import pickle
import re
from matplotlib import cm
import matplotlib.pyplot as plt
plt.ion()


subjects_dir = "/home/jev/hdd/freesurfer/subjects/"
proc_dir = "../proc/"
spacing = 4
calc = False
band_info = {}
cmap = "inferno"
band_info["alpha_1"] = {"freqs":list(np.arange(10,13)),"cycles":7}

band = "alpha_1"
freqs = band_info[band]["freqs"]
fs_src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,"fsaverage", spacing))
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc=parc,subjects_dir=subjects_dir)
views = {"left":{"view":"lateral","distance":500,"hemi":"lh"},
         "right":{"view":"lateral","distance":500,"hemi":"rh"},
         "upper":{"view":"dorsal","distance":500},
         "caudal":{"view":"caudal", "distance":500}}

region_names = [lab.name for lab in labels]
regions = []
for rn in region_names:
    for l in labels:
        if l.name == rn:
            regions.append(l)

mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "DEN59":"ATT_26","WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31",
           "Mun79":"ATT_35","ATT_37_fsaverage":"ATT_37",
           "ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}
sub_key = {v: k for k,v in mri_key.items()}

# all subjs
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]

if calc:
    morphs = {}
    for sub in subjs:
        src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,sub,spacing))
        morph = mne.compute_source_morph(src,subject_from=sub_key[sub],
                                         subject_to="fsaverage",
                                         spacing=spacing,
                                         subjects_dir=subjects_dir,
                                         smooth=None)
        morphs[sub] = morph

    group_id = []
    data = [[] for reg in regions]
    filelist = listdir(proc_dir+"/stcs/")
    for filename in filelist:
        match = re.search("rest_{}-{}Hz_[0-9]*_ico4-lh.stc".format(freqs[0],freqs[-1]),filename)
        if not match:
            continue
        print(filename)
        trial_info = re.match("nc_(ATT_[0-9]+)_rest", filename).groups()
        stc = mne.read_source_estimate("{}/stcs/{}".format(proc_dir,filename))
        stc = morphs[trial_info[0]].apply(stc)
        for reg_idx,reg in enumerate(regions):
            temp_data = mne.extract_label_time_course(stc,reg,fs_src,mode="mean")
            data[reg_idx].append(temp_data.mean())
    data = np.array(data)
    data = data.mean(axis=1)
    np.save("{}rest_dics.npy".format(proc_dir), data)
else:
    data = np.load("{}rest_dics.npy".format(proc_dir))

data_norm = (data) / (data.max())
rgbs = cm.get_cmap(cmap)(data_norm)
rgbs[:,-1] = data_norm
brain = plot_rgba(rgbs, regions, parc, background=(1,1,1))
img = make_brain_image(views, brain, cbar=cmap, vmin=0, vmax=data.max(),
                       orient="horizontal", cbar_label="DICS power (NAI normalised)")
fig, ax = plt.subplots(1,1, figsize=(38.4, 12.8))
ax.imshow(img)
ax.axis("off")
plt.tight_layout()
plt.savefig("../images/dics_rest.png")
