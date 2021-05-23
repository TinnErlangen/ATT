from os import listdir
import numpy as np
from cnx_utils import plot_directed_cnx, make_brain_figure, load_sparse, phi
import mne
import pickle
import re
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
plt.ion()

class AlphaCMap:
    def __init__(self, base_color):
        self.base_color = base_color
    def __call__(self, X):
        rgba = np.zeros((len(X), 4))
        rgba[:, :3] = self.base_color
        rgba[:, -1] = X
        return rgba

def get_vminmax(mat, top_cnx, bot_cnx=None, centre=0.5):
    matflat = mat[mat!=0]
    matflat -= centre
    matflat = np.abs(matflat)
    matflat = np.sort(matflat)
    vmin = matflat[-top_cnx]
    if bot_cnx:
        vmax = matflat[-bot_cnx]
    else:
        vmax = matflat[-1]

    return vmin, vmax

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33", "ATT_34",
         "ATT_35", "ATT_36", "ATT_37"]

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
views = {"left":{"view":"lateral","distance":850,"hemi":"lh"},
         "right":{"view":"lateral","distance":850,"hemi":"rh"},
         "upper":{"view":"dorsal","distance":850}}

region_names = [lab.name for lab in labels]
regions = []
for rn in region_names:
    for l in labels:
        if l.name == rn:
            regions.append(l)

data = []
for sub_idx,sub in enumerate(subjs):
    # we actually only need the dPTE to get the number of trials
    data_temp = load_sparse("{}nc_{}_rest_dPTE_{}.sps".format(proc_dir, sub,
                                                              band))
    for epo_idx in range(data_temp.shape[0]):
        data.append(data_temp[epo_idx,])
data = np.array(data)
data = data.mean(axis=0)

top = 150
brain = plot_directed_cnx(data, regions, parc, top_cnx=top, centre=0.5)
vmin, vmax = get_vminmax(data, top)
print("Top {}: {} - {}".format(top, vmin, vmax))
fig1 = make_brain_figure(views, brain)
fig1.suptitle("Resting state dPTE: Strongest {} connections".format(top), fontsize=48)
plt.tight_layout()

top = 300
bottom = 150
brain = plot_directed_cnx(data, regions, parc, top_cnx=top, bot_cnx=bottom, centre=0.5)
vmin, vmax = get_vminmax(data, top, bot_cnx=bottom)
print("Top {}-{}: {} - {}".format(top, bottom, vmin, vmax))
fig2 = make_brain_figure(views, brain)
fig2.suptitle("Resting state dPTE: Strongest {}-{} connections".format(top, bottom), fontsize=48)
plt.tight_layout()


#plt.savefig("test.png")
