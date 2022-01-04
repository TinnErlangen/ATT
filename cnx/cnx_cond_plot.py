from os import listdir
import numpy as np
from cnx_utils import plot_directed_cnx, make_brain_image, load_sparse, phi
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
band_info = {}
band_info["theta_0"] = {"freqs":list(np.arange(4,9)),"cycles":3, "name":"theta"}
band_info["alpha_0"] = {"freqs":list(np.arange(8,11)),"cycles":5, "name":"low alpha"}
band_info["alpha_1"] = {"freqs":list(np.arange(10,14)),"cycles":7, "name":"high alpha"}
band_info["beta_0"] = {"freqs":list(np.arange(13,31)),"cycles":9, "name":"beta"}
band_info["gamma_0"] = {"freqs":list(np.arange(31,49)),"cycles":9, "name":"gamma"}
cond_keys = {"rest":"Resting state", "audio":"Audio task",
             "visual":"Visual task", "visselten":"Auditory distraction",
             "zaehlen":"Backwards counting"}

bands = list(band_info.keys())
conds = list(cond_keys.keys())

for band in bands:
    for cond in conds:
        freqs = band_info[band]["freqs"]
        fs_src = mne.read_source_spaces("{}{}_ico{}-src.fif".format(proc_dir,"fsaverage", spacing))
        parc = "RegionGrowing_70"
        labels = mne.read_labels_from_annot("fsaverage",parc=parc,subjects_dir=subjects_dir)
        views = {"left":{"view":"lateral","distance":850,"hemi":"lh"},
                 "right":{"view":"lateral","distance":850,"hemi":"rh"},
                 "upper":{"view":"dorsal","distance":850},
                 "caudal":{"view":"caudal", "distance":850}}

        region_names = [lab.name for lab in labels]
        regions = []
        for rn in region_names:
            for l in labels:
                if l.name == rn:
                    regions.append(l)

        data = []
        for sub_idx,sub in enumerate(subjs):
            # we actually only need the dPTE to get the number of trials
            data_temp = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                                    cond, band))
            for epo_idx in range(data_temp.shape[0]):
                data.append(data_temp[epo_idx,])
        data = np.array(data)
        data = data.mean(axis=0)

        fontsize=110
        top = 100
        background = (1,1,1)
        brain_a = plot_directed_cnx(data, regions, parc, top_cnx=top, centre=0.5,
                                  background=background)
        vmin, vmax = get_vminmax(data, top)
        print("Top {}: {} - {}".format(top, vmin, vmax))
        img = make_brain_image(views, brain_a, text="A", text_loc="lup", text_pan=0,
                               fontsize=fontsize)
        fig1, ax1 = plt.subplots(1,1, figsize=(38.4, 12.8))
        ax1.imshow(img)
        ax1.axis("off")
        fig1.suptitle("{}, {} band, Strongest {} connections, dPTE magnitude range "
                      "{:.4f}-{:.4f}".format(cond_keys[cond], band_info[band]["name"],
                                             top, vmax, vmin), fontsize=48)
        plt.tight_layout()
        fig1.canvas.draw()
        mat1 = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
        mat1 = mat1.reshape(fig1.canvas.get_width_height()[::-1] + (3,))

        top = 200
        bottom = 100
        brain_b = plot_directed_cnx(data, regions, parc, top_cnx=top, bot_cnx=bottom,
                                  centre=0.5, background=background)
        vmin, vmax = get_vminmax(data, top, bot_cnx=bottom)
        print("Top {}-{}: {} - {}".format(top, bottom, vmin, vmax))
        img = make_brain_image(views, brain_b, text="B", text_loc="lup", text_pan=0,
                               fontsize=fontsize)
        fig2, ax2 =  plt.subplots(1,1, figsize=(38.4, 12.8))
        ax2.imshow(img)
        ax2.axis("off")
        fig2.suptitle("{}, {} band, Strongest {}-{} connections, dPTE magnitude range "
                      "{:.4f}-{:.4f}".format(cond_keys[cond], band_info[band]["name"],
                                             bottom, top, vmax, vmin), fontsize=48)
        plt.tight_layout()
        fig2.canvas.draw()
        mat2 = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
        mat2 = mat2.reshape(fig2.canvas.get_width_height()[::-1] + (3,))

        fig, axes = plt.subplots(2, 1, figsize=(38.4, 25.6))
        axes[0].imshow(mat1)
        axes[0].axis("off")
        axes[1].imshow(mat2)
        axes[1].axis("off")

        fig.tight_layout()
        fig.savefig("../images/cnx_{}_{}_top.png".format(cond, band))

        brain_a.close()
        brain_b.close()
        plt.close("all")
