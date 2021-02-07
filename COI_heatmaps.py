import pickle
import mne
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
font = {'weight' : 'bold',
        'size'   : 20}
matplotlib.rc('font', **font)
import matplotlib.pyplot as plt
plt.ion()

proc_dir = "/home/jev/ATT_dat/lmm/"
band = "alpha_1"
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
mat_n = len(labels)
triu_inds = np.triu_indices(mat_n, k=1)

# Get the y-location of the label
label_ypos = {}
for label_idx, label in enumerate(labels):
    label_ypos[label.name] = (np.mean(label.pos[:, 1]), label_idx, label.name[-2:])
# sort by y location
label_ypos = {k:v for k,v in sorted(label_ypos.items(), key=lambda item: item[1][0])}
label_hemi = {hemi:{k:v for k,v in label_ypos.items() if v[-1] == hemi} for hemi in ["lh", "rh"]}

cond_keys = {"Intercept":"Resting","C(Block, Treatment('rest'))[T.task]":"Task"}

conds = ["rest","audio","visual","visselten","zaehlen"]
z_name = ""
no_Z = True
if no_Z:
    z_name = "no_Z"
    conds = ["rest","audio","visual","visselten"]

simp_conds = ["Intercept", "C(Block, Treatment('rest'))[T.task]"]

COIs = {"Left Motor":["L3969-lh", "L3395-lh"],
        "Left Parietal-Occipital":["L4236_L1933-lh"],
        "Right Parietal-Occipital":["L4236_L1933-rh"],
        "Left Parietal":["L4557-lh", "L7491_L4556-lh"]}

with open("{}{}/aic{}.pickle".format(proc_dir,band,z_name), "rb") as f:
      aic_comps = pickle.load(f)
predicted = aic_comps["predicted"]

### simple model

COIs = {"Left Motor":["L3969-lh", "L3395-lh"],
        "Left Parietal":["L4557-lh", "L7491_L4557-lh"]}

pred = predicted["simple"]
# generate mask
mask = np.zeros((mat_n, mat_n))
mask[triu_inds[0], triu_inds[1]] = aic_comps["dual_winner"]

inds = list(np.where(aic_comps["dual_winner"])[0])
cnx_preds = {c:np.zeros((mat_n, mat_n)) for c in simp_conds}
dfs = []
for cond in simp_conds:
    for idx in inds:
        cnx_preds[cond][triu_inds[0][idx],triu_inds[1][idx]] = pred[idx][cond]
    cnx_preds[cond] *= mask
    cnx_preds[cond][cnx_preds[cond]!=0] -= 0.5
    cnx_preds[cond] += cnx_preds[cond].T * -1

    for COI_name, regs in COIs.items():
        row_inds = np.array([label_names.index(r) for r in regs])
        for hemi, label_ypos in label_hemi.items():
            cnx_row = cnx_preds[cond][row_inds,].mean(axis=0, keepdims=True)
            col_order = np.array([v[1] for v in label_hemi[hemi].values()])
            df = pd.DataFrame(cnx_row[:,col_order],
                              ["{} {} to {}".format(cond_keys[cond], COI_name, hemi)],
                              [label_names[idx][:-3] for idx in list(col_order)])
            dfs.append(df)
heat_df = dfs[0].append(dfs[1:])
sns.heatmap(heat_df, xticklabels=True)
