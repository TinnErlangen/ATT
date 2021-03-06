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
from cnx_utils import plot_rgba

proc_dir = "/home/jev/ATT_dat/lmm/"
band = "alpha_1"
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [label.name for label in labels]
mat_n = len(labels)
triu_inds = np.triu_indices(mat_n, k=1)
hemi_name = {"lh":"left", "rh":"right"}

# Get the y-location of the label
label_ypos = {}
for label_idx, label in enumerate(labels):
    label_ypos[label.name] = (np.mean(label.pos[:, 1]), label_idx, label.name[-2:])
# sort by y location
label_ypos = {k:v for k,v in sorted(label_ypos.items(), key=lambda item: item[1][0])}
label_hemi = {hemi:{k:v for k,v in label_ypos.items() if v[-1] == hemi} for hemi in ["lh", "rh"]}

cond_keys = {"Intercept":"Resting","C(Block, Treatment('rest'))[T.task]":"Task",
             "params":"Sig. change"}

COI_cols = {'left motor': 'white', 'right motor': 'white',
            'left par-occipital': 'white', 'right par-occipital': 'white',
            'left parietal': 'white', 'right parietal': 'white'}


conds = ["rest","audio","visual","visselten","zaehlen"]
z_name = ""
no_Z = True
if no_Z:
    z_name = "no_Z"
    conds = ["rest","audio","visual","visselten"]

simp_conds = ["Intercept", "C(Block, Treatment('rest'))[T.task]"]

with open("{}{}/aic{}.pickle".format(proc_dir,band,z_name), "rb") as f:
      aic_comps = pickle.load(f)
predicted = aic_comps["predicted"]
params = aic_comps["simple_sig_params"][:,1]

### simple model

COIs = {"left motor":["L3969-lh", "L3395-lh"],
        "left parietal":["L4557-lh", "L7491_L4557-lh"]}

for COI_k, COI_v in COIs.items():
    pred = predicted["simple"]
    # generate mask
    mask = np.zeros((mat_n, mat_n))
    mask[triu_inds[0], triu_inds[1]] = aic_comps["dual_winner"]

    inds = list(np.where(aic_comps["dual_winner"])[0])
    cnx_preds = {c:np.zeros((mat_n, mat_n)) for c in simp_conds}
    cnx_preds["params"] = np.zeros((mat_n, mat_n))
    dfs = []

    for idx in inds:
        cnx_preds["params"][triu_inds[0][idx],triu_inds[1][idx]] = params[idx]
        for cond in simp_conds:
            cnx_preds[cond][triu_inds[0][idx],triu_inds[1][idx]] = pred[idx][cond]

    for cond in cnx_preds.keys():
        if cond in ["Intercept", "C(Block, Treatment('rest'))[T.task]"]:
            cnx_preds[cond] *= mask
            cnx_preds[cond][cnx_preds[cond]!=0] -= 0.5
        cnx_preds[cond] += cnx_preds[cond].T * -1
        if cond == "params":
            cnx_preds["params"][cnx_preds["params"]==0] = np.nan

        row_inds = np.array([label_names.index(r) for r in COI_v])
        for hemi, label_ypos in label_hemi.items():
            cnx_row = cnx_preds[cond][row_inds,].mean(axis=0, keepdims=True)
            col_order = np.array([v[1] for v in label_hemi[hemi].values()])
            df = pd.DataFrame(cnx_row[:,col_order],
                              ["{}, {} to {} cortex".format(cond_keys[cond], COI_k, hemi_name[hemi])],
                              [label_names[idx][:-3] for idx in list(col_order)])
            dfs.append(df)
    heat_df = dfs[0].append(dfs[1:])

    # eliminate all-zero columns
    for col in heat_df.columns:
        if all(np.isnan(heat_df[col][-2:])):
            del heat_df[col]
    #heat_df += 0.5

    fig, axes = plt.subplots(1, 2, figsize=(38.4, 10.8))
    sns.heatmap(heat_df, xticklabels=True, vmin=-0.02, vmax=0.02, ax=axes[0],
                cbar_kws={"label":"dPTE"})

    # sort out the colors for each region
    regs = list(heat_df.columns)
    tile_num = len(regs)//10+1 if len(regs)%10 else len(regs)//10
    x = np.tile(np.arange(10),tile_num)[:len(regs)]
    rgb = plt.get_cmap("tab10")(x)
    reg_colors = {k:rgb[idx] for idx, k in enumerate(regs)}

    # recolor the xticklabels
    xtls = axes[0].get_xticklabels()
    for xtl in xtls:
        xtl.set_c(reg_colors[xtl.get_text()])
    axes[0].set_xticklabels(xtls)

    labels_to_plot = [l for r in reg_colors.keys() for l in labels if r==l.name[:-3] and "lh" in l.name]
    brain = plot_rgba(np.stack(list(reg_colors.values())), labels_to_plot, parc,
                      figsize=(1920,1080), hemi="lh")
    for roi in COI_v:
        lab = labels[label_names.index(roi)]
        brain.add_label(lab, color=COI_cols[COI_k], borders=True, hemi="lh")
    brain.show_view(**{"view":"lateral","distance":300,"hemi":"lh"})
    img_lat = brain.screenshot()
    brain.show_view(**{"view":"dorsal","distance":400,"hemi":"lh"})
    img_dor = brain.screenshot()
    brain.close()
    img = np.hstack((img_lat,img_dor[:,640:1280,]))
    axes[1].imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("../images/coi_simp_{}.png".format(COI_k))

# cond model

cond_keys = {"Intercept":"Resting",
             "C(Block, Treatment('rest'))[T.audio]":"Audio",
             "C(Block, Treatment('rest'))[T.visual]":"Visual",
             "C(Block, Treatment('rest'))[T.visselten]":"Audio distraction",
             "params":"Sig. change"}

pred = predicted["cond"]
params = aic_comps["sig_params"]

COIs = {"left motor":["L3969-lh", "L3395-lh"],
        "right motor":["L3969-rh", "L3395-rh"],
        "left par-occipital":["L4236_L1933-lh"],
        "right par-occipital":["L4236_L1933-rh"],
        "left parietal":["L4557-lh", "L7491_L4557-lh"],
        "right parietal":["L4557-rh", "L7491_L4557-rh"]}

for COI_k, COI_v in COIs.items():
    cond_dict = aic_comps["cond_dict"]
    stat_cond_dict = {v:k for k,v in cond_dict.items()}
    mask = np.zeros((mat_n, mat_n))
    cond_winners = aic_comps["single_winner_ids"] == 2
    mask[triu_inds[0], triu_inds[1]] = cond_winners
    cond_vals = list(cond_dict.values())
    for stat_cond in list(cond_vals)[1:]:
        inds = list(np.where(cond_winners)[0])
        cnx_preds = {c:np.zeros((mat_n, mat_n)) for c in ["Intercept", stat_cond]}
        cnx_preds["params"]= np.zeros((mat_n, mat_n))

        dfs = []

        for idx in inds:
            cnx_preds["params"][triu_inds[0][idx],triu_inds[1][idx]] = params[idx][cond_vals.index(stat_cond)]
            for cond in ["Intercept", stat_cond]:
                cnx_preds[cond][triu_inds[0][idx],triu_inds[1][idx]] = pred[idx][cond]

        for cond in cnx_preds.keys():
            if cond != "params":
                cnx_preds[cond] *= mask
                cnx_preds[cond][cnx_preds[cond]!=0] -= 0.5
                cnx_preds[cond] += cnx_preds[cond].T * -1
            else:
                cnx_preds[cond] += cnx_preds[cond].T * -1
                cnx_preds["params"][cnx_preds["params"]==0] = np.nan

            row_inds = np.array([label_names.index(r) for r in COI_v])
            print("{}, {}".format(COI_v, row_inds))
            for hemi, label_ypos in label_hemi.items():
                cnx_row = np.nanmean(cnx_preds[cond][row_inds,], axis=0, keepdims=True)
                col_order = np.array([v[1] for v in label_hemi[hemi].values()])
                df = pd.DataFrame(cnx_row[:,col_order],
                                  ["{}, {} to {} cortex".format(cond_keys[cond], COI_k, hemi_name[hemi])],
                                  [label_names[idx][:-3] for idx in list(col_order)])
                dfs.append(df)
        heat_df = dfs[0].append(dfs[1:])

        #eliminate all-zero columns
        for col in heat_df.columns:
            if all(np.isnan(heat_df[col][-2:])):
                del heat_df[col]
        #heat_df += 0.5

        fig, axes = plt.subplots(1, 2, figsize=(38.4, 10.8))
        sns.heatmap(heat_df, xticklabels=True, vmin=-0.02, vmax=0.02, ax=axes[0],
                    cbar_kws={"label":"dPTE"})

        # sort out the colors for each region
        regs = list(heat_df.columns)
        tile_num = len(regs)//10+1 if len(regs)%10 else len(regs)//10
        x = np.tile(np.arange(10),tile_num)[:len(regs)]
        rgb = plt.get_cmap("tab10")(x)
        reg_colors = {k:rgb[idx] for idx, k in enumerate(regs)}

        # recolor the xticklabels
        xtls = axes[0].get_xticklabels()
        for xtl in xtls:
            xtl.set_c(reg_colors[xtl.get_text()])
        axes[0].set_xticklabels(xtls)

        labels_to_plot = [l for r in reg_colors.keys() for l in labels if r==l.name[:-3] and "lh" in l.name]

        brain = plot_rgba(np.stack(list(reg_colors.values())), labels_to_plot, parc,
                          figsize=(1920,1080), hemi="lh")

        for roi in COI_v:
            # for display purposes, show everything as lh
            this_roi = roi[:-3] + "-lh"
            lab = labels[label_names.index(this_roi)]
            brain.add_label(lab, color=COI_cols[COI_k], borders=True, hemi="lh")

        brain.show_view(**{"view":"lateral","distance":300,"hemi":"lh"})
        img_lat = brain.screenshot()
        brain.show_view(**{"view":"dorsal","distance":400,"hemi":"lh"})
        img_dor = brain.screenshot()
        brain.close()
        img = np.hstack((img_lat,img_dor[:,640:1280,]))
        axes[1].imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("../images/coi_cond_{}_{}.png".format(COI_k,
                                                          cond_keys[cond]))
