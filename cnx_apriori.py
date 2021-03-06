from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd
from cnx_utils import load_sparse, plot_directed_cnx, phi
import numpy as np
import mne
import pickle

def mass_uv_mixedlmm(formula, data, uv_data, group_id, re_formula=None):
    mods = [[] for source_idx in range(uv_data.shape[1])]
    for source_idx in range(uv_data.shape[1]):
        for dest_idx in range(uv_data.shape[2]):
            if all(uv_data[:,source_idx,dest_idx]==0):
                mods[source_idx].append(None)
                continue
            #print("Source {}, Destination {}".format(source_idx, dest_idx), end="\r")
            print("Source {}, Destination {}".format(source_idx, dest_idx))
            data_temp = data.copy()
            data_temp["Brain"] = uv_data[:,source_idx,dest_idx]
            model = MixedLM.from_formula(formula, data_temp, groups=group_id)
            mod_fit = model.fit()
            mods[source_idx].append(mod_fit)
    return mods

proc_dir = "/home/jeff/ATT_dat/proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
freq = "alpha_0"
avg_trials = False
conds = ["audio", "visual", "visselten"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
factor_levels = [len(conds), len(wavs)]
effects = ["A","B","A:B"]
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
effect_idx = 0
p_thresh = 0.05
cluster_idx = 0

df = pd.read_pickle("../behave/laut")
df_ang = pd.read_pickle("../behave/ang")
df["Angenehm"]=df_ang["Angenehm"]

regions = ["L2235-lh", "L2235-rh", "L2340_L1933-lh", "L2340_L1933-rh"]
sub_labels = []
sub_labels_inds = []
for reg in regions:
    for lab_idx,lab in enumerate(labels):
        if reg == lab.name:
            sub_labels.append(lab)
            sub_labels_inds.append(lab_idx)

data = []
predictor_vars = ["Laut","Angenehm","Subj","Block","Wav"]
dm = pd.DataFrame(columns=predictor_vars)
idx_borders = []
idx_border = 0
group_id = []
for sub_idx,sub in enumerate(subjs):
    idx_borders.append([idx_border])
    # make the df and data object for this particular subject
    for cond_idx,cond in enumerate(conds):
        for wav_idx,wav in enumerate(wavs):
            dPTE = load_sparse("{}nc_{}_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                               cond, wav, freq))
            for d_idx in range(len(dPTE)):
                tril_inds = np.tril_indices(dPTE[d_idx,].shape[0],k=-1)
                unter_triang = dPTE[d_idx,].copy().T
                unter_triang = 0.5 - (unter_triang - 0.5)
                dPTE[d_idx,tril_inds[0],tril_inds[1]] = unter_triang[tril_inds[0],tril_inds[1]]
            sel_inds = (df["Block"]==cond) & (df["Wav"]==wav) & (df["Subj"]==sub)
            if avg_trials:
                dm = dm.append(df[sel_inds])
                data.append(dPTE[:,sub_labels_inds].mean(axis=0))
                group_id.append(sub_idx)
                idx_border += 1
            else:
                for epo_idx in range(len(dPTE)):
                    dm = dm.append(df[sel_inds])
                    data.append(dPTE[epo_idx,sub_labels_inds])
                    group_id.append(sub_idx)
                    idx_border += 1
data = np.array(data)

formula = "Brain ~ Laut + Angenehm + Block + Wav"
mods = mass_uv_mixedlmm(formula, dm, data, group_id)

for mod_idx,mod in enumerate(mods):
    for m_idx,m in enumerate(mod):
        if m is not None:
            m.save("{}{}/src_{}-dest_{}.pickle".format(proc_dir,freq,mod_idx,m_idx))
