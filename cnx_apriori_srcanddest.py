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
conds = ["audio", "visual", "visselten", "zaehlen"]
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

src_regions = {"Left A1":["L2235-lh"], "Left V1":["L2340_L1933-lh"],
               "Right V1":["L2340_L1933-rh"]}
dest_regions = {"Left M1":["L8143_L7523-lh", "L3395-lh", "L3969-lh"],
                "Left Superior-Parietal":["L4556-lh","L8143-lh","L7491_L4557-lh"]}

# get the indices of paired regions
pairs_info = {}
for src_reg_k,src_reg_v in src_regions.items():
    src_inds = []
    for v in src_reg_v:
        for lab_idx,lab in enumerate(labels):
            if v == lab.name:
                src_inds.append(lab_idx)
        for dest_reg_k, dest_reg_v in dest_regions.items():
            pairs = []
            for lab_idx,lab in enumerate(labels):
                for v in dest_reg_v:
                    if v == lab.name:
                        pairs += [(x,lab_idx) for x in src_inds]
            pairs_info[src_reg_k+"-"+dest_reg_k] = {"inds":pairs}

data = {pi:[] for pi in pairs_info.keys()}
predictor_vars = ["Laut","Angenehm","Subj","Block","Wav"]
#predictor_vars = ["Subj","Block"]
dm = pd.DataFrame(columns=predictor_vars)
group_id = []
for sub_idx,sub in enumerate(subjs):
    # make the df and data object for this particular subject
    for cond_idx,cond in enumerate(conds):
        dPTE = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                           cond, freq))
        for wav_idx,wav in enumerate(wavs):
            dPTE = load_sparse("{}nc_{}_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                               cond, wav, freq))
            for d_idx in range(len(dPTE)):
                tril_inds = np.tril_indices(dPTE[d_idx,].shape[0],k=-1)
                unter_triang = dPTE[d_idx,].copy().T
                unter_triang = 0.5 - (unter_triang - 0.5)
                dPTE[d_idx,tril_inds[0],tril_inds[1]] = unter_triang[tril_inds[0],tril_inds[1]]
            sel_inds = (df["Block"]==cond) & (df["Wav"]==wav) & (df["Subj"]==sub)
            #sel_inds = (df["Block"]==cond) & (df["Subj"]==sub)

            dPTE_slices = {}
            for k,v in pairs_info.items():
                temp_inds = list(zip(*v["inds"]))
                dPTE_slices[k] = dPTE[:,temp_inds[0],temp_inds[1]].mean(axis=1)

            if avg_trials:
                dm = dm.append(df[sel_inds])
                #dm = dm.append({"Subj":sub,"Block":cond},ignore_index=True)
                for k,v in dPTE_slices.items():
                    data[k].append(v.mean())
                group_id.append(sub_idx)
            else:
                for epo_idx in range(len(dPTE)):
                    dm = dm.append(df[sel_inds])
                    #dm = dm.append({"Subj":sub,"Block":cond},ignore_index=True)
                    for k,v in dPTE_slices.items():
                        data[k].append(v[epo_idx,])
                    group_id.append(sub_idx)

formula = "Brain ~ Laut + Angenehm + C(Block, Treatment('audio')) + Wav"
#formula = "Brain ~ C(Block, Treatment('rest'))"
mod_fits = {}
for k,v in data.items():
    dm_temp = dm.copy()
    dm_temp["Brain"] = v
    model = MixedLM.from_formula(formula, dm_temp, groups=group_id)
    mod_fits[k] = model.fit()
