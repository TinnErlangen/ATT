from statsmodels.regression.mixed_linear_model import MixedLMResults
import numpy as np
from cnx_utils import plot_rgba, write_brain_image
import mne
import pickle
import matplotlib.pyplot as plt
plt.ion()

def plot_vec(vec, labs):
    brains = []
    nz_inds = vec!=0
    vec_abs = np.abs(vec)
    vec_min, vec_max = vec_abs[nz_inds].min(), vec_abs[nz_inds].max()
    vec_abs[nz_inds] = (vec_abs[nz_inds] - vec_min) / (vec_max - vec_min)
    vec_norm = vec_abs * np.sign(vec)
    for lab_idx, lab in enumerate(labs):
        pinds, ninds = vec_norm[:, lab_idx]>0, vec_norm[:, lab_idx]<0
        this_rgba = np.zeros((len(labels), 4))
        this_rgba[pinds, 0] = 1
        this_rgba[ninds,2] = 1
        this_rgba[:,3] = abs(vec_norm[:,lab_idx])

        brains.append(plot_rgba(this_rgba, labels, parc, lup_title=lab))

    return brains

proc_dir = "/home/jev/ATT_dat/lmm_dics/"
band = "alpha_1"
node_n = 70
threshold = 0.05
cond_threshold = 0.05
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage", parc)
mat_n = len(labels)
calc_aic = False

views = {"left":{"view":"lateral","distance":800,"hemi":"lh"},
         "right":{"view":"lateral","distance":800,"hemi":"rh"},
         "upper":{"view":"dorsal","distance":900}
}

models = ["null", "simple", "cond"]
vars = ["aics", "order", "probs", "threshed"]
conds = ["audio","visual","visselten"]
#var_base = "C(Block, Treatment('audio'))" # stem of the condition names in statsmodels format
#stat_conds = [var_base+"[T."+cond+"]" for cond in conds[1:]]
stat_conds = ["RT", "RT:C(Block, Treatment('audio'))[T.visual]",
              "RT:C(Block, Treatment('audio'))[T.visselten]"]

if calc_aic:
    aics = {mod:np.empty(node_n) for mod in models}
    aics_pvals = {mod:[None for n in range(node_n)] for mod in models}
    aics_params = {mod:[None for n in range(node_n)] for mod in models}
    aics_confint = {mod:[None for n in range(node_n)] for mod in models}
    for mod in models:
        for n_idx in range(node_n):
            print(n_idx)
            this_mod = MixedLMResults.load("{}{}/{}_reg70_lmm_byresp_{}.pickle".format(proc_dir,band,mod,n_idx))
            aics[mod][n_idx] = this_mod.aic
            aics_pvals[mod][n_idx] = this_mod.pvalues
            aics_params[mod][n_idx] = this_mod.params
            aics_confint[mod][n_idx] = this_mod.conf_int()

    aic_comps = {var:np.empty((node_n,len(models))) for var in vars}
    aic_comps["models"] = models
    aic_comps["winner_ids"] = np.empty(node_n)
    aic_comps["winner_margin"] = np.empty(node_n)
    aic_comps["simp_params"] = np.zeros(node_n)
    aic_comps["simp_confint_params"] = np.zeros((node_n,2))
    aic_comps["cond_params"] = np.zeros((node_n,len(stat_conds)))
    aic_comps["cond_confint_params"] = np.zeros((node_n,len(stat_conds),2))
    for n_idx in range(node_n):
        aic_array = np.array([aics[mod][n_idx] for mod in models])
        aic_comps["aics"][n_idx,] = aic_array
        aic_prob = np.exp((aic_array.min()-aic_array)/2)
        aic_comps["probs"][n_idx,] = aic_prob
        aic_order = np.argsort(aic_prob)
        aic_comps["order"][n_idx,] = aic_order
        aic_comps["winner_ids"][n_idx] = np.where(aic_order==len(models)-1)[0][0]
        aic_comps["winner_margin"][n_idx] = np.sort(aic_prob.copy())[len(models)-2] - aic_array.min()
        aic_threshed = aic_prob.copy()
        aic_threshed[aic_threshed<threshold] = 0
        aic_threshed[aic_threshed>0] = 1
        aic_comps["threshed"][n_idx,] = aic_threshed
        if np.array_equal(aic_threshed, [0,1,0]) or np.array_equal(aic_threshed, [0,1,1]):
            aic_comps["simp_params"][n_idx] = aics_params["simple"][n_idx]["RT"]
            aic_comps["simp_confint_params"][n_idx] = \
              [aics_confint["simple"][n_idx][0]["RT"],
               aics_confint["simple"][n_idx][1]["RT"]]
        if np.array_equal(aic_threshed, [0,0,1]):
            for sc_idx,sc in enumerate(stat_conds):
                aic_comps["cond_params"][n_idx][sc_idx] = \
                  aics_params["cond"][n_idx][sc]
                aic_comps["cond_confint_params"][n_idx][sc_idx] = \
                  [aics_confint["cond"][n_idx][0][sc], aics_confint["cond"][n_idx][1][sc]]

    with open("{}{}/aic_byresp.pickle".format(proc_dir,band), "wb") as f:
        pickle.dump(aic_comps,f)
else:
    with open("{}{}/aic_byresp.pickle".format(proc_dir,band), "rb") as f:
        aic_comps = pickle.load(f)

brains = []
vec = np.expand_dims(aic_comps["simp_params"],1)
simp_brains = plot_vec(vec, ["Simple: RT Slope"])
write_brain_image("simple_rt", views, simp_brains[0], dir="../images/")

vec = aic_comps["cond_params"]
cond_brains = plot_vec(vec, ["Audio RT Slope", "Visual RT Slope", "Visselten RT Slope"])
for cond, brain in zip(conds, cond_brains):
    write_brain_image(cond+"_rt", views, brain, dir="../images/")
