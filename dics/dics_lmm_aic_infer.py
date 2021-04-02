from statsmodels.regression.mixed_linear_model import MixedLMResults
import numpy as np
from cnx_utils import plot_rgba, write_brain_image, plot_parc_compare
import mne
import pickle
import matplotlib.pyplot as plt
plt.ion()

proc_dir = "/home/jev/ATT_dat/lmm_dics/"
band = "alpha_1"
node_n = 70
threshold = 0.05
cond_threshold = 0.05
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage", parc)
mat_n = len(labels)
calc_aic = False
top_cnx = 250

# import os
# os.environ["QT_API"] = "pyqt5"

views = {"left":{"view":"lateral","distance":800,"hemi":"lh"},
         "right":{"view":"lateral","distance":800,"hemi":"rh"},
         "upper":{"view":"dorsal","distance":900}
}

models = ["null","simple","cond"]
vars = ["aics", "order", "probs", "threshed"]
conds = ["rest","audio","visual","visselten","zaehlen"]
conds = ["rest","audio","visual","visselten"]
var_base = "C(Block, Treatment('rest'))" # stem of the condition names in statsmodels format
stat_conds = [var_base+"[T."+cond+"]" for cond in conds[1:]]

if calc_aic:
    aics = {mod:np.empty(node_n) for mod in models}
    aics_pvals = {mod:[None for n in range(node_n)] for mod in models}
    aics_params = {mod:[None for n in range(node_n)] for mod in models}
    aics_confint = {mod:[None for n in range(node_n)] for mod in models}
    for mod in models:
        for n_idx in range(node_n):
            print(n_idx)
            this_mod = MixedLMResults.load("{}{}/{}_reg70_lmm_{}.pickle".format(proc_dir,band,mod,n_idx))
            aics[mod][n_idx] = this_mod.aic
            aics_pvals[mod][n_idx] = this_mod.pvalues
            aics_params[mod][n_idx] = this_mod.params
            aics_confint[mod][n_idx] = this_mod.conf_int()

    aic_comps = {var:np.empty((node_n,len(models))) for var in vars}
    aic_comps["models"] = models
    aic_comps["winner_ids"] = np.empty(node_n)
    aic_comps["winner_margin"] = np.empty(node_n)
    aic_comps["double_winners"] = np.zeros(node_n)
    aic_comps["single_winners"] = np.zeros(node_n)
    aic_comps["dual_winners"] = np.zeros(node_n)
    aic_comps["single_winner_ids"] = np.ones(node_n)*-1
    aic_comps["sig_params"] = np.zeros((node_n,len(stat_conds)))
    aic_comps["confint_params"] = np.zeros((node_n,len(stat_conds),2))
    aic_comps["simp_params"] = np.zeros(node_n)
    aic_comps["simp_confint_params"] = np.zeros((node_n,2))
    for n_idx in range(node_n):
        aic_array = np.array([aics[mod][n_idx] for mod in models])
        aic_comps["aics"][n_idx,] = aic_array
        aic_prob = np.exp((aic_array.min()-aic_array)/2)
        aic_comps["probs"][n_idx,] = aic_prob
        aic_order = np.argsort(aic_prob)
        aic_comps["order"][n_idx,] = aic_order
        aic_comps["winner_ids"][n_idx] = np.where(aic_order==len(models)-1)[0][0]
        aic_comps["winner_margin"][n_idx] = np.sort(aic_array)[1] - aic_array.min()
        aic_threshed = aic_prob.copy()
        aic_threshed[aic_threshed<threshold] = 0
        aic_threshed[aic_threshed>0] = 1
        aic_comps["threshed"][n_idx,] = aic_threshed
        if sum(aic_threshed) == 1:
            aic_comps["single_winners"][n_idx] = 1
            aic_comps["single_winner_ids"][n_idx] = aic_comps["winner_ids"][n_idx]
            if aic_comps["single_winner_ids"][n_idx] == 2: # if the best model was "cond," than find out which conditions were significantly different than rest
                for stat_cond_idx,stat_cond in enumerate(stat_conds):
                    if aics_pvals["cond"][n_idx][stat_cond] < cond_threshold:
                        aic_comps["sig_params"][n_idx][stat_cond_idx] = aics_params["cond"][n_idx][stat_cond]
                        aic_comps["confint_params"][n_idx][stat_cond_idx] = (aics_confint["cond"][n_idx].loc[stat_cond][0], aics_confint["cond"][n_idx].loc[stat_cond][1])
        if np.array_equal(aic_comps["threshed"][n_idx], np.array([0,1,1])):
            aic_comps["dual_winners"][n_idx] = 1
            aic_comps["simp_params"][n_idx] = aics_params["simple"][n_idx]["C(Block, Treatment('rest'))[T.task]"]
            aic_comps["simp_confint_params"][n_idx] = (aics_confint["simple"][n_idx].at["C(Block, Treatment('rest'))[T.task]",0],
                                                  aics_confint["simple"][n_idx].at["C(Block, Treatment('rest'))[T.task]",1])

    with open("{}{}/aic.pickle".format(proc_dir,band), "wb") as f:
        pickle.dump(aic_comps,f)
else:
    with open("{}{}/aic.pickle".format(proc_dir,band), "rb") as f:
        aic_comps = pickle.load(f)

brains = []

# this_rgba = np.zeros((len(labels), 4))
# inds = np.where(aic_comps["single_winner_ids"]==0)[0]
# if len(inds):
#     this_rgba[inds,] = np.array([1,0,0,])
#     brains.append(plot_rgba(this_rgba, labels, parc, lup_title="Null superior"))
#
# this_rgba = np.zeros((len(labels), 4))
# inds = np.where(aic_comps["single_winner_ids"]==1)[0]
# if len(inds):
#     this_rgba[inds,] = np.array([0,1,0,1])
#     brains.append(plot_rgba(this_rgba, labels, parc, lup_title="Simple superior"))
#
# this_rgba = np.zeros((len(labels), 4))
# inds = np.where(aic_comps["single_winner_ids"]==2)[0]
# if len(inds):
#     this_rgba[inds,] = np.array([0,0,1,1])
#     #brains.append(plot_rgba(this_rgba, labels, parc))
#     brains.append(plot_rgba(this_rgba, labels, parc, lup_title="Cond superior"))
#
# this_rgba = np.zeros((len(labels), 4))
# inds = np.where(aic_comps["dual_winners"])[0]
# if len(inds):
#     this_rgba[inds,] = np.array([1,1,0,1])
#     brains.append(plot_rgba(this_rgba, labels, parc, lup_title="Only Null rejected"))

brains.append(plot_parc_compare("aparc", "RegionGrowing_70"))
breakpoint()

vecs = aic_comps["simp_params"].copy()
vecs_abs = abs(vecs)
va_min, va_max, = vecs_abs[vecs_abs!=0].min(), vecs_abs.max()
vec_norm = ((vecs_abs - va_min) / (va_max - va_min)) * np.sign(vecs)
this_rgba = np.zeros((len(labels), 4))
this_rgba[vec_norm>0,0] = 1
this_rgba[vec_norm<0,2] = 1
this_rgba[vec_norm!=0,3] = np.abs(vec_norm[vec_norm!=0])
brains.append(plot_rgba(this_rgba, labels, parc,
                        lup_title="Task power change"))
write_brain_image("task_power", views, brains[-1], dir="../images/")

# changes from rest for each individual task
vecs = aic_comps["sig_params"].copy()
vecs_abs = abs(vecs)
va_min, va_max, = vecs_abs[vecs_abs!=0].min(), vecs_abs.max()
vec_norms = ((vecs_abs - va_min) / (va_max - va_min)) * np.sign(vecs)

inds = np.where(aic_comps["single_winner_ids"]==2)[0]

# audio
this_rgba = np.zeros((len(labels), 4))
vec_norm = vec_norms[:,0]
this_rgba[vec_norm>0,0] = 1
this_rgba[vec_norm<0,2] = 1
this_rgba[vec_norm!=0,3] = np.abs(vec_norm[vec_norm!=0])
brains.append(plot_rgba(this_rgba, labels, parc, lup_title="Audio power change"))
write_brain_image("audio_power", views, brains[-1], dir="../images/")
# visual
this_rgba = np.zeros((len(labels), 4))
vec_norm = vec_norms[:,1]
this_rgba[vec_norm>0,0] = 1
this_rgba[vec_norm<0,2] = 1
this_rgba[vec_norm!=0,3] = np.abs(vec_norm[vec_norm!=0])
brains.append(plot_rgba(this_rgba, labels, parc, lup_title="Visual power change"))
write_brain_image("visual_power", views, brains[-1], dir="../images/")

# visselten
this_rgba = np.zeros((len(labels), 4))
vec_norm = vec_norms[:,2]
this_rgba[vec_norm>0,0] = 1
this_rgba[vec_norm<0,2] = 1
this_rgba[vec_norm!=0,3] = np.abs(vec_norm[vec_norm!=0])
brains.append(plot_rgba(this_rgba, labels, parc, lup_title="Visselten power change"))
write_brain_image("visselten_power", views, brains[-1], dir="../images/")

# rainbow
this_rgba = np.zeros((len(labels), 4))
this_rgba[:,:3] = vec_norms
vec_norms = np.linalg.norm(vecs, axis=1)
vec_norms = (vec_norms - vec_norms[vec_norms!=0].min()) / \
            (vec_norms.max() - vec_norms[vec_norms!=0].min())
this_rgba[:,3] = vec_norms
brains.append(plot_rgba(abs(this_rgba), labels, parc, lup_title="Rainbow"))
brains[-1]._renderer.plotter.add_legend([["Audio","r"],["Visual","g"],
                                         ["Visselten","b"]], bcolor=(0,0,0))
write_brain_image("rainbow_power", views, brains[-1], dir="../images/")
