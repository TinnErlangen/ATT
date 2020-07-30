import numpy as np
from cnx_utils import plot_undirected_cnx, plot_directed_cnx, load_sparse
import mne
import pickle
from collections import Counter
import matplotlib.pyplot as plt
plt.ion()

proc_dir = "/home/jeff/ATT_dat/lmm/"
sps_dir = "/home/jeff/ATT_dat/proc/"
band = "alpha_1"
node_n = 2415
parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
label_names = [l.name for l in labels]
mat_n = len(labels)
top_cnx = 100
bot_cnx = None
var_base = "C(Block, Treatment('rest'))" # stem of the condition names in statsmodels format
models = ["null","simple","cond"]
conds = ["rest","audio","visual","visselten","zaehlen"]
stat_conds = [var_base+"[T."+cond+"]" for cond in conds[1:]] # convert simple cond names to statsmodels cond names


regs = {"left M1 central":"L3395-lh", "left M1 inferior":"L3969-lh",
        "left M1 superior":"L8143_L7523-lh","left sup-parietal":"L4557-lh",
        "left V1":"L2340_L1933-lh","left DS 0":"L2340-lh","left DS 1":"L2685-lh",
        "left DS 2":"L928-lh","left VS 0":"L10017-lh","left VS 1":"L7097_5106-lh",
        "left VS 2":"L7097_L4359-lh","left VS 3":"L5511_L4359-lh",
        "left VS 4":"L7049-lh", "left S1":"L7491_L4557-lh","left A1":"L2235-lh",
        "left orbito-frontal":"L9249_L6698-lh",
        "left SMG inferior":"L5037-lh","left SMG superior":"L7491_L5037-lh"}

cnx = [("left M1 central","left A1"),("left M1 central","left V1")]

with open("{}{}/aic.pickle".format(proc_dir,band), "rb") as f:
    aic_comps = pickle.load(f)

plt.hist(aic_comps["single_winner_ids"])
plt.title("Single winner IDs")

triu_inds = np.triu_indices(mat_n, k=1)
cnx_masks = {mod:np.zeros((mat_n,mat_n)) for mod in models}
cnx_params = {stat_cond:np.zeros((mat_n,mat_n)) for stat_cond in stat_conds}
brains = []
colors = [(1,0,0),(0,1,0),(0,0,1)]
models, colors = ["cond"], [(0,0,1)]
for color, mod in zip(colors, models):
    mod_idx = aic_comps["models"].index(mod)
    for n_idx in range(node_n):
        if aic_comps["single_winner_ids"][n_idx] == mod_idx:
            cnx_masks[mod][triu_inds[0][n_idx],triu_inds[1][n_idx]] = 1
        if mod == "cond":
            for stat_cond_idx,stat_cond in enumerate(stat_conds):
                if aic_comps["sig_params"][n_idx][stat_cond_idx]:
                    cnx_params[stat_cond][triu_inds[0][n_idx],triu_inds[1][n_idx]] = aic_comps["sig_params"][n_idx][stat_cond_idx]

all_params = np.abs(np.array([cnx_params[stat_cond] for stat_cond in stat_conds]).flatten())
all_params.sort()
alpha_max, alpha_min = all_params[-1:], all_params[-top_cnx].min()
alpha_max, alpha_min = None, None
params_brains = []
# for stat_cond,cond in zip(stat_conds,["audio","visual","visselten","zaehlen"]):
#     params_brains.append(plot_directed_cnx(cnx_params[stat_cond],labels,parc,
#                          alpha_min=alpha_min,alpha_max=alpha_max,
#                          ldown_title=cond, top_cnx=top_cnx))

for c in cnx:
    plt.figure()
    plt.title("{} - {}".format(c[0],c[1]))
    reg_inds = np.sort((label_names.index(regs[c[0]]),label_names.index(regs[c[1]])))
    betas = []
    for sc in stat_conds:
        betas.append(cnx_params[sc][reg_inds[0],reg_inds[1]])
    plt.bar(np.arange(len(stat_conds)),np.array(betas))
