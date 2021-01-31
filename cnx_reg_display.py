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
top_cnx = 75
bot_cnx = None
var_base = "C(Block, Treatment('rest'))" # stem of the condition names in statsmodels format
models = ["null","simple","cond"]
conds = ["rest","audio","visual","visselten","zaehlen"]
stat_conds = [var_base+"[T."+cond+"]" for cond in conds[1:]] # convert simple cond names to statsmodels cond names


regs = {"left M1 superior":"L3395-lh", "left M1 central":"L3969-lh",
        "left M1 dorsal":"L8143_L7523-lh","left sup-parietal posterior":"L4557-lh",
        "left S1 superior0":"L7491_L4557-lh","left S1 superior1":"L8143-lh",
        "left V1":"L2340_L1933-lh","left DS 0":"L2340-lh","left DS 1":"L2685-lh",
        "left orbito-frontal posterior":"L6698_L1154-lh","left DS 2":"L928-lh",
        "left VS 0":"L10017-lh","left VS 1":"L7097_L5106-lh",
        "left VS 2":"L7097_L4359-lh","left VS 3":"L5511_L4359-lh",
        "left VS 4":"L7049-lh","left A1":"L2235-lh",
        "left A1 alt":"L7755-lh","left orbito-frontal anterior":"L9249_L6698-lh",
        "left SMG inferior":"L5037-lh","left SMG superior":"L7491_L5037-lh"}

m1_complex = {"name":"M1 Complex","areas":("left M1 central","left M1 superior")}
sp_complex = {"name":"SP Complex","areas":("left sup-parietal posterior", "left S1 superior0")}
cnx = [("left A1",m1_complex),("left VS 0",m1_complex),
       ("left VS 1",m1_complex),("left VS 2",m1_complex),
       ("left VS 3",m1_complex),("left VS 4",m1_complex),
       ("left V1",m1_complex),("left orbito-frontal posterior",m1_complex),
       ("left A1",sp_complex),("left VS 0",sp_complex),
       ("left VS 1",sp_complex),("left VS 2",sp_complex),
       ("left VS 3",sp_complex),("left VS 4",sp_complex),
       ("left V1",sp_complex),("left orbito-frontal posterior",sp_complex),
       ("left DS 0",m1_complex),("left DS 1",m1_complex),
       ("left DS 2",m1_complex),("left DS 0",sp_complex),
       ("left DS 1",sp_complex),("left DS 2",sp_complex),
       ("left SMG inferior",sp_complex),("left SMG inferior",m1_complex),
       ("left SMG superior",sp_complex),("left SMG superior",m1_complex),
       ("left A1 alt",sp_complex),("left A1 alt",m1_complex),
       ("left orbito-frontal anterior",m1_complex),
       ("left orbito-frontal anterior",sp_complex)]

with open("{}{}/aic.pickle".format(proc_dir,band), "rb") as f:
    aic_comps = pickle.load(f)

# plt.hist(aic_comps["single_winner_ids"])
# plt.title("Single winner IDs")

triu_inds = np.triu_indices(mat_n, k=1)
cnx_masks = {mod:np.zeros((mat_n,mat_n)) for mod in models}
cnx_params = {stat_cond:np.zeros((mat_n,mat_n)) for stat_cond in stat_conds}
cnx_confint = {stat_cond:np.zeros((mat_n,mat_n,2)) for stat_cond in stat_conds}
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
                    cnx_params[stat_cond][triu_inds[1][n_idx],triu_inds[0][n_idx]] = -1 * aic_comps["sig_params"][n_idx][stat_cond_idx]
                    cnx_confint[stat_cond][triu_inds[0][n_idx],triu_inds[1][n_idx]] = aic_comps["confint_params"][n_idx][stat_cond_idx]
                    cnx_confint[stat_cond][triu_inds[1][n_idx],triu_inds[0][n_idx],] = -1 * np.flip(aic_comps["confint_params"][n_idx][stat_cond_idx])

all_params = np.abs(np.array([cnx_params[stat_cond] for stat_cond in stat_conds]).flatten())
all_params.sort()
alpha_max, alpha_min = all_params[-1:], all_params[-top_cnx].min()
alpha_max, alpha_min = None, None
params_brains = []
# for stat_cond,cond in zip(stat_conds,["audio","visual","visselten","zaehlen"]):
#     params_brains.append(plot_directed_cnx(cnx_params[stat_cond],labels,parc,
#                          alpha_min=alpha_min,alpha_max=alpha_max,
#                          ldown_title=cond, top_cnx=top_cnx))

bar_width = 0.35
x = np.arange(len(stat_conds))
xs = [x-bar_width/2,x+bar_width/2]
for c in cnx:
    plt.figure()
    if type(c[0]) == dict:
        plt.title("{} - {}".format(c[0]["name"],c[1]))
        for this_x,c0 in zip(xs,c[0]["areas"]):
            reg_inds = (label_names.index(regs[c0]),label_names.index(regs[c[1]]))
            betas = []
            confs = []
            for sc in stat_conds:
                betas.append(cnx_params[sc][reg_inds[0],reg_inds[1]])
                confs.append(cnx_confint[sc][reg_inds[0],reg_inds[1],])
            betas, confs = np.array(betas), np.array(confs).T
            confs -= betas
            plt.bar(this_x,np.array(betas),bar_width,yerr=np.abs(confs),label=c0)
    elif type(c[1]) == dict:
        plt.title("{} - {}".format(c[0],c[1]["name"]))
        for this_x, c1 in zip(xs, c[1]["areas"]):
            reg_inds = (label_names.index(regs[c[0]]),label_names.index(regs[c1]))
            betas = []
            confs = []
            for sc in stat_conds:
                betas.append(cnx_params[sc][reg_inds[0],reg_inds[1]])
                confs.append(cnx_confint[sc][reg_inds[0],reg_inds[1],])
            betas, confs = np.array(betas), np.array(confs).T
            confs -= betas
            plt.bar(this_x,np.array(betas),bar_width,yerr=np.abs(confs),label=c1)
    else:
        plt.title("{} - {}".format(c[0],c[1]))
        reg_inds = (label_names.index(regs[c[0]]),label_names.index(regs[c[1]]))
        betas = []
        confs = []
        for sc in stat_conds:
            betas.append(cnx_params[sc][reg_inds[0],reg_inds[1]])
            confs.append(cnx_confint[sc][reg_inds[0],reg_inds[1],])
        betas, confs = np.array(betas), np.array(confs).T
        confs -= betas
        plt.bar(x,np.array(betas),yerr=np.abs(confs))

    ax = plt.gca()
    ax.legend()
    ax.set_xticks(x)
    plt.gca().set_xticklabels(conds[1:])
    plt.gca().set_ylim((0,0.02))
