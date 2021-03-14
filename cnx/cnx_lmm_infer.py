from statsmodels.regression.mixed_linear_model import MixedLMResults
import numpy as np
import pickle
import mne
from cnx_utils import load_sparse, cnx_cluster, plot_directed_cnx
from scipy.stats import norm
from mayavi import mlab

proc_dir = "/home/jeff/ATT_dat/proc/"
band = "alpha_1"
indep_var = "None"
perm_count = 200
perm_num = 5
vec_mat_len = 2415
effect_names = ["C(Block, Treatment('rest'))[T.audio]",
                "C(Block, Treatment('rest'))[T.visselten]",
                 "C(Block, Treatment('rest'))[T.visual]",
                 "C(Block, Treatment('rest'))[T.zaehlen]"]
effects = ["audio","visselten","visual","zaehlen"]

# load up the main result
mod_fits = []
for vml in range(vec_mat_len):
    mod_fits.append(MixedLMResults.load("{}{}/reg70_lmm_{}.pickle".format(proc_dir,band,vml)))

# extract p and z values
eff_dict = {e:{"effect_name":en,"pvals":[],"zvals":[]} for e,en in zip(effects,effect_names)}
for effect in effects:
    for mf in mod_fits:
        pval = mf.pvalues.get(eff_dict[effect]["effect_name"])
        zval = mf.tvalues.get(eff_dict[effect]["effect_name"])
        eff_dict[effect]["pvals"].append(pval)
        eff_dict[effect]["zvals"].append(zval)
for k,v in eff_dict.items():
    v["pvals"] = np.array(v["pvals"])
    v["zvals"] = np.array(v["zvals"])

# put into clusters and weight them
for effect in effects:
    comp_f, out_edges = cnx_cluster(eff_dict[effect]["zvals"],
                                    eff_dict[effect]["pvals"],
                                    70)
    eff_dict[effect]["clusters"] = (comp_f,out_edges)

# # load up the permutations
# perms = []
# for perm_idx in range(perm_num):
#     perm = np.load("{}cnx_{}_{}_perm_{}_{}.npy".format(proc_dir, indep_var,
#                                                        band, perm_count,
#                                                        perm_idx))
#     perm = perm.reshape((perm.shape[0],perm_count,vec_mat_len))
#     perms.append(perm)
# perms = np.hstack(perms)
#
# # get distribution of null H cluster weights
# for eff_idx, eff in enumerate(effects):
#     perm = perms[eff_idx,]
#     perm_weights = []
#     for perm_idx in range(len(perm)):
#         zvals = perm[perm_idx,]
#         pvals = 2*norm.cdf(-np.abs(zvals))
#         comp_z, _ = cnx_cluster(zvals, pvals, 70)
#         perm_weights.append(np.array(comp_z).max())
#     perm_weights = np.array(perm_weights)
#     eff_dict[eff]["weight_thresh"] = np.quantile(perm_weights,0.95)
#
# with open("{}{}/cnx_results.pickle".format(proc_dir,band),"wb") as f:
#     pickle.dump(eff_dict,f)

# now load up dPTEs
with open("{}{}/cnx_results.pickle".format(proc_dir,band),"rb") as f:
    eff_dict = pickle.load(f)

parc = "RegionGrowing_70"
labels = mne.read_labels_from_annot("fsaverage",parc)
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_31", "ATT_33",
         "ATT_34", "ATT_35", "ATT_36", "ATT_37"]
top_cnx = 150

# average by subject, hold resting state separate because it was baseline
dPTEs = [[] for k in eff_dict.keys()]
rests = []
for sub in subjs:
    idx = 0
    for k in eff_dict.keys():
        dPTE = load_sparse("{}nc_{}_{}_dPTE_{}.sps".format(proc_dir, sub,
                                                           k, band))
        dPTEs[idx].append(dPTE.mean(axis=0))
        idx += 1
    dPTE = load_sparse("{}nc_{}_rest_dPTE_{}.sps".format(proc_dir, sub,
                                                         band))
    rests.append(dPTE.mean(axis=0))
dPTEs = np.mean(dPTEs,axis=1)
rest = np.mean(rests,axis=0)

# mask by significant edges
contrasts = {"audio":dPTEs[0]-rest,"visselten":dPTEs[1]-rest,
             "visual":dPTEs[2]-rest,"zaehlen":dPTEs[3]-rest,}
for k,v in contrasts.items():
    temp_dpte = np.zeros(v.shape)
    for edge in eff_dict[k]["clusters"][1][0]:
        temp_dpte[edge[0],edge[1]] = v[edge[0],edge[1]]
    contrasts[k] = temp_dpte

brains = []
for k,v in contrasts.items():
    brains.append(plot_directed_cnx(v,labels,parc,lup_title=k,top_cnx=top_cnx))
