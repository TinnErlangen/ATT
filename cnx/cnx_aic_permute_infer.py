import numpy as np
import pickle

proc_dir = "/home/jev/hdd/ATT_dat/proc/"
band = "alpha_1"
file_num = 10
per_file = 100
perm_n = file_num*per_file
node_n = 2415
models = ["null","simple","cond"]
threshold = 0.001
# 
# with open("{}{}/aicnoZ.pickle".format(proc_dir,band), "rb") as f:
#     aic_comps = pickle.load(f)

aics = np.zeros((len(models),node_n,perm_n))
for f_idx, pn_idx in enumerate(range(0,perm_n,per_file)):
    with open("{}{}/perm_aic_{}.pickle".format(proc_dir,band,f_idx),"rb") as f:
        aic = pickle.load(f)
    for mod_idx,mod in enumerate(models):
        aics[mod_idx,:,pn_idx:pn_idx+per_file] = aic[mod]
aics[0,] = np.broadcast_to(aics[0,:,0],(perm_n,node_n)).T

p_vals = np.zeros(aics.shape)

counter = {"null":[], "simple":[], "cond":[]}
for p_idx in range(perm_n):
    for v in counter.values():
        v.append(0)
    for n_idx in range(node_n):
        aic_array = aics[:,n_idx,p_idx]
        p_vals[:,n_idx,p_idx] = np.exp((aic_array.min()-aic_array)/2)
        threshed = p_vals[:,n_idx,p_idx].copy()
        threshed[threshed < threshold] = 0
        threshed[threshed!=0] = 1
        if np.array_equal(threshed, np.array([0,1,1])) or np.array_equal(threshed, np.array([0,1,0])):
            counter["simple"][-1] += 1
        elif np.array_equal(threshed, np.array([0,0,1])):
            counter["cond"][-1] += 1
        else:
            counter["null"][-1] += 1
