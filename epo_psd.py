import mne
import numpy as np
import matplotlib.pyplot as plt

proc_dir = "../proc/"
conds  = ["rest", "audio", "visual", "visselten", "zaehlen"]
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29"]

sub = "ATT_20"
plt.ion()
fig = plt.figure()
for cond_idx,cond in enumerate(conds):
    filename = "{dir}nc_{sub}_{cond}-epo.fif".format(dir=proc_dir,sub=sub,cond=cond)
    epo = mne.read_epochs(filename)
    start = np.min(epo.events[:,0])
    finish = np.max(epo.events[:,0]) + epo.times[-1]+epo.info["sfreq"]
    psds, freqs = mne.time_frequency.psd_multitaper(epo,fmax=100,adaptive=True,
                                                    n_jobs=4)
    psd = np.mean(psds,axis=1)
    plt.subplot(5,1,cond_idx+1)
    for l_idx in range(psd.shape[0]):
        col = (epo.events[l_idx,0]-start)/(finish-start)
        print(col)
        plt.plot(psd[l_idx,:40],color=(1-col,0,col),linewidth=2,alpha=0.2)
