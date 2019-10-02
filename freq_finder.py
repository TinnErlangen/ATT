import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()


class LineMarker():
    def __init__(self,line):
        self.lines = lines
        self.cidmouse = self.line.figure.canvas.mpl_connect("button_press_event", self.on_click)
    def on_click(self, event):

        self.line.set_data(event.xdata, self.line.get_data()[1])
        self.line.figure.canvas.draw()
        print(event.inaxes)
    def get_freq(self):
        return self.line.get_data()[0]

class Cycler():
    def __init__(self,subjs,conds,fmax):
        self.subjs = subjs
        self.conds = conds
        self.fmax = fmax
    def go(self):
        self.epos = []
        self.files = []
        self.sub = self.subjs.pop()
        for cond_idx,cond in enumerate(self.conds):
            self.files.append("{dir}nc_{sub}_{cond}-epo.fif".format(dir=proc_dir,sub=self.sub,cond=cond))
            self.epos.append(mne.read_epochs(self.files[-1]))
        plt.close('all')
        self.fig, self.axes = plt.subplots(5,1)
        self.draw()
    def draw(self):
        gen_min, gen_max = np.inf, 0
        lines, cols, epo_idxs, l_idxs, vlines = [], [], [], [], [], []
        for epo_idx,epo in enumerate(self.epos):
            start = np.min(epo.events[:,0])
            finish = np.max(epo.events[:,0]) + epo.times[-1]+epo.info["sfreq"]
            psds, freqs = mne.time_frequency.psd_multitaper(epo,fmax=self.fmax,adaptive=True,
                                                            n_jobs=8, bandwidth=2)
            psd = np.mean(psds,axis=1)
            gen_min = np.min(psd) if np.min(psd)<gen_min else gen_min
            gen_max = np.max(psd) if np.max(psd)>gen_max else gen_max
            for l_idx in range(psd.shape[0]):
                col = (epo.events[l_idx,0]-start)/(finish-start)
                col = (1-col,0,col)
                cols.append(col)
                lines.append(self.axes[epo_idx].plot(psd[l_idx,],color=col,linewidth=2,
                                   alpha=0.2)[0])
                ticks = list(range(0,len(freqs),5))
                labels = [np.round(freqs[x],decimals=1) for x in ticks]
                self.axes[epo_idx].set_xticks(ticks)
                self.axes[epo_idx].set_xticklabels(labels)
                epo_idxs.append(epo_idx)
                l_idxs.append(l_idx)
            mean_psd = np.mean(psd,axis=0)
            freq_max = mean_psd.max()
            self.axes[epo_idx].plot(mean_psd,color="black",linewidth=2)
            vlines.append(self.axes[epo_idx].axvline(x=freq_max))
        self.linemarkers = linemarkers(LineMarkers(vlines))

        for ax in self.axes:
            ax.set_ylim(gen_min,gen_max)
    def save(self):
        epos = self.epos
        # find epo with lowest number of epochs
        low_epo_idx = np.argmin([len(x) for x in epos[1:]])+1
        eves = epos[low_epo_idx].events[:,2]%10
        counts = [np.sum(eves==x) for x in range(1,5)]
        # make new shuffled epoch objects with same number as the one with lowest
        new_epos = [epos[0][np.sort(np.random.permutation(np.arange(
        len(epos[0])))[:np.sum(counts)])]] # resting state first
        for e in epos[1:]: # exclude resting state
            temp_epos = []
            for c in range(4):
                idxs = np.where(e.events[:,2]%10==c+1)[0]
                idxs_perm = np.sort(np.random.permutation(idxs)[:counts[c]])
                temp_epos.append(e[idxs_perm])
            new_epos.append(mne.concatenate_epochs(temp_epos))
        for ne_idx,ne in enumerate(new_epos):
            ne.save(epos[ne_idx].filename[:-8]+"_hand-epo.fif")
    def show_file(self):
        print("Current subject: "+self.sub)

proc_dir = "../proc/"
conds  = ["rest", "audio", "visual", "visselten", "zaehlen"]
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29",
         "ATT_30", "ATT_31", "ATT_32", "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
cyc = Cycler(subjs,conds,fmax=30)
