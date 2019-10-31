import mne
import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.ion()

class LineMarkers():
    def __init__(self,lines,freqs):
        self.lines = lines
        self.freqs = freqs
        self.axis_line = {}
        for line in lines:
            self.cidmouse = line.figure.canvas.mpl_connect("button_press_event", self.on_click)
            self.axis_line[line.axes] = line
    def on_click(self, event):
        if event.inaxes:
            line = self.axis_line[event.inaxes]
            line.set_data(event.xdata, line.get_data()[1])
            line.figure.canvas.draw()
    def get_freqs(self):
        freqs = []
        for line in self.lines:
            freqs.append(line.get_data()[0])
        return freqs

class Cycler():
    def __init__(self,subjs,conds,fmax,f_range):
        self.subjs = subjs
        self.conds = conds
        self.fmax = fmax
        self.f_range = f_range
        self.table = {"conditions":conds}
        self.linemarkers = None
        self.sub = None

    def go(self):
        if self.linemarkers:
            self.table[self.sub] = self.linemarkers.get_freqs()
        self.epos = []
        self.files = []
        self.sub = self.subjs.pop()
        for cond_idx,cond in enumerate(self.conds):
            self.files.append("{dir}nc_{sub}_{cond}_hand-epo.fif".format(dir=proc_dir,sub=self.sub,cond=cond))
            self.epos.append(mne.read_epochs(self.files[-1]))
        plt.close('all')
        self.fig, self.axes = plt.subplots(5,1)
        self.draw()
    def draw(self):
        gen_min, gen_max = np.inf, 0
        lines, cols, epo_idxs, l_idxs, vlines, all_freqs = [], [], [], [], [], []
        for epo_idx,epo in enumerate(self.epos):
            start = np.min(epo.events[:,0])
            finish = np.max(epo.events[:,0]) + epo.times[-1]+epo.info["sfreq"]
            psds, freqs = mne.time_frequency.psd_multitaper(epo,fmax=self.fmax,adaptive=True,
                                                            n_jobs=8, bandwidth=0.8)
            psd = np.mean(psds,axis=1)
            gen_min = np.min(psd) if np.min(psd)<gen_min else gen_min
            gen_max = np.max(psd) if np.max(psd)>gen_max else gen_max
            for l_idx in range(psd.shape[0]):
                col = (epo.events[l_idx,0]-start)/(finish-start)
                col = (1-col,0,col)
                cols.append(col)
                lines.append(self.axes[epo_idx].plot(freqs,psd[l_idx,],
                             color=col,linewidth=2,alpha=0.2)[0])
                epo_idxs.append(epo_idx)
                l_idxs.append(l_idx)
            mean_psd = np.mean(psd,axis=0)
            f_range_inds = (freqs>self.f_range[0]) & (freqs<self.f_range[1])
            f_idx_min = np.where(f_range_inds)[0].min()
            freq_max = freqs[mean_psd[...,f_range_inds].argmax()+f_idx_min]
            print(freq_max)
            self.axes[epo_idx].plot(freqs,mean_psd,color="black",linewidth=2)
            vlines.append(self.axes[epo_idx].axvline(freq_max))
            vlines[-1].set_data(freq_max, vlines[-1].get_data()[1])
            all_freqs.append(freqs)
        self.linemarkers = LineMarkers(vlines,all_freqs)
        for ax in self.axes:
            ax.set_ylim(gen_min,gen_max)

    def save(self):
        with open("peak_freq_table","wb") as f:
            pickle.dump(self.table,f)

    def show_file(self):
        print("Current subject: "+self.sub)

proc_dir = "../proc/"
conds  = ["rest", "audio", "visual", "visselten", "zaehlen"]
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29",
         "ATT_30", "ATT_31", "ATT_32", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]

# ATT_33,29,18 had no peak except a very high, low frequency
cyc = Cycler(subjs,conds,fmax=30,f_range=(7,14))
