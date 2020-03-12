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
    def __init__(self,subjs,conds,wavs,fmax,f_range):
        self.subjs = subjs
        self.conds = conds
        self.wavs = wavs
        self.fmax = fmax
        self.f_range = f_range
        self.linemarkers = None
        self.sub = None
        self.table = {}

    def go(self):
        self.epos = []
        i = 0
        for sub in self.subjs:
            temp_epos = []
            all_bads = []
            for cond_idx,cond in enumerate(self.conds):
                for wav_idx,wav in enumerate(self.wavs):
                    temp_epos.append(mne.read_epochs("{dir}nc_{sub}_{cond}_{wav}_hand-epo.fif".format(dir=proc_dir,sub=sub,cond=cond,wav=wav)))
                    all_bads += temp_epos[-1].info["bads"]
            # merge epochs together here
            for x in temp_epos:
                x.info["bads"] = all_bads
                x.info["dev_head_t"] = temp_epos[0].info["dev_head_t"]
            self.epos.append(mne.concatenate_epochs(temp_epos))
            i += 1
        plt.close('all')
        self.fig, self.axes = plt.subplots(7,4)
        self.axes = [a for sublist in self.axes for a in sublist]
        self.draw()
    def draw(self):
        gen_min, gen_max = np.inf, 0
        lines, cols, epo_idxs, l_idxs, vlines, all_freqs = [], [], [], [], [], []
        for epo_idx,epo in enumerate(self.epos):
            start = np.min(epo.events[:,0])
            finish = np.max(epo.events[:,0]) + epo.times[-1]+epo.info["sfreq"]
            psds, freqs = mne.time_frequency.psd_multitaper(epo,fmax=self.fmax,adaptive=True,
                                                            n_jobs=8, bandwidth=2)
            psd = np.mean(psds,axis=1)
            gen_min = np.min(psd) if np.min(psd)<gen_min else gen_min
            gen_max = np.max(psd) if np.max(psd)>gen_max else gen_max
            for l_idx in range(psd.shape[0]):
                lines.append(self.axes[epo_idx].plot(freqs,psd[l_idx,],
                             color="blue",linewidth=2,alpha=0.2)[0])
                epo_idxs.append(epo_idx)
                l_idxs.append(l_idx)
            self.axes[epo_idx].set_title(self.subjs[epo_idx])
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
            ax.set_ylim(gen_min,gen_max/2)

    def save(self):
        with open("peak_freq_table","wb") as f:
            pickle.dump(self.table,f)

    def show_file(self):
        print("Current subject: "+self.sub)

proc_dir = "../proc/"
conds  = ["audio", "visual", "visselten"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]

# ATT_33,29,18 had no peak except a very high, low frequency
cyc = Cycler(subjs,conds,wavs,fmax=40,f_range=(6,15))
