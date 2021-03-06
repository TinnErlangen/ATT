import mne
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

class ILine:
    def __init__(self,lines,cols,epo_ids,trial_ids):
        self.lines = lines
        self.cols = cols
        self.epo_ids = epo_ids
        self.trial_ids = trial_ids
        self.ons = np.ones(len(lines),dtype=np.bool)
        self.connect()
    def draw(self):
        self.lines[0].figure.canvas.draw()
    def connect(self):
        self.cidmouse = self.lines[0].figure.canvas.mpl_connect("button_press_event", self.on_click)
    def on_click(self, event):
        for l_idx,l in enumerate(self.lines):
            if l.contains(event)[0]:
                print("Trial {}  Epoch {}".format(self.trial_ids[l_idx],self.epo_ids[l_idx]))
                if self.ons[l_idx]:
                    self.ons[l_idx] = False
                    self.lines[l_idx].set_color((0.8,0.8,0.8))
                else:
                    self.ons[l_idx] = True
                    self.lines[l_idx].set_color(self.cols[l_idx])

class Cycler():
    def __init__(self,subjs,conds,fmax,subplot):
        self.subjs = subjs
        self.conds = conds
        self.fmax = fmax
        self.subplot = subplot
    def go(self):
        self.epos = []
        self.files = []
        self.sub = self.subjs.pop()
        for cond_idx,cond in enumerate(self.conds):
            self.files.append("{dir}nc_{sub}_{cond}-epo.fif".format(dir=proc_dir,sub=self.sub,cond=cond))
            self.epos.append(mne.read_epochs(self.files[-1]))
        plt.close('all')
        self.fig, axes = plt.subplots(*self.subplot)
        self.axes = [item for ax in axes for item in ax]
        self.draw()
    def draw(self):

        gen_min, gen_max = np.inf, 0
        lines, cols, epo_idxs, l_idxs = [], [], [], []
        for epo_idx,epo in enumerate(self.epos):
            start = np.min(epo.events[:,0])
            finish = np.max(epo.events[:,0]) + epo.times[-1]+epo.info["sfreq"]
            psds, freqs = mne.time_frequency.psd_multitaper(epo,fmax=self.fmax,adaptive=True,n_jobs=2)
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
            self.axes[epo_idx].plot(np.mean(psd,axis=0),color="black",linewidth=2)
        self.ilines = ILine(lines,cols,np.array(epo_idxs),np.array(l_idxs))
        self.ilines.connect()
        for ax in self.axes:
            ax.set_ylim(gen_min,gen_max)
    def trim(self,draw=False):
        for epo_idx,epo in enumerate(self.epos):
            bad_trials_idx = (self.ilines.epo_ids==epo_idx) & ~self.ilines.ons
            epo.drop(self.ilines.trial_ids[bad_trials_idx])
        if draw:
            plt.close('all')
            self.fig, self.axes = plt.subplots(5,1)
            self.draw()
    def save(self):
        epos = self.epos
        mne.epochs.equalize_epoch_counts(epos)
        for e_idx,e in enumerate(epos):
            e.save(epos[e_idx].filename[:-8]+"_hand-epo.fif", overwrite=True)
    def show_file(self):
        print("Current subject: "+self.sub)

proc_dir = "../proc/"
conds  = ["audio", "visual", "visselten"]
conds  = ["zaehlen"]
wavs = ["4000fftf", "4000cheby", "7000Hz", "4000Hz"]
all_conds = [c+"_"+w for w in wavs for c in conds]
#all_conds += ["rest"]
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29",
         "ATT_30", "ATT_31", "ATT_32", "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]
# subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
#          "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
#          "ATT_24"]
cyc = Cycler(subjs,all_conds,30,(3,4))
