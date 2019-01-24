import mne
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

base_dir ="../"
proc_dir = base_dir+"proc/"
subjs = ["ATT_10","ATT_11","ATT_12","ATT_13","ATT_14","ATT_15","ATT_16","ATT_17","ATT_18","ATT_19"]
subjs = ["NEM_16"]
runs = ["rest","audio","visselten","visual","zaehlen"]
runs = [str(x+1) for x in range(5)]
runs = ["3"]

filelist = []
for sub in subjs:
    for run in runs:
        filelist.append(["{dir}nc_{sub}_{run}_hand-raw.fif".format(dir=proc_dir,sub=sub,run=run),
        "{dir}nc_{sub}_{run}_hand_ref-ica.fif".format(dir=proc_dir,sub=sub,run=run),
        "{dir}nc_{sub}_{run}_hand_meg-ica.fif".format(dir=proc_dir,sub=sub,run=run)])

class Cycler():

    def __init__(self,filelist):
        self.filelist = filelist

    def go(self):
        self.fn = self.filelist.pop()
        self.raw = mne.io.Raw(self.fn[0],preload=True)
        self.icaref = mne.preprocessing.read_ica(self.fn[1])
        self.icameg = mne.preprocessing.read_ica(self.fn[2])

        refcomps = self.icaref.get_sources(self.raw)
        for c in refcomps.ch_names:
            refcomps.rename_channels({c:"REF_"+c})
        self.raw.add_channels([refcomps])
        self.comps = []

        # self.icameg.plot_components(picks=list(range(20)))
        # self.icameg.plot_sources(self.raw)
        # self.icaref.plot_sources(self.raw)
        self.raw.plot(n_channels=64,duration=120,scalings="auto")
        self.raw.plot_psd(fmax=40)


    def plot_props(self,props):
        self.icameg.plot_properties(self.raw,props)

    def show_file(self):
        print("Current raw file: "+self.fn[0])

    def without(self,comps,fmax=40):
        test = self.raw.copy()
        test.load_data()
        test = self.icameg.apply(test,exclude=comps)
        test.plot_psd(fmax=fmax)
        test.plot(duration=30,n_channels=30)
        self.test = test

    def identify_ref(self,threshold=3.5):
        ref_inds, scores = self.icameg.find_bads_ref(self.raw,threshold=threshold)
        if ref_inds:
            self.icameg.plot_scores(scores, exclude=ref_inds)
            print(ref_inds)
            #self.icameg.plot_properties(self.raw,ref_inds)
            self.comps += ref_inds


    def save(self,comps=None):
        if comps:
            self.icameg.apply(self.raw,exclude=self.comps+comps).save(self.fn[0][:-8]+"_ica-epo.fif")
        else:
            print("No components applied, saving anyway for consistency.")
            self.raw.save(self.fn[0][:-8]+"_ica1-epo.fif")

cyc = Cycler(filelist)
