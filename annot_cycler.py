import mne
import matplotlib.pyplot as plt

# This is a function which reduces the pain of hand-cleaning data. Run in
# interactive mode (i.e. python -i annot_cycler.py). Then use the go method (cyc.go())
# to move to the first file. Mark bad channels and sections as desired, then save
# (cyc.save()), and finally cyc.go() to move to the next file.

plt.ion()

base_dir ="../"
proc_dir = base_dir+"proc/"
subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17"]#, "ATT_18", "ATT_19"], "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         #"ATT_24", "ATT_25", "ATT_26", "ATT_27", "ATT_28", "ATT_29"]
runs = [str(x+1) for x in range(5)]
runs = ["1"]

filelist = []
for sub in subjs:
    for run in runs:
        filelist.append("{dir}nc_{sub}_{run}-raw.fif".format(dir=proc_dir,sub=sub,run=run))

class Cycler():

    def __init__(self,filelist):
        self.filelist = filelist

    def go(self):
        self.fn = self.filelist.pop()
        self.raw = mne.io.Raw(self.fn)
        self.raw.plot(duration=30,n_channels=280)

    def plot(self,n_channels=280,duration=30):
        self.raw.plot(duration=duration, n_channels=n_channels)

    def show_file(self):
        print("Current raw file: " + self.fn)

    def save(self):
        self.raw.save(self.fn[:-8]+"_hand-raw.fif",overwrite=True)
        if self.raw.annotations is not None:
            self.raw.annotations.save(self.fn[:-8]+"-annot.fif")
        if self.raw.info["bads"]:
            with open(self.fn[:-8]+"-badchans.txt", "w") as file:
                for b in self.raw.info["bads"]:
                    file.write(b+"\n")

cyc = Cycler(filelist)
