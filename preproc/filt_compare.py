import mne
import matplotlib.pyplot as plt

base_dir = "/media/hdd/jeff/ATT_dat/raw/eegamp_noi_2/"

runs = ["1","2","3","7","8","9"]
conditions = ["Leer","VP mit Coils"]
filts = ["mit online DC","mit online 0,1Hz","mit online 1,0Hz"]
filtfiles = ["c,rfDC","c,rfhp0.1Hz","c,rfhp1.0Hz"]
raws = {}

sub_idx = 1
abs_idx = 0
for cond_idx, cond in enumerate(conditions):
    for filt_idx, filt in enumerate(filts):
        condname = cond + " " + filt
        raws[condname] = mne.io.read_raw_bti(
        pdf_fname=base_dir+runs[abs_idx] + "/" + filtfiles[filt_idx],
        head_shape_fname=None,preload=True)
        raws[condname].info["bads"] =["MEG 188", "MEG 071", "MEG 220"]
        plt.subplot(len(conditions)*2,len(filts),sub_idx)
        raws[condname].plot_psd(fmax=16,ax=plt.gca())
        plt.gcf().axes[-2].set_title(condname)
        plt.gcf().axes[-2].set_ylim([0,70])
        sub_idx += 1
        abs_idx += 1
    sub_idx += 1
    condname = cond + " " + "DC mit offline 0,1Hz"
    raws[condname] = raws[cond + " " + "mit online DC"].copy()
    raws[condname].filter(0.1,None)
    plt.subplot(len(conditions)*2,len(filts),sub_idx)
    raws[condname].plot_psd(fmax=16,ax=plt.gca())
    plt.gcf().axes[-2].set_title(condname)
    plt.gcf().axes[-2].set_ylim([0,70])
    sub_idx += 1
    condname = cond + " " + "DC mit offline 1,0Hz"
    raws[condname] = raws[cond + " " + "mit online DC"].copy()
    raws[condname].filter(1,None)
    plt.subplot(len(conditions)*2,len(filts),sub_idx)
    raws[condname].plot_psd(fmax=16,ax=plt.gca())
    plt.gcf().axes[-2].set_title(condname)
    plt.gcf().axes[-2].set_ylim([0,70])
    sub_idx += 1
