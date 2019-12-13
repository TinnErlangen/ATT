import mne
import numpy as np

mri_key = {"KIL13":"ATT_10","ALC81":"ATT_11","EAM11":"ATT_19","ENR41":"ATT_18",
           "NAG_83":"ATT_36","PAG48":"ATT_21","SAG13":"ATT_20","HIU14":"ATT_23",
           "KIL72":"ATT_25","FOT12":"ATT_28","KOI12":"ATT_16","BLE94":"ATT_29",
           "DEN59":"ATT_26","WOO07":"ATT_12","DIU11":"ATT_34","BII41":"ATT_31",
           "Mun79":"ATT_35","ATT_37_fsaverage":"ATT_37",
           "ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}
mri_key = {"ATT_24_fsaverage":"ATT_24","TGH11":"ATT_14","FIN23":"ATT_17",
           "GIZ04":"ATT_13","BAI97":"ATT_22","WAL70":"ATT_33",
           "ATT_15_fsaverage":"ATT_15"}

#FAO18, WKI71, BRA52 had a defective (?) MRI and fsaverage was used instead
# ATT_30/KER27, ATT_27, ATT_32/EAM67   excluded for too much head movement between blocks

runs = ["rest","audio","visselten","visual","zaehlen"]
#runs = ["rest"]
subjects_dir = "/home/jeff/freesurfer/subjects/"
proc_dir = "/home/jeff/ATT_dat/proc/"
thresh = 0.5

for k,v in mri_key.items():
    trans = "../proc/"+k+"-trans.fif"
    src = mne.setup_source_space(k,surface="white",
                                 subjects_dir=subjects_dir,n_jobs=4)
    src.save("../proc/"+v+"-src.fif", overwrite=True)
    bem_model = mne.make_bem_model(k, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(bem_model)
    mne.write_bem_solution("../proc/"+v+"-bem.fif",bem)
    # src.plot()
    # mne.viz.plot_bem(subject=k, subjects_dir=subjects_dir,
    #              brain_surfaces='white', src=src, orientation='coronal')

    src = mne.read_source_spaces("../proc/"+v+"-src.fif")
    bem = mne.read_bem_solution("../proc/"+v+"-bem.fif")
    fwds = []
    for run in runs:
        epo = mne.read_epochs("{dir}nc_{sub}_{run}_hand-epo.fif".format(
                              dir=proc_dir,sub=v,run=run))
        fwd = mne.make_forward_solution(epo.info, trans=trans, src=src, bem=bem,
                                        meg=True, mindist=5.0, n_jobs=8)
        fwds.append(fwd)
    avg_fwd = mne.average_forward_solutions(fwds)
    del fwds
    mne.write_forward_solution("{dir}nc_{sub}-fwd.fif".format(
                               dir=proc_dir,sub=v), avg_fwd,
                               overwrite=True)
    sen = mne.sensitivity_map(avg_fwd,ch_type="mag",mode="fixed")
    sen.data[sen.data<thresh] = 0
    sens_lh, sens_rh = mne.stc_to_label(sen,src=src,subjects_dir=subjects_dir,smooth=True)
    sens_lh.save("{dir}nc_{sub}".format(dir=proc_dir,sub=v,run=run))
    sens_rh.save("{dir}nc_{sub}".format(dir=proc_dir,sub=v,run=run))
