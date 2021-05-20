import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.regression.mixed_linear_model import MixedLM
plt.ion()

df_laut = pd.read_pickle("../behave/laut")
df_ang = pd.read_pickle("../behave/ang")

blocks = ["audio","visual","visselten"]
exclude = ["zaehlen"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]

for excl in exclude:
    df_laut = df_laut.query("Block!='{}'".format(excl))
    df_ang = df_ang.query("Block!='{}'".format(excl))


plt.figure()
lauts_block = []
for block in blocks:
    lauts_block.append(df_laut.loc[df_laut["Block"]==block]["Laut"].values)
lauts_block = np.array(lauts_block)
lauts_block_mean = np.mean(lauts_block,axis=1)
sem = stats.sem(lauts_block,axis=1)
plt.bar(np.arange(len(blocks)),lauts_block_mean,yerr=sem,tick_label=blocks)
plt.title("Bewertung Laut, nach Block")

plt.figure()
lauts_wav = []
for wav in wavs:
    lauts_wav.append(df_laut.loc[df_laut["Wav"]==wav]["Laut"].values)
lauts_wav = np.array(lauts_wav)
lauts_wav_mean = np.mean(lauts_wav,axis=1)
sem = stats.sem(lauts_wav,axis=1)
plt.bar(np.arange(len(wavs)),lauts_wav_mean, yerr=sem, tick_label=wavs)
plt.title("Bewertung Laut, nach Ton")

plt.figure()
angs_block = []
for block in blocks:
    angs_block.append(df_ang.loc[df_ang["Block"]==block]["Angenehm"].values)
angs_block = np.array(angs_block)
angs_block_mean = np.mean(angs_block,axis=1)
sem = stats.sem(angs_block,axis=1)
plt.bar(np.arange(len(blocks)),angs_block_mean,yerr=sem,tick_label=blocks)
plt.title("Bewertung Angenehm, nach Block")

plt.figure()
angs_wav = []
for wav in wavs:
    angs_wav.append(df_ang.loc[df_ang["Wav"]==wav]["Angenehm"].values)
angs_wav = np.array(angs_wav)
angs_wav_mean = np.mean(angs_wav,axis=1)
sem = stats.sem(angs_wav,axis=1)
plt.bar(np.arange(len(wavs)),angs_wav_mean,yerr=sem,tick_label=wavs)
plt.title("Bewertung Angenehm, nach Ton")

fig, axes = plt.subplots(2,2)
plt.suptitle("Bewertung Laut")
axes = [ax for sublist in axes for ax in sublist]
for block_idx,block in enumerate(blocks):
    lauts = []
    for wav in wavs:
        lauts.append(df_laut.loc[df_laut["Wav"]==wav]["Laut"][df_laut["Block"]==block].values)
    lauts = np.array(lauts)
    lauts_mean = np.mean(lauts,axis=1)
    print(lauts_mean)
    sem = stats.sem(lauts,axis=1)
    plt.sca(axes[block_idx])
    plt.bar(np.arange(len(wavs)),lauts_mean,yerr=sem,tick_label=wavs)
    plt.title(block)

fig, axes = plt.subplots(2,2)
plt.suptitle("Bewertung Angenehm")
axes = [ax for sublist in axes for ax in sublist]
for block_idx,block in enumerate(blocks):
    angs = []
    for wav in wavs:
        angs.append(df_ang.loc[df_laut["Wav"]==wav]["Angenehm"][df_laut["Block"]==block].values)
    angs = np.array(angs)
    angs_mean = np.mean(angs,axis=1)
    print(angs_mean)
    sem = stats.sem(angs,axis=1)
    plt.sca(axes[block_idx])
    plt.bar(np.arange(len(wavs)),angs_mean,yerr=sem,tick_label=wavs)
    plt.title(block)

groups = df_laut["Subj"]
formula = "Laut ~ Block*Wav"
laut_model = MixedLM.from_formula(formula, df_laut, groups=groups)
laut_mf = laut_model.fit()
print(laut_mf.summary())

groups = df_ang["Subj"]
formula = "Angenehm ~ Block*Wav"
ang_model = MixedLM.from_formula(formula, df_ang, groups=groups)
ang_mf = ang_model.fit()
print(ang_mf.summary())

font = {'weight' : 'bold',
        'size'   : 38}
matplotlib.rc('font', **font)

fig, axes = plt.subplots(1, 2, figsize=(38.4, 21.6))
angs_block = []
for block in blocks:
    angs_block.append(df_ang.loc[df_ang["Block"]==block]["Angenehm"].values)
angs_block = np.array(angs_block)
angs_block_mean = np.mean(angs_block,axis=1)
sem = stats.sem(angs_block,axis=1)
eng_blocks = ["Audio", "Visual", "Aud. Distract"]
axes[0].barh(np.arange(len(blocks)), angs_block_mean, xerr=sem,tick_label=eng_blocks)
axes[0].set_title("By condition")
axes[0].set_xlim(-0.45, 0)
axes[0].set_yticks(np.arange(len(eng_blocks)))
axes[0].set_yticklabels(eng_blocks)
axes[0].invert_yaxis()
axes[0].invert_xaxis()

angs_wav = []
for wav in wavs:
    angs_wav.append(df_ang.loc[df_ang["Wav"]==wav]["Angenehm"].values)
angs_wav = np.array(angs_wav)
angs_wav_mean = np.mean(angs_wav,axis=1)
sem = stats.sem(angs_wav,axis=1)
eng_tones = ["4000Hz FFT", "4000Hz Sine", "7000Hz Sine", "4000Hz Chebyshev"]
axes[1].barh(np.arange(len(eng_tones)), angs_wav_mean, xerr=sem)
axes[1].set_yticks(np.arange(len(eng_tones)))
axes[1].set_yticklabels(eng_tones)
axes[1].set_title("By tone")
axes[1].set_xlim(-0.45, 0)
axes[1].invert_yaxis()
axes[1].invert_xaxis()

plt.suptitle("Pleasantness rating")
plt.tight_layout()
