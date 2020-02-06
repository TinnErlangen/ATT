import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
plt.ion()

df_laut = pd.read_pickle("../behave/laut")
df_ang = pd.read_pickle("../behave/ang")

blocks = ["audio","visual","visselten","zaehlen"]
wavs = ["4000fftf","4000Hz","7000Hz","4000cheby"]

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
plt.bar(np.arange(len(wavs)),lauts_wav_mean,yerr=sem,tick_label=wavs)
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
