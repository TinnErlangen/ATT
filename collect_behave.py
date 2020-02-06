import csv
import numpy as np
from copy import deepcopy
from scipy.stats import zscore
import pandas as pd
import matplotlib.pyplot as plt
plt.ion()

subjs = ["ATT_10", "ATT_11", "ATT_12", "ATT_13", "ATT_14", "ATT_15", "ATT_16",
         "ATT_17", "ATT_18", "ATT_19", "ATT_20", "ATT_21", "ATT_22", "ATT_23",
         "ATT_24", "ATT_25", "ATT_26", "ATT_28", "ATT_29", "ATT_29",
         "ATT_31",  "ATT_33", "ATT_34", "ATT_35", "ATT_36",
         "ATT_37"]

tone_temp = {"4000fftf":[],"4000Hz":[],"7000Hz":[],"4000cheby":[]}
table = {"audio":deepcopy(tone_temp),"visselten":deepcopy(tone_temp),
         "vis":deepcopy(tone_temp),"zaehlen":deepcopy(tone_temp)}

# hierarchical table
# block_key = {"Audio modulations only":0,
#              "Infrequent visual modulations, ignore audio":1,
#              "Visual modulations only":2, "No modulations, count backward.":3}
# wav_key = {"4000_fftf.wav":0,"4000_cheby.wav":1,"4000Hz.wav":2,"7000Hz.wav":3}
# table = np.zeros((32,len(subjs)))
# labels = [["audio","visselten","vis","zaehlen"],["4000fftf","4000cheby",
#           "4000Hz","7000Hz"],["Laut","Angenehm"]]
# labels = pd.MultiIndex.from_product(labels,names=["Block","Wav","Rating"])
# for sub_idx,sub in enumerate(subjs):
#     with open("../behave/"+sub+".txt") as file:
#         next(file)
#         reader = csv.DictReader(file,delimiter="\t")
#         for row in reader:
#             cond_index = 0
#             cond_index += block_key[row["Block"]]*8
#             cond_index += wav_key[row["Wavfile"]]*2
#             table[cond_index,sub_idx] = np.array(row["Laut"]).astype("float")
#             table[cond_index+1,sub_idx] = np.array(row["Angenehm"]).astype("float")
# df = pd.DataFrame(table,index=labels,columns=subjs)
# df_laut = df.xs("Laut",level="Rating").apply(zscore)
# df_ang = df.xs("Angenehm",level="Rating").apply(zscore)

# non-hierarchical
block_key = {"Audio modulations only":"audio", "Visual modulations only":"visual",
             "Infrequent visual modulations, ignore audio":"visselten",
             "No modulations, count backward.":"zaehlen"}
wav_key = {"4000_fftf.wav":"4000fftf","4000_cheby.wav":"4000cheby","4000Hz.wav":"4000Hz",
           "7000Hz.wav":"7000Hz"}
table_laut = []
table_ang = []
for sub_idx,sub in enumerate(subjs):
    with open("../behave/"+sub+".txt") as file:
        next(file)
        reader = csv.DictReader(file,delimiter="\t")
        for row in reader:
            table_laut.append([float(row["Laut"]),
                              block_key[row["Block"]],wav_key[row["Wavfile"]],
                              sub])
            table_ang.append([float(row["Angenehm"]),
                              block_key[row["Block"]],wav_key[row["Wavfile"]],
                              sub])
df_laut = pd.DataFrame(table_laut,columns=["Laut","Block","Wav","Subj"])
df_ang = pd.DataFrame(table_ang,columns=["Angenehm","Block","Wav","Subj"])
for sub in subjs:
    zs = zscore(df_laut[df_laut["Subj"]==sub]["Laut"])
    df_laut.loc[df_laut["Subj"]==sub,["Laut"]] = zs
    zs = zscore(df_ang[df_ang["Subj"]==sub]["Angenehm"])
    df_ang.loc[df_ang["Subj"]==sub,["Angenehm"]] = zs

df_laut = df_laut.sort_values(["Subj","Block","Wav"])
df_laut = pd.concat([df_laut,pd.get_dummies(df_laut["Block"])],axis=1)
df_laut = pd.concat([df_laut,pd.get_dummies(df_laut["Wav"])],axis=1)
df_laut = pd.concat([df_laut,pd.get_dummies(df_laut["Subj"])],axis=1)
df_laut.to_pickle("../behave/laut")

df_ang = df_ang.sort_values(["Subj","Block","Wav"])
df_ang = pd.concat([df_ang,pd.get_dummies(df_ang["Block"])],axis=1)
df_ang = pd.concat([df_ang,pd.get_dummies(df_ang["Wav"])],axis=1)
df_ang = pd.concat([df_ang,pd.get_dummies(df_ang["Subj"])],axis=1)
df_ang.to_pickle("../behave/ang")
