import numpy as np
import matplotlib.pyplot as plt
plt.ion()

fs = 24
txt_w, txt_h = 0, 0.85
fig, axes = plt.subplots(5, 1, figsize=(20, 21.6))

img = np.load("../images/theta0_task.npy")
axes[0].imshow(img)
axes[0].text(txt_w, txt_h, "A| Theta general task", transform=axes[0].transAxes,
             fontsize=fs)

img = np.load("../images/alpha0_task.npy")
axes[1].imshow(img)
axes[1].text(txt_w, txt_h, "B| Low alpha general task", transform=axes[1].transAxes,
             fontsize=fs)

img = np.load("../images/alpha1_task.npy")
axes[2].imshow(img)
axes[2].text(txt_w, txt_h, "C| High alpha general task", transform=axes[2].transAxes,
             fontsize=fs)

img = np.load("../images/alpha1_motor.npy")
axes[3].imshow(img)
axes[3].text(txt_w, txt_h, "D| High alpha motor response tasks", transform=axes[3].transAxes,
             fontsize=fs)

img = np.load("../images/alpha1_zaehlen.npy")
axes[4].imshow(img)
axes[4].text(txt_w, txt_h, "E| High alpha counting backwards", transform=axes[4].transAxes,
             fontsize=fs)

for ax in axes:
    ax.axis("off")

plt.suptitle("Main patterns of difference from resting state", fontsize=34)
plt.tight_layout()

plt.savefig("../images/fig2.png")
plt.savefig("../images/fig2.tif")

# supplemental 1

fig, axes = plt.subplots(2, 1, figsize=(20, 12))

img = np.load("../images/beta0_rest.npy")
axes[0].imshow(img)
axes[0].text(txt_w, txt_h, "A| Beta resting state", transform=axes[0].transAxes,
             fontsize=fs)

img = np.load("../images/gamma0_rest.npy")
axes[1].imshow(img)
axes[1].text(txt_w, txt_h, "B| Gamma resting task", transform=axes[1].transAxes,
             fontsize=fs)

for ax in axes:
    ax.axis("off")

plt.tight_layout()

plt.savefig("../images/fig_s1.png")
plt.savefig("../images/fig_s1.tif")
