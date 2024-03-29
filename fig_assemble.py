import numpy as np
import matplotlib.pyplot as plt
plt.ion()

fs = 24
txt_w, txt_h = 0, 0.85
mos_str = """
          AAAAV
          BBBBW
          CCCCX
          DDDDY
          EEEEZ
          """
mos_str = """
          A
          B
          C
          D
          E
          """
fig, axes = plt.subplot_mosaic(mos_str, figsize=(21.6, 21.6))

img_a = np.load("../images/theta0_task.npy")
img_b = np.load("../images/params_bar_theta_0_t.npy")
img = np.concatenate((img_a, img_b[...,:3]), axis=1)
axes["A"].imshow(img)
axes["A"].text(txt_w, txt_h, "A| Theta general task", transform=axes["A"].transAxes,
               fontsize=fs)

img_a = np.load("../images/alpha0_task.npy")
img_b = np.load("../images/params_bar_alpha_0_t.npy")
img = np.concatenate((img_a, img_b[...,:3]), axis=1)
axes["B"].imshow(img)
axes["B"].text(txt_w, txt_h, "B| Low alpha general task", transform=axes["B"].transAxes,
             fontsize=fs)

img_a = np.load("../images/alpha1_task.npy")
img_b = np.load("../images/params_bar_alpha_1_t.npy")
img = np.concatenate((img_a, img_b[...,:3]), axis=1)
axes["C"].imshow(img)
axes["C"].text(txt_w, txt_h, "C| High alpha general task", transform=axes["C"].transAxes,
             fontsize=fs)

img_a = np.load("../images/alpha1_motor.npy")
img_b = np.load("../images/params_bar_alpha_1_m.npy")
img = np.concatenate((img_a, img_b[...,:3]), axis=1)
axes["D"].imshow(img)
axes["D"].text(txt_w, txt_h, "D| High alpha motor response tasks", transform=axes["D"].transAxes,
             fontsize=fs)

img_a = np.load("../images/alpha1_zaehlen.npy")
img_b = np.load("../images/params_bar_alpha_1_z.npy")
img = np.concatenate((img_a, img_b[...,:3]), axis=1)
axes["E"].imshow(img)
axes["E"].text(txt_w, txt_h, "E| High alpha counting backwards", transform=axes["E"].transAxes,
             fontsize=fs)

for ax in axes.values():
    ax.axis("off")

plt.suptitle("Main patterns of task-based change from resting state", fontsize=34)
plt.tight_layout()

plt.savefig("../images/fig4.png")
plt.savefig("../images/fig4.tif")

# supplemental 1

mos_str = """
          AAAAY
          BBBBZ
          """
fig, axes = plt.subplot_mosaic(mos_str, figsize=(27, 21.6))

img = np.load("../images/beta0_rest.npy")
axes["A"].imshow(img)
axes["A"].text(txt_w, txt_h, "A| Beta (13-30Hz)", transform=axes["A"].transAxes,
               fontsize=fs)
img = np.load("../images/beta_0_rest_annotated.npy")
axes["Y"].imshow(img)

img = np.load("../images/gamma0_rest.npy")
axes["B"].imshow(img)
axes["B"].text(txt_w, txt_h, "B| Gamma (31-48Hz)", transform=axes["B"].transAxes,
             fontsize=fs)
img = np.load("../images/gamma_0_rest_annotated.npy")
axes["Z"].imshow(img)

for k in axes.keys():
    axes[k].axis("off")
plt.suptitle("High band resting state directed connectivity", fontsize=34)
plt.tight_layout()

plt.savefig("../images/fig_s1.png")
plt.savefig("../images/fig_s1.tif")

# # supplemental 2
#
# fig, axes = plt.subplots(1, 2, figsize=(19.2*2, 19.2))
# img = np.load("../images/params_bar_alpha_1_LA1.npy")
# axes[0].imshow(img)
# axes[0].axis("off")
# axes[0].set_title("\nto primary auditory cortex", fontsize=34)
# img = np.load("../images/params_bar_alpha_1_LV1.npy")
# axes[1].imshow(img)
# axes[1].axis("off")
# axes[1].set_title("\nto primary visual cortex", fontsize=34)
# plt.suptitle("High alpha from left parietal and motor cortex", fontsize=34)
# plt.tight_layout()
#
# plt.savefig("../images/fig3.png")
# plt.savefig("../images/fig3.tif")

# # supplemental 3
#
# fig, axes = plt.subplots(1, 2, figsize=(19.2*2, 19.2))
# img = np.load("../images/params_bar_theta_0_c.npy")
# axes[0].imshow(img)
# axes[0].axis("off")
# axes[0].set_title("Theta", fontsize=34)
# img = np.load("../images/params_bar_alpha_0_c.npy")
# axes[1].imshow(img)
# axes[1].axis("off")
# axes[1].set_title("Low alpha", fontsize=34)
# plt.suptitle("Patterns of divergence from rest", fontsize=34)
# plt.tight_layout()
#
# plt.savefig("../images/fig_s3.png")
# plt.savefig("../images/fig_s3.tif")
