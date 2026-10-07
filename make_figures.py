"""Figures from verified artifacts (no re-simulation)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/benchmark.json"))

# 1. autoencoder 2-D map (colored by digit)
try:
    Z = np.load("results/figures/ae_codes.npy")
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    D = load_digits(); X = D.data / 16.0; y = D.target
    _, _, _, yte = train_test_split(X, y, test_size=len(X) - 1297, random_state=0, stratify=y)
    plt.figure(figsize=(5, 4)); plt.scatter(Z[:, 0], Z[:, 1], c=yte, cmap="tab10", s=6)
    plt.title(f"Autoencoder 2-number map (1NN {R['04_ae']['nn_ae']*100:.1f}%)"); plt.colorbar(label="digit")
    plt.tight_layout(); plt.savefig("results/figures/fig_ae_map.png", dpi=120); plt.close()
except Exception as e:
    print("ae fig skipped:", e)

# 2. t-SNE map
try:
    Y = np.load("results/figures/tsne_Y.npy"); yy = np.load("results/figures/tsne_y.npy")
    plt.figure(figsize=(5, 4)); plt.scatter(Y[:, 0], Y[:, 1], c=yy, cmap="tab10", s=6)
    plt.title(f"t-SNE map (1NN {R['05_tsne']['nn_tsne']*100:.1f}%)"); plt.colorbar(label="digit")
    plt.tight_layout(); plt.savefig("results/figures/fig_tsne_map.png", dpi=120); plt.close()
except Exception as e:
    print("tsne fig skipped:", e)

# 3. distillation + deep + ff bars
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
ax[0].bar(["sigmoid-8", "relu-8"], [R["06_deep"]["sig_acc"], R["06_deep"]["relu_acc"]])
ax[0].set_title("Depth: ReLU fixes vanishing blame"); ax[0].set_ylim(0, 1)
ax[1].bar(["hard-3s", "soft-3s"], [R["07_distill"]["hard_3"], R["07_distill"]["soft_3"]])
ax[1].set_title("Distillation: unseen-digit recall"); ax[1].set_ylim(0, 1)
ax[2].bar(["FF", "backprop"], [R["08_ff"]["acc_ff"], R["08_ff"]["acc_bp"]])
ax[2].set_title("Forward-Forward vs backprop"); ax[2].set_ylim(0, 1)
plt.tight_layout(); plt.savefig("results/figures/fig_bars.png", dpi=120); plt.close()

# 4. live markets
lv = R["09_live"]["assets"]
plt.figure(figsize=(5, 3.5))
plt.bar(list(lv.keys()), [lv[k]["R2"] for k in lv])
plt.title("Live markets: autoencoder R² on 20-day windows"); plt.ylim(0, 1)
plt.tight_layout(); plt.savefig("results/figures/fig_live.png", dpi=120); plt.close()
print("figures written")
