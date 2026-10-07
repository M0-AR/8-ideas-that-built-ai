"""Demo GIFs for README/docs (all generated from the repo's own code, seed 0).

- docs/demo_xor.gif      : XOR decision boundary learned step by step
- docs/demo_dreams.gif   : RBM dream chain: noise -> digit over Gibbs steps
Small, fast, Pillow writer (no ffmpeg needed).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.makedirs("docs", exist_ok=True)

def xor_gif():
    import importlib.util
    spec = importlib.util.spec_from_file_location("x", "src/01_backprop_xor.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rng = np.random.default_rng(0)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    W1 = rng.normal(0, 1, (2, 2)); b1 = np.zeros((1, 2))
    W2 = rng.normal(0, 1, (2, 1)); b2 = np.zeros((1, 1))
    snaps = []
    gx, gy = np.meshgrid(np.linspace(-0.5, 1.5, 50), np.linspace(-0.5, 1.5, 50))
    G = np.c_[gx.ravel(), gy.ravel()]
    for step in range(1200):
        out, cache = m.forward(X, W1, b1, W2, b2)
        dout = (out - y.reshape(-1, 1)) / 4
        dW1, db1, dW2, db2 = m.backward(dout, cache, W2, W1, b1)
        W1 -= 3.0 * dW1; b1 -= 3.0 * db1; W2 -= 3.0 * dW2; b2 -= 3.0 * db2
        if step in {0, 5, 15, 30, 60, 120, 250, 500, 800, 1199}:
            Z, _ = m.forward(G, W1, b1, W2, b2)
            snaps.append((step, Z.reshape(gx.shape), float(0.5 * ((out.ravel() - y) ** 2).mean())))
    fig, ax = plt.subplots(figsize=(4, 3.2))
    cont = [ax.contourf(gx, gy, snaps[0][1], levels=20, cmap="coolwarm", alpha=0.8)]
    pts = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=80, edgecolors="k")
    title = ax.set_title("")
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)

    def upd(i):
        nonlocal cont
        cont[0].remove()
        cont[0] = ax.contourf(gx, gy, snaps[i][1], levels=20, cmap="coolwarm", alpha=0.8)
        title.set_text(f"XOR learned live — step {snaps[i][0]}, loss {snaps[i][2]:.4f}")
        return cont[0],
    ani = animation.FuncAnimation(fig, upd, frames=len(snaps), interval=600)
    ani.save("docs/demo_xor.gif", writer="pillow", fps=2)
    plt.close(fig)
    print("docs/demo_xor.gif ok")

def dreams_gif():
    import importlib.util
    spec = importlib.util.spec_from_file_location("r", "src/03_boltzmann_dreams.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rng = np.random.default_rng(0)
    (Xtr, ytr), _ = m.load_digits01()
    rbm = m.RBM(seed=0)
    rbm.train_cd1((Xtr > 0.5).astype(float), epochs=8, seed=0)
    v = (rng.random((4, 64)) < 0.5).astype(float)
    frames = [v.reshape(4, 8, 8).copy()]
    for _ in range(11):
        h = (rng.random((4, 64)) < rbm.p_h(v)).astype(float)
        v = (rng.random((4, 64)) < rbm.p_v(h)).astype(float)
        frames.append(v.reshape(4, 8, 8).copy())
    fig, axes = plt.subplots(1, 4, figsize=(5, 1.6))
    ims = [a.imshow(frames[0][i], cmap="gray", vmin=0, vmax=1) for i, a in enumerate(axes)]
    for a in axes: a.axis("off")
    title = fig.suptitle("")
    def upd(i):
        for k in range(4): ims[k].set_data(frames[i][k])
        title.set_text(f"Dreaming: pure noise → digits (Gibbs step {i * 5})")
        return ims
    ani = animation.FuncAnimation(fig, upd, frames=len(frames), interval=500)
    ani.save("docs/demo_dreams.gif", writer="pillow", fps=2)
    plt.close(fig)
    print("docs/demo_dreams.gif ok")

if __name__ == "__main__":
    xor_gif(); dreams_gif()
