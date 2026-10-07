"""08 — Forward only (Forward-Forward, Hinton Dec 2022). Clean rewrite.

No backward pass. Label written into picture: 10 extra pixels, one on.
Positive data = digit + true label; negative = digit + wrong label.
Each layer: goodness g = SUM of squared activities. p = sigmoid(g - theta).
  pos loss = softplus(-(g-th)) -> pushes g up; neg loss = softplus(+(g-th)) -> down.
Each layer has its own theta near its operating point; layer-norm pass-through
(unit direction only) so next layer must find something new.
Inference: try 10 labels, keep max total goodness. Baseline: backprop same arch.
Numpy + sklearn only.
"""
import numpy as np
SEED = 0
LABEL_SCALE = 1.0

def load():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    D = load_digits(); X = D.data / 16.0; y = D.target
    return train_test_split(X, y, test_size=500, random_state=0, stratify=y)

def add_label(X, y, s=LABEL_SCALE):
    L = np.zeros((len(X), 10)); L[np.arange(len(X)), y] = s
    return np.concatenate([X, L], 1)

def _layer_g(A):
    return (A ** 2).sum(1)

def train_ff(Xtr, ytr, h1=128, h2=64, epochs=30, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    d_in = 74
    W1 = rng.normal(0, 0.05, (d_in, h1)); b1 = np.zeros(h1)
    W2 = rng.normal(0, 0.05, (h1, h2)); b2 = np.zeros(h2)
    # thetas from initial operating point (pos & neg start together)
    A0 = np.maximum(add_label(Xtr[:512], ytr[:512]) @ W1 + b1, 0)
    th1 = float(_layer_g(A0).mean())
    N0 = A0 / (np.linalg.norm(A0, axis=1, keepdims=True) + 1e-9)
    B0 = np.maximum(N0 @ W2 + b2, 0)
    th2 = float(_layer_g(B0).mean())
    n = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, 128):
            xb = Xtr[perm[i:i + 128]]; yb = ytr[perm[i:i + 128]]; m = len(xb)
            yw = rng.integers(0, 10, m); same = yw == yb
            yw[same] = (yw[same] + rng.integers(1, 10, same.sum())) % 10
            xp = add_label(xb, yb); xn = add_label(xb, yw)
            # ---- layer 1 ----
            Ap = np.maximum(xp @ W1 + b1, 0); gp = _layer_g(Ap)
            An = np.maximum(xn @ W1 + b1, 0); gn = _layer_g(An)
            pp = 1 / (1 + np.exp(-(gp - th1))); pn = 1 / (1 + np.exp(-(gn - th1)))
            dAp = (-(1 - pp))[:, None] * 2 * Ap   # dL/dA pos
            dAn = (pn)[:, None] * 2 * An          # dL/dA neg
            dW1 = (xp.T @ (dAp * (Ap > 0)) + xn.T @ (dAn * (An > 0))) / m
            db1 = ((dAp * (Ap > 0)).sum(0) + (dAn * (An > 0)).sum(0)) / m
            W1 -= lr * dW1; b1 -= lr * db1
            # ---- layer 2 on normalized directions (recompute with new W1) ----
            Ap2 = np.maximum(xp @ W1 + b1, 0); An2 = np.maximum(xn @ W1 + b1, 0)
            Np = Ap2 / (np.linalg.norm(Ap2, axis=1, keepdims=True) + 1e-9)
            Nn = An2 / (np.linalg.norm(An2, axis=1, keepdims=True) + 1e-9)
            Bp = np.maximum(Np @ W2 + b2, 0); g2p = _layer_g(Bp)
            Bn = np.maximum(Nn @ W2 + b2, 0); g2n = _layer_g(Bn)
            qp = 1 / (1 + np.exp(-(g2p - th2))); qn = 1 / (1 + np.exp(-(g2n - th2)))
            dBp = (-(1 - qp))[:, None] * 2 * Bp
            dBn = (qn)[:, None] * 2 * Bn
            dW2 = (Np.T @ (dBp * (Bp > 0)) + Nn.T @ (dBn * (Bn > 0))) / m
            db2 = ((dBp * (Bp > 0)).sum(0) + (dBn * (Bn > 0)).sum(0)) / m
            W2 -= lr * dW2; b2 -= lr * db2
    return (W1, b1, th1, W2, b2, th2)

def goodness(X, y_, P):
    W1, b1, th1, W2, b2, th2 = P
    A = np.maximum(add_label(X, y_) @ W1 + b1, 0)
    g1 = _layer_g(A)
    N = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = np.maximum(N @ W2 + b2, 0)
    return g1 + _layer_g(B)

def predict(X, P):
    G = np.stack([goodness(X, np.full(len(X), c), P) for c in range(10)], 1)
    return G.argmax(1), G

def run(epochs=30, lr=0.05, seed=0):
    Xtr, Xte, ytr, yte = load()
    P = train_ff(Xtr, ytr, epochs=epochs, lr=lr, seed=seed)
    pred, _ = predict(Xte, P)
    acc = float((pred == yte).mean())
    i6 = np.where(yte == 6)[0][0]
    G6 = np.stack([goodness(Xte[i6:i6 + 1], np.array([c]), P) for c in range(10)], 1).ravel()
    return {"acc_ff": acc, "G6": G6.tolist(), "P": P}

def run_bp(epochs=20, lr=0.2, seed=0):
    import importlib.util
    spec = importlib.util.spec_from_file_location("d06", "src/06_deep_relu_dropout.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    Xtr, Xte, ytr, yte = load()
    r = m.train_mlp(Xtr, ytr, Xte, yte, hidden=(128, 64), act="relu", epochs=epochs, lr=lr, seed=seed)
    return r["acc_test"]

if __name__ == "__main__":
    r = run()
    print(f"forward-forward acc {r['acc_ff']*100:.1f}%")
    print("example test-6 goodness per label:", [round(v, 2) for v in r["G6"]], "true=6")
    print(f"backprop same-arch acc {run_bp()*100:.1f}%")
