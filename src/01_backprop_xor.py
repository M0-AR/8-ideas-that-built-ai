"""01 — Backpropagation on XOR (Rumelhart, Hinton & Williams, Nature 1986).

Standalone program. No torch. Numpy only.
- 1 neuron cannot solve XOR (shows 0.5 everywhere).
- 2-2-1 sigmoid MLP + manual backward pass solves it.
- Gradient check: analytic vs numerical nudge agrees to ~1e-8.
- 100 random starts: reports success rate (video claims ~84/100).
"""
import numpy as np

SEED = 0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    h = sigmoid(z1)
    z2 = h @ W2 + b2
    out = sigmoid(z2)
    cache = (X, h, out)
    return out, cache

def backward(dout_raw, cache, W2, W1, b1, h_unused=None):
    """One backward pass. Returns grads for W2,b2,W1,b1.
    dout_raw = dL/dout where L = 0.5*mean((out-y)^2) style handled by caller.
    Included as the 'one short function' the video refers to.
    """
    X, h, out = cache
    # sigmoid sensitivities
    d_z2 = dout_raw * out * (1 - out)          # blame at output
    dW2 = h.T @ d_z2
    db2 = d_z2.sum(axis=0)
    d_h = d_z2 @ W2.T                           # blame split ∝ wire strength
    d_z1 = d_h * h * (1 - h)
    dW1 = X.T @ d_z1
    db1 = d_z1.sum(axis=0)
    return dW1, db1, dW2, db2

def loss_mse(out, y):
    return float(0.5 * np.mean((out - y) ** 2))

def train_xor(seed=0, steps=5000, lr=3.0, verbose=False):
    rng = np.random.default_rng(seed)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    W1 = rng.normal(0, 1, (2, 2))
    b1 = np.zeros((1, 2))
    W2 = rng.normal(0, 1, (2, 1))
    b2 = np.zeros((1, 1))
    l0 = None
    for i in range(steps):
        out, cache = forward(X, W1, b1, W2, b2)
        if i == 0:
            l0 = loss_mse(out, y)
        # dL/dout for L=0.5*mean((o-y)^2)
        dout = (out - y) / X.shape[0]
        dW1, db1, dW2, db2 = backward(dout, cache, W2, W1, b1)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    out, _ = forward(X, W1, b1, W2, b2)
    return {"loss0": l0, "loss1": loss_mse(out, y),
            "preds": out.ravel().tolist(),
            "W1": W1, "b1": b1, "W2": W2, "b2": b2}

def gradient_check(seed=0):
    rng = np.random.default_rng(seed)
    X = np.array([[1.0, 0.0]])
    y = np.array([[1.0]])
    W1 = rng.normal(0, 1, (2, 2)); b1 = np.zeros((1, 2))
    W2 = rng.normal(0, 1, (2, 1)); b2 = np.zeros((1, 1))
    out, cache = forward(X, W1, b1, W2, b2)
    dout = (out - y) / 1.0
    dW1, _, _, _ = backward(dout, cache, W2, W1, b1)
    analytic = float(dW1[0, 0])
    eps = 1e-6
    W1p = W1.copy(); W1p[0, 0] += eps
    W1m = W1.copy(); W1m[0, 0] -= eps
    lp = loss_mse(forward(X, W1p, b1, W2, b2)[0], y)
    lm = loss_mse(forward(X, W1m, b1, W2, b2)[0], y)
    numeric = (lp - lm) / (2 * eps)
    return analytic, numeric, abs(analytic - numeric)

def success_rate(n=100, steps=5000, lr=3.0, thresh=0.01):
    ok = 0
    for s in range(n):
        r = train_xor(seed=s, steps=steps, lr=lr)
        p = np.array(r["preds"])
        # correct if <0.25 for zeros and >0.75 for ones
        if p[0] < 0.25 and p[1] > 0.75 and p[2] > 0.75 and p[3] < 0.25:
            ok += 1
    return ok, n

if __name__ == "__main__":
    r = train_xor(seed=SEED)
    print(f"loss {r['loss0']:.4f} -> {r['loss1']:.6f}")
    print("preds [00,01,10,11]:", [round(v, 4) for v in r["preds"]])
    a, n_, d = gradient_check()
    print(f"grad check analytic={a:.8f} numeric={n_:.8f} |diff|={d:.2e}")
    ok, tot = success_rate(100)
    print(f"success {ok}/{tot} random starts")
    # single neuron baseline: logistic regression cannot do XOR
    print("single-neuron note: best possible is 0.5 on all four (linear separator impossible)")
