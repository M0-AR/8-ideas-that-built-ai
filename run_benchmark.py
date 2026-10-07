"""Benchmark orchestrator: runs 01-09, writes results/benchmark.json + figures.

Usage: python3 run_benchmark.py [--fast|--full]
  --fast: reduced epochs/steps (<~2 min, CI-friendly)
  --full: paper-grade settings (default modules' __main__ values)
Reproducibility: SEED=0 everywhere (override HINTON_SEED), pinned reqs,
single command via docker compose.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
SEED = int(os.environ.get("HINTON_SEED", "0"))

def run_all(full=True):
    import importlib
    mods = {}
    for n in ["01_backprop_xor", "02_embeddings_family", "03_boltzmann_dreams",
              "04_autoencoder_squeeze", "05_tsne_map", "06_deep_relu_dropout",
              "07_distillation_dark_knowledge", "08_forward_forward",
              "09_live_market_transfer"]:
        mods[n] = importlib.import_module(n)
    R = {"seed": SEED, "full": full, "steps": {}}
    t0 = time.time()
    def tick(k):
        R["steps"][k] = round(time.time() - t0, 1)

    m = mods["01_backprop_xor"]
    r = m.train_xor(seed=SEED); a, n_, d = m.gradient_check(); ok, tot = m.success_rate(20)
    R["01_xor"] = {"loss": r["loss1"], "preds": [round(v, 4) for v in r["preds"]],
                   "grad_diff": d, "success_20": [ok, tot]}; tick("01")

    m = mods["02_embeddings_family"]
    r = m.train(seed=SEED)
    hs = [m.train(seed=s)["hidden_ok"] for s in range(5)]
    R["02_embed"] = {"n_q": r["n_questions"], "train_acc": r["train_acc"],
                     "hidden": r["hidden_ok"], "hidden_avg5": float(np.mean(hs)),
                     "twins": r["twins"], "twins_nopc1": r.get("twins_nopc1"),
                     "pc1_gap": round(float(r.get("pc1_gap", 0)), 3)}; tick("02")

    m = mods["03_boltzmann_dreams"]
    r = m.run(seed=SEED, epochs=30 if full else 8)
    R["03_rbm"] = {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in r.items() if k != "rbm"}; tick("03")

    m = mods["04_autoencoder_squeeze"]
    r = m.run(seed=SEED, epochs=60 if full else 20)
    R["04_ae"] = {"mse_ae": round(r["mse_ae"], 4), "mse_pca": round(r["mse_pca"], 4),
                  "nn_ae": round(r["nn_ae"], 4), "nn_pca": round(r["nn_pca"], 4)}
    np.save("results/figures/ae_codes.npy", r["Zte"]); tick("04")

    if full:
        m = mods["05_tsne_map"]
        r = m.run(steps=300)
        R["05_tsne"] = {"nn_raw": round(r["nn_raw"], 4), "nn_tsne": round(r["nn_tsne"], 4),
                        "nn_pca": round(r["nn_pca"], 4)}
        np.save("results/figures/tsne_Y.npy", r["Y"]); np.save("results/figures/tsne_y.npy", r["y"]); tick("05")
    else:
        R["05_tsne"] = {"skipped": "fast-mode"}; tick("05")

    m = mods["06_deep_relu_dropout"]
    ep = 30 if full else 15
    Xtr, ytr, Xte, yte = m.load()
    sig = m.train_mlp(Xtr, ytr, Xte, yte, act="sigmoid", epochs=ep, lr=0.1)
    rel = m.train_mlp(Xtr, ytr, Xte, yte, act="relu", epochs=ep, lr=0.1)
    mem = m.train_mlp(Xtr, ytr, Xte, yte, hidden=(256, 256), act="relu", epochs=ep, lr=0.1, spoil=1 / 3, seed=1)
    do = m.train_mlp(Xtr, ytr, Xte, yte, hidden=(256, 256), act="relu", drop=0.5, epochs=ep, lr=0.1, spoil=1 / 3, seed=1)
    R["06_deep"] = {"sig_acc": round(sig["acc_test"], 4), "sig_ratio": float(sig["blame_ratio"]),
                    "relu_acc": round(rel["acc_test"], 4), "relu_ratio": float(rel["blame_ratio"]),
                    "mem_train": round(mem["acc_spoiled_train"], 4), "mem_test": round(mem["acc_test"], 4),
                    "do_train": round(do["acc_spoiled_train"], 4), "do_test": round(do["acc_test"], 4)}; tick("06")

    m = mods["07_distillation_dark_knowledge"]
    r = m.run(epochs=30 if full else 20)
    R["07_distill"] = {"teacher": round(r["acc_teacher"], 4), "hard_3": round(r["hard_3"], 4),
                       "soft_3": round(r["soft_3"], 4), "hard_other": round(r["hard_other"], 4),
                       "soft_other": round(r["soft_other"], 4),
                       "n_threes": r["n_threes_test"]}; tick("07")

    m = mods["08_forward_forward"]
    r = m.run(epochs=30 if full else 12); bp = m.run_bp()
    R["08_ff"] = {"acc_ff": round(r["acc_ff"], 4), "acc_bp": round(bp, 4),
                  "G6": [round(v, 2) for v in r["G6"]]}; tick("08")

    m = mods["09_live_market_transfer"]
    r = m.run()
    R["09_live"] = {"all_live": r["all_live"], "live": r["live"],
                    "assets": {k: {"R2": round(v["r2"], 3),
                                   "corr_vol": round(v["corr_code0_vol"], 3),
                                   "synth": v["synthetic_fallback"]} for k, v in r["assets"].items()}}; tick("09")

    R["total_s"] = round(time.time() - t0, 1)
    os.makedirs("results/figures", exist_ok=True)
    with open("results/benchmark.json", "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps(R, indent=2))
    return R

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--fast", action="store_true"); ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    run_all(full=(not a.fast))
