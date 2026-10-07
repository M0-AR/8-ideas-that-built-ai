"""Threshold tests: every claim has a number; CI runs --fast."""
import json, os, subprocess, sys

def load():
    assert os.path.exists("results/benchmark.json"), "run run_benchmark.py first"
    return json.load(open("results/benchmark.json"))

def test_all():
    R = load()
    assert R["01_xor"]["loss"] < 0.002, R["01_xor"]
    assert R["01_xor"]["grad_diff"] < 1e-8, R["01_xor"]
    assert R["02_embed"]["train_acc"] == 1.0, R["02_embed"]
    assert R["02_embed"]["hidden"] >= 1, R["02_embed"]  # seed-0 good run
    assert R["03_rbm"]["mse1"] < R["03_rbm"]["mse0"], R["03_rbm"]
    assert R["03_rbm"]["e_real"] < R["03_rbm"]["e_noise"], R["03_rbm"]
    assert R["03_rbm"]["acc_fix"] > R["03_rbm"]["acc_blank"], R["03_rbm"]
    assert R["04_ae"]["mse_ae"] < R["04_ae"]["mse_pca"], R["04_ae"]
    assert R["04_ae"]["nn_ae"] > R["04_ae"]["nn_pca"], R["04_ae"]
    if "skipped" not in R["05_tsne"]:
        assert R["05_tsne"]["nn_tsne"] > 0.90, R["05_tsne"]
        assert R["05_tsne"]["nn_tsne"] > R["05_tsne"]["nn_pca"], R["05_tsne"]
    assert R["06_deep"]["relu_acc"] > R["06_deep"]["sig_acc"] + 0.3, R["06_deep"]
    assert R["06_deep"]["sig_ratio"] < 1e-3, R["06_deep"]
    assert R["07_distill"]["hard_3"] < 0.2, R["07_distill"]
    assert R["07_distill"]["soft_3"] > 0.4, R["07_distill"]
    assert R["08_ff"]["acc_ff"] > 0.80, R["08_ff"]
    assert R["09_live"]["assets"]["BTC"]["R2"] > 0, R["09_live"]
    print("ALL THRESHOLD TESTS PASSED")
