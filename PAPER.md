# From Blame Assignment to Forward-Only Learning: A Verified Reconstruction of Hinton's Eight Ideas (with Live-Market Transfer)

## 1. Introduction

For forty years the dominant objection to connectionism was *credit assignment*: when a network errs, which weight is to blame? Hinton's career is a sequence of increasingly radical answers — propagate blame backwards (1986), sidestep labels with energies (1985), squeeze through bottlenecks (2006), map without decoding (2008), scale depth with better units and noise (2010–12), transfer resemblance rather than labels (2015), and finally abolish the backward pass entirely (2022). The 2024 Nobel Prize in Physics (shared with Hopfield) cited the Boltzmann-machine lineage.

This paper reconstructs all eight answers as minimal programs, checks every quantitative claim of the lecture *"Nobody drew these digits"* against executed code, and adds what the lecture lacks: seed audits, threshold-gated benchmarks, honest negative results, and verification on live 2026 market data.

## 2. Related work (one paragraph per idea)

**2.1 Backpropagation.** Rumelhart, Hinton & Williams (Nature 1986) showed a forward pass plus one backward pass assigns every weight its exact gradient share (blame ∝ wire strength × endpoint sensitivity), verified here to 3.1e-12 against finite differences. Minsky & Papert (1969) had shown one neuron cannot do XOR; the 2-2-1 MLP repaired it and invented helper features unaided.
**2.2 Embeddings.** Hinton (1986) fed one-hot persons/relations through a 6-unit bottleneck; nationality, generation and tree position emerged untaught. Modern language models begin identically (token → vector).
**2.3 Boltzmann machines.** Ackley, Hinton & Sejnowski (1985); contrastive divergence, "awake minus asleep" (Hinton 2002). The Nobel committee named this work. Energy E(v,h) = −vWh − bv − ch; learning makes data low-energy.
**2.4 Autoencoders.** Hinton & Salakhutdinov (Science 2006): pretrain + backprop through a narrow waist beats PCA (.0325 vs .0523), founding modern representation learning.
**2.5 t-SNE.** Van der Maaten & Hinton (JMLR 2008): match neighbor probabilities under a heavy-tailed Student-t map; 97.8% 1-NN at 2-D vs 98% at 64-D. Caveat preserved: trust adjacency, never island sizes.
**2.6 Depth.** Nair & Hinton (2010) ReLU preserves blame magnitude; Srivastava et al./Hinton (2012/14) dropout breaks co-adaptation; Krizhevsky, Sutskever & Hinton (2012) AlexNet 15.3% vs 26.2% top-5 turned the field.
**2.7 Distillation.** Hinton, Vinyals & Dean (2015): temperature-softened teacher outputs carry "dark knowledge" of class resemblance; students learn unseen classes (98.6% of never-shown threes in the paper).
**2.8 Forward-Forward.** Hinton (2022): two forward passes on positive/negative (label-overlaid) data, per-layer goodness objectives, layer-norm pass-through — no backward wiring, closer to biology and to low-power hardware.

## 3. Method

Nine standalone NumPy programs (no autograd, no torch): XOR MLP with explicit `backward()`; family-tree embedding model (24 persons, 12 relations, 96 questions, 6-D person code); Bernoulli RBM (64+64, CD-1); 64-32-2-32-64 autoencoder; exact t-SNE (perplexity 30, early exaggeration, 300 steps); 8-layer sigmoid/ReLU MLPs + spoiled-label dropout study; teacher–student distillation at T=8 with all 3s removed; two-layer Forward-Forward with sum-of-squares goodness; live-market transfer (BTC/EURUSD/SPY windows through the same autoencoder/RBM lens). Fixed seed 0, CPU-only.

## 4. Experiments and alignment with the lecture

Full table in README (all numbers from `results/benchmark.json`). Highlights: gradient check 3.1e-12; AE MSE .0397 vs PCA .0530; t-SNE 1-NN 98.8/98.7/57.9; vanishing ratio 6.3e-06 (≈160,000×, lecture: 200,000×); ReLU 94.4% vs sigmoid 10.2%; distillation 0% → 66.7% unseen-digit recall; FF 91.4% vs BP 96.2% with discriminative goodness (4.63 vs 3.33). Two claims required honest recalibration (see §5): loss scales (0.5·mean vs sum) and epoch budgets (30 vs 300 CD epochs).

## 5. Novel findings (negative results as contributions)

**(F1) Seat-structure fragility.** Nationality separates in PC1 every seed (gap ≈ 0.9), but exact twin pairing is ≈ chance (1/12) with or without PC1 removal; hidden-question accuracy averages 1.8/4 over seeds, not 2.6/4. The lecture's 12/12 is a best-seed phenomenon. Implication: co-occurrence statistics teach *family*; tree *position* needs architectural bias. Proposed PhD experiment: scale relations 8→16 and measure twin-score curves.
**(F2) Memorization needs scale.** At n=1,297, 33% label noise is *resisted*, not memorized (67.5% spoiled-train / 92.4% clean), yet dropout still gains +1.6 pts. Implication: report spoiled-train agreement alongside test accuracy; dropout's small-data role is variance reduction.
**(F3) FF theta calibration.** Mean-goodness FF stalls at chance (9.6%) from vanishing label gradients; sum-of-squares goodness with thetas from init statistics restores 91.4%. Publishable as a hardware note.
**(F4) Market transfer.** Identical autoencoders compress 20-day return windows (R² .24–.35) with bottleneck–volatility correlation up to .34 — Hinton's unsupervised lens works on 2026 markets, with honestly lower compressibility than vision.

## 6. Threats to validity

sklearn digits (1,797) is smaller/easier than MNIST/ImageNet; exact 1986 trees and 2012-scale nets are approximated; t-SNE run at 300 (not 600) steps; live-data endpoints can rate-limit (fallbacks are flagged, never silent); single-seed headline numbers are accompanied by multi-seed audits but not full confidence intervals.

## 7. Future work (PhD roadmap)

(i) Twin-score scaling laws over relation-set size and bottleneck width; (ii) memorization phase diagram (n, p, width) unifying F2 with Zhang et al. (2017); (iii) FF θ auto-calibration rules for analog hardware; (iv) energy-based market-stress early warning from RBM energies (F4); (v) capsule/layer-norm/wake-sleep extensions the lecture omits.

## References

See README reference list (Rumelhart–Hinton–Williams 1986; Ackley–Hinton–Sejnowski 1985; Hinton 2002, 1986, 2022; Hinton–Salakhutdinov 2006; Van der Maaten–Hinton 2008; Nair–Hinton 2010; Krizhevsky–Sutskever–Hinton 2012; Srivastava et al. 2014; Hinton–Vinyals–Dean 2015; Nobel 2024; FAIR; Nüst et al. Dockerfile rules; NeurIPS 2020 code guidelines).
