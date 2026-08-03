# Simulation notes (R1.4)

## Setup
5 simulated base detectors, population prevalence 3.4% (matches AI4I 2020's
actual ~3.39% failure rate). Each detector's per-trial recall/FPR is drawn from a Normal
distribution centered on its EMPIRICAL mean/std from the stability experiment (task 1):

  SVM: recall~0.339 (+/-0.052), FPR~0.0018 (+/-0.0007)
  RF: recall~0.736 (+/-0.047), FPR~0.0027 (+/-0.0009)
  KNN: recall~0.371 (+/-0.047), FPR~0.0037 (+/-0.0012)
  LR: recall~0.239 (+/-0.047), FPR~0.0033 (+/-0.0011)
  GBC: recall~0.739 (+/-0.044), FPR~0.0032 (+/-0.0013)

A Gaussian copula gives the 5 detectors' errors a controllable pairwise correlation rho
(compound symmetry), swept from 0 (independent errors) to 0.8 (highly correlated errors).
40 trials of n=20000 simulated instances per rho value.

## Result
- At rho=0 (independent detectors): OR-logic recall exceeds majority-voting recall by
  +0.516 (see `recall_vs_rho.png`).
- At rho=0.8 (highly correlated detectors): the OR-logic vs. majority recall
  gap is +0.374.

## Takeaway
OR-logic's recall advantage over majority voting is **largest when the base detectors are
diverse (low pairwise error correlation) and individually reasonably precise** -- exactly
the setting the empirical detector stats above describe (each base model on the real data
has specificity/FPR in a similar narrow band, and the five models use structurally
different algorithms -- SVM, tree ensembles, instance-based, and linear -- so their errors
are not tightly coupled). As detectors become more correlated (rho -> 0.8), OR-logic's
extra recall shrinks toward whatever the majority rule already captures, because a
"union of five near-identical detectors" behaves like one detector.

This links directly back to why `class_weight='balanced'` was removed from SVM/RF/LR in
the locked config (see `outputs/balanced_variant/README.txt`): making every base model
individually MORE willing to fire (lower precision, higher FPR) does not just move along
this same simulated curve -- it pushes every detector's FPR up simultaneously, and
OR-logic's false-positive count scales with the union of all five FPRs regardless of rho.
The simulation here isolates the *diversity* dimension (rho) holding each detector's own
recall/FPR fixed; the balanced-weights experiment (imbalance/, R1.6) isolates the
*individual detector precision* dimension. Both point the same direction: OR-logic works
best over diverse, individually precise detectors, not over detectors that have each been
independently pushed toward high recall/low precision.
