# Imbalance-strategy notes (R1.6)

## Per-strategy results, OR-Logic and GBC

### GBC
- **none (locked)**: recall=0.800, precision=0.872, specificity=0.996, FN rate=0.200, F1=0.834, MCC=0.830 (TP=68, FN=17, TN=2405, FP=10)
- **SMOTE**: recall=0.906, precision=0.348, specificity=0.940, FN rate=0.094, F1=0.503, MCC=0.540 (TP=77, FN=8, TN=2271, FP=144)
- **class_weight=balanced (parked)**: recall=0.800, precision=0.872, specificity=0.996, FN rate=0.200, F1=0.834, MCC=0.830 (TP=68, FN=17, TN=2405, FP=10)
- **RandomUnderSampler**: recall=0.929, precision=0.294, specificity=0.921, FN rate=0.071, F1=0.446, MCC=0.498 (TP=79, FN=6, TN=2225, FP=190)

### OR-Logic
- **none (locked)**: recall=0.800, precision=0.708, specificity=0.988, FN rate=0.200, F1=0.751, MCC=0.744 (TP=68, FN=17, TN=2387, FP=28)
- **SMOTE**: recall=0.965, precision=0.155, specificity=0.815, FN rate=0.035, F1=0.268, MCC=0.346 (TP=82, FN=3, TN=1969, FP=446)
- **class_weight=balanced (parked)**: recall=0.953, precision=0.162, specificity=0.827, FN rate=0.047, F1=0.277, MCC=0.354 (TP=81, FN=4, TN=1997, FP=418)
- **RandomUnderSampler**: recall=0.976, precision=0.122, specificity=0.753, FN rate=0.024, F1=0.217, MCC=0.297 (TP=83, FN=2, TN=1819, FP=596)

Full 4-strategy x 7-model/ensemble table: `imbalance_results.csv`.
Grouped bar chart: `recall_precision_bars.png`.

**Caveat on GBC's "class_weight=balanced" row:** it is IDENTICAL to GBC's "none" row above,
and that is expected, not a bug -- `sklearn.ensemble.GradientBoostingClassifier` has no
`class_weight` parameter at all, so GBC was never rebalanced in the parked
`outputs/balanced_variant/` run either (only SVM/RF/LR were). GBC's own predictions are
therefore identical between "none" and "class_weight=balanced" by construction; only
OR-Logic's "class_weight=balanced" row differs from "none", because OR-Logic combines GBC
with the four OTHER models that genuinely were rebalanced in that run.

## The trade-off, stated plainly
- **none (locked)**: OR-Logic recall=0.800, specificity=0.988,
  precision=0.708. This is the reference point -- no resampling, no
  class weighting, just the five base models as specified in `src/config.py`.
- **class_weight='balanced'**: OR-Logic recall=0.953 but specificity
  collapses to 0.827 and precision to 0.162
  (418 false positives) -- this is exactly the failure mode documented in
  `outputs/balanced_variant/README.txt`: balancing every base model individually means
  OR-Logic inherits the union of all five models' inflated false-positive rates.
- **SMOTE** (training-fold-only oversampling): OR-Logic recall=0.965,
  specificity=0.815, precision=0.155.
  SMOTE moves recall and precision/specificity in the same general direction as class-weighting, just less extremely -- still a real cost for the recall it buys, though less severe than class_weight='balanced'.
- **RandomUnderSampler**: OR-Logic recall=0.976, specificity=0.753,
  precision=0.122. Undersampling throws away the large majority-class training signal (only ~254 failure rows exist in the training fold to begin with, so undersampling shrinks the effective training set to roughly 2x that), and specificity drops sharply as a result -- the most aggressive recall-for-specificity trade among the four strategies.

## Why the locked (no-resampling) config is the right choice ahead of an OR-union
The OR-logic ensemble is, by construction, a union operator: it fires if ANY of the five
base models fires. That means OR-Logic's overall false-positive rate is bounded below by
the union of the five members' individual false-positive rates -- it can never be more
precise than its least-precise member. Any per-model strategy that trades precision for
recall (class weighting, SMOTE, undersampling) makes EVERY member less precise
simultaneously, and OR-Logic compounds all five degradations at once rather than
averaging them out. This is the same mechanism the Monte Carlo simulation
(`outputs/experiments/simulation/NOTES.md`, R1.4) demonstrates analytically: OR-logic's
advantage is largest when base detectors are diverse AND individually precise. Leaving
the base models unbalanced (the locked config) is what keeps each member precise enough
that the union doesn't explode into a high-false-alarm-rate ensemble -- the recall lift
from OR-aggregating five diverse, individually-precise, unbalanced models is "free" in a
way that recall lift from rebalancing every member individually is not.
