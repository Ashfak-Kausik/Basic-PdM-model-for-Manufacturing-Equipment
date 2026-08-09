# MLP baseline notes (R1.3 / R1.11)

## What this is, and isn't

A single, lightly-tuned `MLPClassifier(hidden_layer_sizes=(64, 32),
activation='relu', alpha=1e-4, max_iter=500, early_stopping=True,
n_iter_no_change=15, random_state=42)` inside a `StandardScaler` Pipeline,
fit on the same modified nine-feature set as the five locked base learners.
No `class_weight` / rebalancing, matching the locked base learners' unbalanced
setup, so the comparison is apples-to-apples.

This architecture was chosen as a **standard, off-the-shelf small-tabular
MLP** -- two hidden layers, ReLU, mild L2 (`alpha=1e-4`), early stopping so
training doesn't run the full 500 epochs by default. It was **not** hyper-
parameter-searched or tuned to any metric. No result below was used to pick
or adjust the config after the fact; this is the first and only run.

The MLP is a **standalone comparison baseline only**: it is not in
`src/config.py`'s five canonical base learners, and it does not feed into
Traditional Voting or OR-Logic. All existing committed numbers are untouched.

## Verdict: MLP clearly underperforms GBC on recall

| | SEED=42 recall | 30-seed mean recall | AUC (SEED=42) |
|---|---|---|---|
| **MLP** | 0.494 (42/85 failures caught) | 0.481 [0.236, 0.684] | 0.977 |
| **GBC** | 0.800 (68/85) | 0.739 [0.661, 0.827] (from `stability_summary.csv`) | 0.976 |
| **OR-Logic** | 0.800 (68/85) | 0.776 [0.703, 0.839] (from `stability_summary.csv`) | 0.898 |

Paired Wilcoxon signed-rank test, MLP vs GBC recall across the same 30
stability seeds (`mlp_vs_gbc_paired.csv`): GBC has higher recall than the
MLP in **all 30/30 seeds** (0 ties, 0 MLP wins), mean diff = -0.258 (-25.8 pp),
Wilcoxon p = 1.73e-6, rank-biserial effect size = -1.0. This is a large,
completely consistent effect, not a marginal one -- **the MLP is clearly
worse than GBC on recall**, both on the single locked split and across
resamples.

Precision (0.792) and specificity (0.995) are comparable to GBC's, so this
isn't a case of the MLP trading recall for a large precision/specificity
gain elsewhere -- it is simply catching fewer of the actual failures.

## Flag: counter to naive expectation -- AUC says the MLP and GBC are near-identical

The MLP's ROC AUC (0.977) is statistically indistinguishable from GBC's
(0.976) on the SEED=42 split -- i.e. the MLP's *ranking* of failure
probability is essentially as good as GBC's. The large recall gap therefore
looks like a **default-threshold artifact**, not a genuine discriminative-
power gap: at `predict()`'s default 0.5 cutoff, the MLP is more conservative
about calling "failure" on this ~3.4%-positive class than GBC is, so it
under-calls positives despite ranking them correctly. This is worth flagging
honestly rather than glossing over -- a threshold-tuned or calibrated MLP
might close much of this recall gap, but that is exactly the kind of
per-metric tuning this baseline deliberately avoids (see "What this is,
and isn't" above). The as-shipped, default-threshold MLP is the fair,
untouched comparison point, and on that comparison it underperforms.

## Files

- `mlp_results.csv` -- MLP SEED=42 full metrics + AUC, with GBC/OR-Logic
  context rows read verbatim from `outputs/results_clean.csv` (metrics) and
  `outputs/experiments/autofeat/autofeat_results.csv` (AUC, same SEED=42
  modified split) -- neither refit here.
- `mlp_stability_raw.csv` -- MLP metrics for each of the same 30 seeds used
  in `outputs/experiments/stability/stability_raw.csv`.
- `mlp_stability.csv` -- mean + 95% CI (2.5/97.5 percentile) over those 30
  seeds for recall, precision, specificity, FN rate, MCC.
- `mlp_vs_gbc_paired.csv` -- Wilcoxon signed-rank test, MLP vs GBC recall,
  paired per seed. GBC's per-seed recall is read from
  `outputs/experiments/stability/stability_raw.csv`, not refit.
