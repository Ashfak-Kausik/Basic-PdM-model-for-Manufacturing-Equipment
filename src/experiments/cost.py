"""
Reviewer R1.9 -- inference latency & model size.

On the LOCKED SEED=42 modified-data split, measures per-model fit time,
total-test predict time, and per-sample inference latency (all averaged
over 10 repeats), plus serialized (joblib) model size. Both ensembles are
included: ensemble latency = sum of the five member models' predict
latencies + the (tiny) aggregation step (stacking predictions into a
DataFrame and thresholding the row-sum).

Everything here reuses the locked model factory (src/models.py) as-is --
no hyperparameter changes. All models in the locked config already run
single-threaded by default (none of SVM/RF/KNN/LR/GBC set n_jobs > 1), so
these numbers are directly CPU-single-thread comparable without further
changes.

Outputs:
  outputs/experiments/cost/system_info.txt
  outputs/experiments/cost/cost_summary.csv
  outputs/experiments/cost/latency_bar.png
"""
import platform
import re
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone

from src.experiments.common import BASE_MODEL_NAMES, EXPERIMENTS_DIR, locked_seed42_split, ensembles_from_preds
from src.models import build_models

OUT_DIR = EXPERIMENTS_DIR / "cost"
N_REPEATS = 10


def get_system_info() -> str:
    cpu_model = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            m = re.search(r"model name\s*:\s*(.+)", f.read())
            if m:
                cpu_model = m.group(1).strip()
    except OSError:
        pass

    ram_gb = "unknown"
    try:
        with open("/proc/meminfo") as f:
            m = re.search(r"MemTotal:\s*(\d+)\s*kB", f.read())
            if m:
                ram_gb = f"{int(m.group(1)) / (1024 * 1024):.1f} GB"
    except OSError:
        pass

    import os

    lines = [
        "System info (measured once, for the latency numbers below)",
        "=" * 60,
        f"CPU: {cpu_model}",
        f"Logical CPUs available: {os.cpu_count()}",
        f"RAM: {ram_gb}",
        f"Python: {platform.python_version()}  |  Platform: {platform.platform()}",
        "All locked-config models run single-threaded by default (no n_jobs>1 "
        "set anywhere in src/config.py), so these numbers are single-thread "
        "CPU-comparable across models.",
    ]
    return "\n".join(lines) + "\n"


def model_size_kb(fitted_pipe) -> float:
    buf = BytesIO()
    joblib.dump(fitted_pipe, buf)
    return len(buf.getbuffer()) / 1024.0


def time_repeats(fn, n=N_REPEATS):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = pd.Series(times)
    return arr.mean(), arr.std(ddof=1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sysinfo = get_system_info()
    (OUT_DIR / "system_info.txt").write_text(sysinfo)
    print(sysinfo)

    X_train, X_test, y_train, y_test = locked_seed42_split()
    n_test = len(X_test)

    rows = []
    fitted = {}
    member_latency_s = {}  # mean total predict time per model, for ensemble rollup

    for name in BASE_MODEL_NAMES:
        template = build_models()[name]

        # Fit time: fresh unfit clone each repeat, so later repeats aren't
        # measuring a refit-over-fitted-state shortcut.
        def do_fit(name=name, template=template):
            pipe = clone(template)
            pipe.fit(X_train, y_train)
            fitted[name] = pipe  # keep the last-fit copy for predict timing / size

        fit_mean, fit_std = time_repeats(do_fit)

        pipe = fitted[name]

        def do_predict(pipe=pipe):
            pipe.predict(X_test)

        predict_mean, predict_std = time_repeats(do_predict)
        member_latency_s[name] = predict_mean

        per_sample_us_mean = (predict_mean / n_test) * 1e6
        per_sample_us_std = (predict_std / n_test) * 1e6

        size_kb = model_size_kb(pipe)

        rows.append(
            {
                "model_or_ensemble": name,
                "fit_time_s_mean": fit_mean,
                "fit_time_s_std": fit_std,
                "predict_time_total_s_mean": predict_mean,
                "predict_time_total_s_std": predict_std,
                "per_sample_latency_us_mean": per_sample_us_mean,
                "per_sample_latency_us_std": per_sample_us_std,
                "model_size_kb": size_kb,
                "n_test": n_test,
            }
        )
        print(f"{name}: fit={fit_mean * 1000:.2f}ms  predict_total={predict_mean * 1000:.3f}ms  "
              f"per_sample={per_sample_us_mean:.2f}us  size={size_kb:.1f}KB")

    # --- ensembles: sum of member predict latencies + aggregation step ---
    preds = {name: fitted[name].predict(X_test) for name in BASE_MODEL_NAMES}

    def do_aggregate():
        ensembles_from_preds(preds, X_test.index)

    agg_mean, agg_std = time_repeats(do_aggregate)

    sum_member_predict = sum(member_latency_s.values())
    sum_member_size = sum(r["model_size_kb"] for r in rows)

    for ens_name in ["Traditional Voting", "OR-Logic"]:
        ens_predict_total = sum_member_predict + agg_mean
        per_sample_us = (ens_predict_total / n_test) * 1e6
        rows.append(
            {
                "model_or_ensemble": ens_name,
                "fit_time_s_mean": sum(r["fit_time_s_mean"] for r in rows if r["model_or_ensemble"] in BASE_MODEL_NAMES),
                "fit_time_s_std": float("nan"),  # sum of independent stds not meaningful as a single std
                "predict_time_total_s_mean": ens_predict_total,
                "predict_time_total_s_std": float("nan"),
                "per_sample_latency_us_mean": per_sample_us,
                "per_sample_latency_us_std": float("nan"),
                "model_size_kb": sum_member_size,
                "n_test": n_test,
                "note": f"= sum of 5 member predict times ({sum_member_predict * 1000:.3f}ms) + aggregation ({agg_mean * 1000:.4f}ms)",
            }
        )
        print(f"{ens_name}: predict_total={ens_predict_total * 1000:.3f}ms  per_sample={per_sample_us:.2f}us  "
              f"size={sum_member_size:.1f}KB (sum of 5 members)")

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "cost_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    # --- bar chart: per-sample latency ---
    order = BASE_MODEL_NAMES + ["Traditional Voting", "OR-Logic"]
    plot_df = df.set_index("model_or_ensemble").loc[order]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4c72b0"] * len(BASE_MODEL_NAMES) + ["#dd8452", "#c44e52"]
    ax.bar(order, plot_df["per_sample_latency_us_mean"], color=colors)
    ax.set_ylabel("Per-sample inference latency (microseconds)")
    ax.set_title(f"Inference latency by model/ensemble (n_test={n_test}, {N_REPEATS} repeats, single-thread CPU)")
    ax.set_yscale("log")
    for i, v in enumerate(plot_df["per_sample_latency_us_mean"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latency_bar.png", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'latency_bar.png'}")


if __name__ == "__main__":
    main()
