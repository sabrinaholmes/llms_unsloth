"""
Create reward plots for Centaur and Llama, each comparing against human baseline.

Layout per model: 2 rows (avg reward, max reward) × 2 cols (rough, smooth)
Each subplot: lines for scenario 0 (accumulation) and scenario 1 (maximization),
              solid = long horizon (10), dashed = short horizon (5)
Human baseline: shown in dark red / black
Model: shown in blue (centaur) / teal (llama)
"""

import os
import json
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_IN = os.path.join(BASE, "data/in/experimentData1D.csv")
MODELS = {
    "centaur-70B-adapter": os.path.join(BASE, "data/out/generative/centaur-70B-adapter"),
    "llama-70B-adapter":   os.path.join(BASE, "data/out/generative/llama-70B-adapter"),
}
OUT_DIR = os.path.join(BASE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
# Human: scenario 0=dark red, scenario 1=black
# Centaur: scenario 0=steelblue, scenario 1=navy
# Llama: scenario 0=darkorange, scenario 1=saddlebrown
COLORS = {
    "human":   {0: "#B22222", 1: "#222222"},
    "centaur-70B-adapter": {0: "#E69F00", 1: "#D55E00"},
    "llama-70B-adapter":   {0: "#56B4E9", 1: "#0072B2"},
}
HORIZON_STYLE = {5: "--", 10: "-"}
HORIZON_LABEL = {5: "Short horizon (5)", 10: "Long horizon (10)"}
SCENARIO_LABEL = {0: "Accumulation", 1: "Maximization"}
KERNEL_LABEL   = {"rough": "Rough", "smooth": "Smooth"}


# ── data loading ───────────────────────────────────────────────────────────────

def load_human(csv_path):
    """
    Returns list of dicts:
      participant, scenario, kernel (rough/smooth), horizon (5/10), rewards (list, init removed)
    kernel 0 → rough, kernel 1 → smooth
    horizon determined by round length: 6 → short (5 choices), 11 → long (10 choices)
    First element of each round is the revealed init click → dropped.
    """
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        sh = json.loads(row["searchHistory"])
        scenario = int(row["scenario"])
        kernel   = "rough" if int(row["kernel"]) == 0 else "smooth"
        pid      = row["id"]
        for round_rewards in sh["ycollect"]:
            n = len(round_rewards)
            if n == 6:
                horizon = 5
            elif n == 11:
                horizon = 10
            else:
                continue  # unexpected length
            # drop the first (init) click
            rewards = round_rewards[1:]
            records.append(dict(participant=pid, scenario=scenario,
                                kernel=kernel, horizon=horizon, rewards=rewards))
    return records


def load_model(folder_path):
    """
    Returns list of dicts with same keys as load_human.
    Trials are already 1-indexed; no init reward included (it's separate).
    """
    singles = os.path.join(folder_path, "singles")
    records = []
    for fpath in glob.glob(os.path.join(singles, "participant_*.csv")):
        bn = os.path.basename(fpath)
        m = re.search(r"participant_(\d+)", bn)
        if not m:
            continue
        pid = int(m.group(1))
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        for (env_idx, horizon_val), grp in df.groupby(["env_idx", "horizon"]):
            grp = grp.sort_values("trial")
            scenario = int(grp["scenario"].iloc[0])
            kernel   = str(grp["kernel"].iloc[0])
            horizon  = int(horizon_val)
            rewards  = grp["reward"].tolist()
            records.append(dict(participant=pid, scenario=scenario,
                                kernel=kernel, horizon=horizon, rewards=rewards))
    return records


# ── aggregation ────────────────────────────────────────────────────────────────

def running_max(rewards):
    """Cumulative maximum over the reward sequence."""
    out = []
    m = -np.inf
    for r in rewards:
        m = max(m, r)
        out.append(m)
    return out


def compute_stats(records, scenario, kernel, horizon):
    """
    For the subset matching (scenario, kernel, horizon), compute
    per-participant mean-reward and running-max arrays, then return
    (trials, mean_avg, sem_avg, mean_max, sem_max).
    """
    subsets = [r["rewards"] for r in records
               if r["scenario"] == scenario
               and r["kernel"] == kernel
               and r["horizon"] == horizon]
    if not subsets:
        return None

    length = min(len(s) for s in subsets)
    trimmed = np.array([s[:length] for s in subsets], dtype=float)
    maxs    = np.array([running_max(s[:length]) for s in subsets], dtype=float)

    n = trimmed.shape[0]
    trials  = np.arange(1, length + 1)
    mean_avg = trimmed.mean(axis=0)
    sem_avg  = trimmed.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(length)
    mean_max = maxs.mean(axis=0)
    sem_max  = maxs.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros(length)

    return trials, mean_avg, sem_avg, mean_max, sem_max


# ── plotting ───────────────────────────────────────────────────────────────────

def draw_lines(ax, stats, color, ls, lw=2.0, alpha=1.0, shade_alpha=0.15,
               metric="avg", zorder=2):
    if stats is None:
        return
    trials, mean_avg, sem_avg, mean_max, sem_max = stats
    mean = mean_avg if metric == "avg" else mean_max
    sem  = sem_avg  if metric == "avg" else sem_max
    ax.plot(trials, mean, color=color, ls=ls, lw=lw, alpha=alpha, zorder=zorder)
    ax.fill_between(trials, mean - sem, mean + sem,
                    color=color, alpha=shade_alpha, zorder=zorder - 1)


def make_figure(model_name, human_records, model_records):
    kernels   = ["rough", "smooth"]
    scenarios = [0, 1]
    horizons  = [5, 10]
    metrics   = ["avg", "max"]
    metric_labels = {"avg": "Average reward", "max": "Maximum reward"}

    human_color = COLORS["human"]
    model_color = COLORS[model_name]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="row")
    fig.suptitle(model_name.replace("-", " ").title(), fontsize=14, y=1.01)

    for row_idx, metric in enumerate(metrics):
        for col_idx, kernel in enumerate(kernels):
            ax = axes[row_idx, col_idx]

            for scenario in scenarios:
                for horizon in horizons:
                    ls = HORIZON_STYLE[horizon]

                    # human
                    h_stats = compute_stats(human_records, scenario, kernel, horizon)
                    draw_lines(ax, h_stats,
                               color=human_color[scenario], ls=ls,
                               lw=2.5, alpha=1.0, shade_alpha=0.20,
                               metric=metric, zorder=4)

                    # model (behind human)
                    m_stats = compute_stats(model_records, scenario, kernel, horizon)
                    draw_lines(ax, m_stats,
                               color=model_color[scenario], ls=ls,
                               lw=1.8, alpha=0.75, shade_alpha=0.10,
                               metric=metric, zorder=2)

            ax.set_xlabel("Trial", fontsize=10)
            ax.set_ylabel(metric_labels[metric], fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(KERNEL_LABEL[kernel], fontsize=11)

    # ── legend ─────────────────────────────────────────────────────────────────
    legend_handles = []
    # scenario colours — human
    for s in scenarios:
        legend_handles.append(
            mlines.Line2D([], [], color=human_color[s], lw=2.5,
                          label=f"Human – {SCENARIO_LABEL[s]}"))
    legend_handles.append(mlines.Line2D([], [], color="none", label=""))
    # scenario colours — model
    for s in scenarios:
        legend_handles.append(
            mlines.Line2D([], [], color=model_color[s], lw=1.8, alpha=0.75,
                          label=f"{model_name.split('-')[0].capitalize()} – {SCENARIO_LABEL[s]}"))
    legend_handles.append(mlines.Line2D([], [], color="none", label=""))
    # horizon line styles
    for h in horizons:
        legend_handles.append(
            mlines.Line2D([], [], color="grey", ls=HORIZON_STYLE[h], lw=2,
                          label=HORIZON_LABEL[h]))

    axes[0, 1].legend(handles=legend_handles, frameon=False,
                      fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    return fig


def make_comparison_figure(records_dict):
    """
    Centaur vs Llama without human data.
    records_dict: {"centaur-70B-adapter": [...], "llama-70B-adapter": [...]}
    Layout: 2 rows (avg, max) × 2 cols (rough, smooth)
    """
    kernels   = ["rough", "smooth"]
    scenarios = [0, 1]
    horizons  = [5, 10]
    metrics   = ["avg", "max"]
    metric_labels = {"avg": "Average reward", "max": "Maximum reward"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="row")
    fig.suptitle("Centaur vs Llama", fontsize=14, y=1.01)

    for row_idx, metric in enumerate(metrics):
        for col_idx, kernel in enumerate(kernels):
            ax = axes[row_idx, col_idx]

            for model_name, model_records in records_dict.items():
                color_map = COLORS[model_name]
                for scenario in scenarios:
                    for horizon in horizons:
                        ls = HORIZON_STYLE[horizon]
                        stats = compute_stats(model_records, scenario, kernel, horizon)
                        draw_lines(ax, stats,
                                   color=color_map[scenario], ls=ls,
                                   lw=2.0, alpha=1.0, shade_alpha=0.15,
                                   metric=metric, zorder=2)

            ax.set_xlabel("Trial", fontsize=10)
            ax.set_ylabel(metric_labels[metric], fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_title(KERNEL_LABEL[kernel], fontsize=11)

    # legend
    legend_handles = []
    for model_name, label in [("centaur-70B-adapter", "Centaur"),
                               ("llama-70B-adapter", "Llama")]:
        color_map = COLORS[model_name]
        for s in scenarios:
            legend_handles.append(
                mlines.Line2D([], [], color=color_map[s], lw=2,
                              label=f"{label} – {SCENARIO_LABEL[s]}"))
        legend_handles.append(mlines.Line2D([], [], color="none", label=""))
    for h in horizons:
        legend_handles.append(
            mlines.Line2D([], [], color="grey", ls=HORIZON_STYLE[h], lw=2,
                          label=HORIZON_LABEL[h]))

    axes[0, 1].legend(handles=legend_handles, frameon=False,
                      fontsize=8.5, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    return fig


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading human data...")
    human_records = load_human(DATA_IN)
    print(f"  {len(human_records)} human rounds loaded")

    for model_name, model_path in MODELS.items():
        print(f"Loading {model_name}...")
        model_records = load_model(model_path)
        print(f"  {len(model_records)} model rounds loaded")

        fig = make_figure(model_name, human_records, model_records)
        out_path = os.path.join(OUT_DIR, f"reward_by_trial_{model_name}.png")
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
        plt.close(fig)
        print(f"  Saved {out_path}")

    # Centaur vs Llama comparison (no human)
    print("Creating centaur vs llama comparison...")
    model_records_dict = {name: load_model(path) for name, path in MODELS.items()}
    fig = make_comparison_figure(model_records_dict)
    out_path = os.path.join(OUT_DIR, "reward_by_trial_centaur_vs_llama.png")
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.02, dpi=300, transparent=True)
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()
