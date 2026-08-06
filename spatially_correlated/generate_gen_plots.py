"""
Create reward plots for Centaur and Llama, each comparing against human baseline.

Layout per model: 2 rows (avg reward, max reward) × 2 cols (rough, smooth)
Each subplot: lines for scenario 0 (accumulation) and scenario 1 (maximization),
              solid = long horizon (10), dashed = short horizon (5)
Human baseline: shown in dark red / black
Model: shown in blue (centaur) / teal (llama)
"""

import os
import sys
import json
import glob
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import plotting_utils

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_IN = os.path.join(BASE, "data/in/experimentData1D.csv")
MODELS = {
    "centaur-70B-adapter": os.path.join(BASE, "data/out/generative/centaur-70B-adapter"),
    "llama-70B-adapter":   os.path.join(BASE, "data/out/generative/llama-70B-adapter"),
}
OUT_DIR = os.path.join(BASE, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
FIGSIZE=(8,6)
BASE_FONT=24

# ── colour palette ─────────────────────────────────────────────────────────────
# Human: scenario 0=dark red, scenario 1=black
# Centaur: scenario 0/1 = toggle below (orange or lila shades)
# Llama: scenario 0=darkorange, scenario 1=saddlebrown

# Toggle: "orange" or "lila" — controls which shades Centaur is drawn in.
CENTAUR_COLOR_MODE = "orange"

# {0: light, 1: dark} — reversed vs. plotting_utils' [dark, light] list order.
CENTAUR_COLOR_OPTIONS = {
    "orange": {0: plotting_utils.CENTAUR_ORANGE[1], 1: plotting_utils.CENTAUR_ORANGE[0]},
    "lila":   {0: plotting_utils.CENTAUR_LILA[1], 1: plotting_utils.CENTAUR_LILA[0]},
}

COLORS = {
    "human":   {0: "#8C8C8C", 1: "#222222"},
    "centaur-70B-adapter": CENTAUR_COLOR_OPTIONS[CENTAUR_COLOR_MODE],
    "llama-70B-adapter":   {0: "#56B4E9", 1: "#0072B2"},
}
HORIZON_STYLE = {5: "--", 10: "-"}
HORIZON_LABEL = {5: "Short horizon (5)", 10: "Long horizon (10)"}
SCENARIO_LABEL = {0: "Accumulation", 1: "Maximization"}
KERNEL_LABEL   = {"rough": "Rough", "smooth": "Smooth"}
MODEL_LABELS   = {
    "centaur-70B-adapter": "Centaur-70B",
    "llama-70B-adapter":   "Llama-Instruct-3.1-70B",
}


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


def make_figure(model_name, human_records, model_records, base_font=20):
    kernels   = ["rough", "smooth"]
    scenarios = [0, 1]
    horizons  = [5, 10]
    metrics   = ["avg", "max"]
    metric_labels = {"avg": "Average reward", "max": "Maximum reward"}

    human_color = COLORS["human"]
    model_color = COLORS[model_name]
    plotting_utils.set_dynamic_fontsize(fig_width=9, base_font=base_font)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="row")
    fig.suptitle(model_name.replace("-", " ").title(), y=1.01)

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

            ax.set_xlabel("Trial")
            ax.set_ylabel(metric_labels[metric])
            ax.spines[["top", "right"]].set_visible(False)
            plotting_utils.style_y_gridlines(ax)
            ax.set_title(KERNEL_LABEL[kernel])

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
                          label=f"{MODEL_LABELS[model_name]} – {SCENARIO_LABEL[s]}"))
    legend_handles.append(mlines.Line2D([], [], color="none", label=""))
    # horizon line styles
    for h in horizons:
        legend_handles.append(
            mlines.Line2D([], [], color="grey", ls=HORIZON_STYLE[h], lw=2,
                          label=HORIZON_LABEL[h]))

    axes[0, 1].legend(handles=legend_handles, frameon=False,
                      loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    return fig


def make_comparison_figure(records_dict, base_font=20):
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

    plotting_utils.set_dynamic_fontsize(fig_width=12, base_font=base_font)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey="row")
    fig.suptitle("Centaur vs Llama", y=1.01)

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

            ax.set_xlabel("Trial Number")
            ax.set_ylabel(metric_labels[metric])
            ax.spines[["top", "right"]].set_visible(False)
            plotting_utils.style_y_gridlines(ax)
            ax.set_title(KERNEL_LABEL[kernel])

    # legend
    legend_handles = []
    for model_name, label in [("centaur-70B-adapter", MODEL_LABELS["centaur-70B-adapter"]),
                               ("llama-70B-adapter",   MODEL_LABELS["llama-70B-adapter"])]:
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
                       loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    return fig


def make_all_figure(human_records, model_records_dict, base_font=20):
    """
    Plot centaur, llama, and human all together.
    Layout: 2 rows (avg, max) × 2 cols (rough, smooth)
    Legend: single horizontal line at the bottom of the figure.
    """
    kernels   = ["rough", "smooth"]
    scenarios = [0, 1]
    horizons  = [5, 10]
    metrics   = ["avg", "max"]
    metric_labels = {"avg": "Average Reward", "max": "Maximum Reward"}

    plotting_utils.set_dynamic_fontsize(fig_width=18, base_font=base_font)
    fig, axes = plt.subplots(2, 2, figsize=(18, 16), sharey="row")
    #fig.suptitle("Centaur, Llama & Human", fontsize=14, y=1.01)

    source_styles = [
        ("human",               human_records,                      2.5, 1.0,  0.20, 4),
        ("centaur-70B-adapter", model_records_dict["centaur-70B-adapter"], 1.8, 0.85, 0.12, 3),
        ("llama-70B-adapter",   model_records_dict["llama-70B-adapter"],   1.8, 0.85, 0.12, 2),
    ]

    for row_idx, metric in enumerate(metrics):
        for col_idx, kernel in enumerate(kernels):
            ax = axes[row_idx, col_idx]
            for source_key, records, lw, alpha, shade_alpha, zorder in source_styles:
                color_map = COLORS[source_key]
                for scenario in scenarios:
                    for horizon in horizons:
                        stats = compute_stats(records, scenario, kernel, horizon)
                        draw_lines(ax, stats,
                                   color=color_map[scenario],
                                   ls=HORIZON_STYLE[horizon],
                                   lw=lw, alpha=alpha, shade_alpha=shade_alpha,
                                   metric=metric, zorder=zorder)
            ax.set_xlabel("Trial Number")
            ax.set_ylabel(metric_labels[metric])
            ax.spines[["top", "right"]].set_visible(False)
            plotting_utils.style_y_gridlines(ax)
            ax.set_title(KERNEL_LABEL[kernel])

    # legend — 3 rows (one per source), 4 entries each: scenario × horizon combined
    short_h = {5: "short", 10: "long"}
    source_order = [
        ("human",               "Human",                            5, 1.0),
        ("centaur-70B-adapter", MODEL_LABELS["centaur-70B-adapter"], 5, 1.0),
        ("llama-70B-adapter",   MODEL_LABELS["llama-70B-adapter"],   5, 1.0),
    ]
    for (source_key, source_label, lw, alpha), y_anchor in zip(
            source_order, [-0.04, -0.11, -0.18]):
        row_handles = [
            mlines.Line2D([], [], color=COLORS[source_key][s],
                          ls=HORIZON_STYLE[h], lw=lw, alpha=alpha,
                          label=f"{source_label} – {SCENARIO_LABEL[s]}, {short_h[h]}")
            for s in scenarios for h in horizons
        ]
        fig.legend(handles=row_handles, frameon=False,
                   loc="lower left", ncol=2,
                   bbox_to_anchor=(0.0, y_anchor))
    plt.tight_layout()
    return fig


def make_avg_reward_long_horizon_figures(human_records, model_records_dict, base_font=BASE_FONT):
    """
    Two separate single-panel figures (rough, smooth), showing only
    average reward, long horizon (10) only — no short-horizon lines.
    """
    kernels   = ["rough", "smooth"]
    scenarios = [0, 1]
    horizon   = 10

    source_styles = [
        ("human",               human_records,                            2.5, 1.0,  0.20, 4),
        ("centaur-70B-adapter", model_records_dict["centaur-70B-adapter"], 1.8, 0.85, 0.12, 3),
        ("llama-70B-adapter",   model_records_dict["llama-70B-adapter"],   1.8, 0.85, 0.12, 2),
    ]
    source_labels = {
        "human":               "Human",
        "centaur-70B-adapter": MODEL_LABELS["centaur-70B-adapter"],
        "llama-70B-adapter":   MODEL_LABELS["llama-70B-adapter"],
    }

    figs = {}
    for kernel in kernels:
        plotting_utils.set_dynamic_fontsize(fig_width=FIGSIZE[0], base_font=base_font)
        fig, ax = plt.subplots(figsize=FIGSIZE)

        for source_key, records, lw, alpha, shade_alpha, zorder in source_styles:
            color_map = COLORS[source_key]
            for scenario in scenarios:
                stats = compute_stats(records, scenario, kernel, horizon)
                draw_lines(ax, stats,
                           color=color_map[scenario], ls="-",
                           lw=lw, alpha=alpha, shade_alpha=shade_alpha,
                           metric="avg", zorder=zorder)

        ax.set_xlabel("Trial")
        ax.set_ylabel("Average reward")
        ax.tick_params(axis='y', length=3, color='#888888', labelcolor='#666666')
        ax.spines[["top", "right"]].set_visible(False)
        plotting_utils.style_y_gridlines(ax)
        #make title bold
        ax.set_title(f"{KERNEL_LABEL[kernel]}",weight='bold')

        plt.tight_layout()
        figs[kernel] = fig
    return figs


def make_avg_reward_long_horizon_legend(base_font=20):
    """Standalone legend figure matching make_avg_reward_long_horizon_figures."""
    plotting_utils.set_dynamic_fontsize(fig_width=6, base_font=base_font)
    scenarios = [0, 1]
    source_styles = [
        ("human",               2.5, 1.0),
        ("centaur-70B-adapter", 1.8, 0.85),
        ("llama-70B-adapter",   1.8, 0.85),
    ]
    source_labels = {
        "human":               "Human",
        "centaur-70B-adapter": MODEL_LABELS["centaur-70B-adapter"],
        "llama-70B-adapter":   MODEL_LABELS["llama-70B-adapter"],
    }

    legend_handles = [
        mlines.Line2D([], [], color=COLORS[source_key][s], lw=lw, alpha=alpha,
                      label=f"{source_labels[source_key]} – {SCENARIO_LABEL[s]}")
        for source_key, lw, alpha in source_styles
        for s in scenarios
    ]

    fig = plt.figure(figsize=(3.5, 2.2))
    fig.legend(handles=legend_handles, frameon=False, loc="center")
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
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    print("Loading all model records...")
    model_records_dict = {name: load_model(path) for name, path in MODELS.items()}

    # Centaur vs Llama comparison (no human)
    print("Creating centaur vs llama comparison...")
    fig = make_comparison_figure(model_records_dict)
    out_path = os.path.join(OUT_DIR, "reward_by_trial_centaur_vs_llama.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # All three together
    print("Creating centaur + llama + human combined figure...")
    fig = make_all_figure(human_records, model_records_dict)
    out_path = os.path.join(OUT_DIR, "reward_by_trial_all.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Average reward only, long horizon only, rough/smooth as separate figures
    print("Creating average-reward long-horizon rough/smooth figures...")
    avg_figs = make_avg_reward_long_horizon_figures(human_records, model_records_dict)
    for kernel, fig in avg_figs.items():
        out_path = os.path.join(OUT_DIR, f"avg_reward_long_horizon_{kernel}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    legend_fig = make_avg_reward_long_horizon_legend()
    legend_path = os.path.join(OUT_DIR, "avg_reward_long_horizon_legend.png")
    legend_fig.savefig(legend_path, dpi=150, bbox_inches="tight")
    plt.close(legend_fig)
    print(f"  Saved {legend_path}")


def sweep_fontsizes(base_fonts=(14, 16, 18, 20, 22, 24, 26)):
    """
    Rerun every plotting function once per candidate base_font, so you can
    compare them side by side instead of manually editing base_font and
    rerunning the script each time.

    Output goes to figures/fontsize_sweep/<base_font>/, mirroring the
    filenames main() produces, one subfolder per candidate size.
    """
    print("Loading human data...")
    human_records = load_human(DATA_IN)
    print("Loading model data...")
    model_records_dict = {name: load_model(path) for name, path in MODELS.items()}

    sweep_root = os.path.join(OUT_DIR, "fontsize_sweep")
    os.makedirs(sweep_root, exist_ok=True)

    for base_font in base_fonts:
        sweep_dir = os.path.join(sweep_root, str(base_font))
        os.makedirs(sweep_dir, exist_ok=True)
        print(f"Rendering figures at base_font={base_font} -> {sweep_dir}")

        for model_name, model_records in model_records_dict.items():
            fig = make_figure(model_name, human_records, model_records, base_font=base_font)
            fig.savefig(os.path.join(sweep_dir, f"reward_by_trial_{model_name}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        fig = make_comparison_figure(model_records_dict, base_font=base_font)
        fig.savefig(os.path.join(sweep_dir, "reward_by_trial_centaur_vs_llama.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig = make_all_figure(human_records, model_records_dict, base_font=base_font)
        fig.savefig(os.path.join(sweep_dir, "reward_by_trial_all.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

        avg_figs = make_avg_reward_long_horizon_figures(human_records, model_records_dict, base_font=base_font)
        for kernel, fig in avg_figs.items():
            fig.savefig(os.path.join(sweep_dir, f"avg_reward_long_horizon_{kernel}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        legend_fig = make_avg_reward_long_horizon_legend(base_font=base_font)
        legend_fig.savefig(os.path.join(sweep_dir, "avg_reward_long_horizon_legend.png"),
                            dpi=150, bbox_inches="tight")
        plt.close(legend_fig)

    print(f"Done. Compare folders under {sweep_root}")


if __name__ == "__main__":
    if "--sweep-fontsizes" in sys.argv:
        idx = sys.argv.index("--sweep-fontsizes")
        sizes = [int(s) for s in sys.argv[idx + 1:] if s.lstrip("-").isdigit()]
        sweep_fontsizes(tuple(sizes) if sizes else (14, 16, 18, 20, 22, 24, 26))
    else:
        main()
