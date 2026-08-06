"""
Rerun every plotting script in llms_unsloth in one pass -- e.g. after tweaking
plotting_utils._FONT_ROLE_MULTIPLIERS, to regenerate all figures with the new
font settings without manually invoking each script by hand.

Each entry below is run as a subprocess (python <script> [args]) from a fixed
cwd, exactly matching how the script is normally invoked by hand. Subprocess
isolation sidesteps the fact these scripts weren't written to be imported --
several execute plotting code at module level, one calls os.chdir(), one
calls sys.exit() under a caching guard, and three files share the literal
name generate_gen_plots.py -- none of that leaks into other scripts or into
this driver.

Usage:
    python rerun_all_plots.py                          # run everything
    python rerun_all_plots.py --list                   # show registry, don't run
    python rerun_all_plots.py --only rl_waltmann_generate_gen_plots predictive_plots
"""
import argparse
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

PREDICTIVE_TASKS = ["rl", "horizon", "spatially_correlated", "multi_cue_judgment", "rl_waltmann"]


def _abs(*parts):
    return os.path.join(REPO_ROOT, *parts)


SCRIPTS = [
    dict(name="plot_panel_a",
         script=_abs("plot_panel_a.py"), cwd=REPO_ROOT, args=[]),
    dict(name="predictive_plots",
         script=_abs("predictive_plots.py"), cwd=REPO_ROOT,
         arg_variants=[["--task", t] for t in PREDICTIVE_TASKS]),
    dict(name="horizon_generate_gen_plots",
         script=_abs("horizon", "generate_gen_plots.py"), cwd=REPO_ROOT, args=[]),
    dict(name="multi_cue_plot_rmse_training",
         script=_abs("multi_cue_judgment", "plot_rmse_training.py"),
         cwd=_abs("multi_cue_judgment"), args=[]),
    dict(name="rl_analyze_strategies",
         script=_abs("rl", "analyze_strategies.py"), cwd=_abs("rl"), args=[]),
    dict(name="rl_generate_gen_plots",
         script=_abs("rl", "generate_gen_plots.py"), cwd=REPO_ROOT,
         args=["--base", _abs("rl", "data", "out", "generative"),
               "--out", _abs("rl", "figures")]),
    dict(name="rl_waltmann_generate_gen_plots",
         script=_abs("rl_waltmann", "generate_gen_plots.py"),
         cwd=_abs("rl_waltmann"), args=[]),
    dict(name="rl_waltmann_plot_no_reward",
         script=_abs("rl_waltmann", "plot_no_reward.py"),
         cwd=_abs("rl_waltmann"), args=[]),
    dict(name="rl_waltmann_transition_analysis",
         script=_abs("rl_waltmann", "transition_analysis.py"),
         cwd=_abs("rl_waltmann"), args=[]),
    dict(name="simulated_rl_reversal_learning_figure",
         script=_abs("simulated_rl", "reversal_learning_figure.py"),
         cwd=_abs("simulated_rl"), args=[]),
    dict(name="spatially_correlated_generative_plots",
         script=_abs("spatially_correlated", "generative_plots.py"),
         cwd=_abs("spatially_correlated"), args=[]),
]


def run_one(entry, args, timeout):
    env = dict(os.environ, MPLBACKEND="Agg")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, entry["script"], *args],
            cwd=entry["cwd"], env=env, timeout=timeout,
            capture_output=True, text=True,
        )
        elapsed = time.time() - t0
        return proc.returncode == 0, elapsed, proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - t0
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        return False, elapsed, stdout, stderr + "\n[TIMEOUT]", None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", metavar="NAME",
                         help="Run only these registered script names (see --list)")
    parser.add_argument("--list", action="store_true",
                         help="List registered scripts and exit")
    parser.add_argument("--timeout", type=int, default=600,
                         help="Per-invocation timeout in seconds (default: 600)")
    parsed = parser.parse_args()

    if parsed.list:
        for entry in SCRIPTS:
            print(f"{entry['name']:45s} -> {entry['script']}")
        return

    to_run = [e for e in SCRIPTS if not parsed.only or e["name"] in parsed.only]
    if parsed.only:
        unknown = set(parsed.only) - {e["name"] for e in SCRIPTS}
        if unknown:
            parser.error(f"unknown script name(s): {', '.join(sorted(unknown))}")

    results = []
    for entry in to_run:
        variants = entry.get("arg_variants") or [entry.get("args", [])]
        for args in variants:
            label = entry["name"] if len(variants) == 1 else f"{entry['name']} {' '.join(args)}"
            print(f"\n=== {label} ===")
            if entry.get("note"):
                print(f"  note: {entry['note']}")
            ok, elapsed, stdout, stderr, rc = run_one(entry, args, timeout=parsed.timeout)
            if stdout:
                print(stdout, end="" if stdout.endswith("\n") else "\n")
            if stderr:
                print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
            status = "OK" if ok else "FAILED"
            print(f"  -> {status} (exit={rc}, {elapsed:.1f}s)")
            results.append((label, status, elapsed))

    print("\n" + "=" * 60)
    print("SUMMARY")
    for label, status, elapsed in results:
        print(f"  [{status:6s}] {label} ({elapsed:.1f}s)")
    n_failed = sum(1 for _, status, _ in results if status == "FAILED")
    print(f"\n{len(results)} run, {n_failed} failed")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()
