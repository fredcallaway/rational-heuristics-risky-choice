#!/usr/bin/env python
"""Regenerate the four corrected line figures in new-figs/, self-contained.

These are the Experiment 1 "condition line" figures whose dispersion (alpha^-1) panels
were plotted without the inverse-alpha flip in the published paper. This script is a
trimmed COPY of code/python/make_figures.py::exp1_condition_lines (and the one helper it
needs, process_data.errors), restricted to the four panels that the correction replaces,
with the dispersion flip applied. It does NOT import or run the main pipeline, touch
stats, or recluster; it only reads the already-processed model/human trial CSVs.

Output -> <repo>/new-figs/<figN>_<params>.png. By default it writes to a temp dir and
diffs against the committed new-figs/ instead of overwriting; pass --write to overwrite.

    python make_new_figs.py            # regenerate to temp, diff vs committed new-figs/
    python make_new_figs.py --write    # overwrite new-figs/

The flip is the `# flip for inverse alpha` lines, matching make_figures.py:397,402.
"""

import argparse
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, ".."))
REPO_ROOT = os.path.normpath(os.path.join(CODE, ".."))
DATA = os.path.join(CODE, "data")

MODEL = os.path.join(DATA, "model", "exp1", "processed", "trials.csv")
MODEL_EXCLUDE = os.path.join(DATA, "model", "exp1", "processed", "trials_exclude.csv")
HUMAN = os.path.join(DATA, "human", "1.0", "processed", "trials.csv")
HUMAN_EXCLUDE = os.path.join(DATA, "human", "1.0", "processed", "trials_exclude.csv")

# The four replacement figures, keyed by their new-figs/ filename. Each is one or two
# rows of (param, y-axis label, exclude-flag, y-limits), exactly as in make_figures.py.
FIGURES = {
    "fig5_nr_clicks__processing_pattern.png": [
        ("nr_clicks", "Information Gathered", False, (2, 25)),
        ("processing_pattern", "Processing Pattern\n(attribute → alternative)", False, (-1, -.2)),
    ],
    "figE2_click_var_outcome__click_var_gamble.png": [
        ("click_var_outcome", "Attribute Variance", False, (0, .2)),
        ("click_var_gamble", "Alternative Variance", False, (0, .06)),
    ],
    "fig6_payoff_gross_relative.png": [
        ("payoff_gross_relative", "Decision Quality", False, (0, 1.05)),
    ],
    "figF1_payoff_gross_relative_exclude.png": [
        ("payoff_gross_relative", "Decision Quality\n(with exclusions)", True, (0, 1.05)),
    ],
}


def errors(dat):
    """Bootstrap 95% CI half-widths for a 1-D array (copied from process_data.errors)."""
    X = np.nanmean(dat)
    ci = [np.nanmean(np.random.choice(dat, len(dat))) - X for _ in range(int(1e4))]
    return abs(np.percentile(ci, [2.5, 97.5]))


def load(exclude):
    model = pd.read_csv(MODEL_EXCLUDE if exclude else MODEL, low_memory=False)
    human = pd.read_csv(HUMAN_EXCLUDE if exclude else HUMAN, low_memory=False)
    return model, human


def make_figure(rows, out_path):
    fig = plt.figure(figsize=(32, 8 * len(rows)))
    gs = gridspec.GridSpec(len(rows), 3, width_ratios=[1, 1, 1.25])
    ft_ticks, ft_labels, ft_legend = 32, 42, 36

    for r, (param, label, exclude, ylim) in enumerate(rows):
        df_model, df_human = load(exclude)
        last = (r + 1) == len(rows)

        # ----- stakes (sigma) -----
        plt.sca(plt.subplot(gs[3 * r]))
        plt.plot(df_model.groupby("sigma")[param].mean().values, color="#17becf", lw=8)
        dat = df_human.groupby(["sigma", "pid"])[param].mean()
        y = [dat.loc[i].mean() for i in dat.index.levels[0]]
        err = np.array([errors(dat.loc[i]) for i in dat.index.levels[0]]).T
        plt.errorbar(range(len(y)), y, yerr=err, color="#1f77b4", lw=8)
        plt.ylabel(label, fontsize=ft_labels)
        plt.xticks(np.arange(len(y)), dat.index.levels[0], fontsize=ft_ticks)
        plt.ylim(ylim); plt.yticks(fontsize=ft_ticks); plt.grid(True)
        if last:
            plt.xlabel(r"Stakes [$\sigma$]", fontsize=ft_labels)
        else:
            plt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # ----- dispersion (alpha) -- FLIPPED for inverse alpha -----
        plt.sca(plt.subplot(gs[3 * r + 1]))
        dat_m = df_model.groupby("alpha")[param].mean().values
        plt.plot(np.flip(dat_m), color="#17becf", lw=8)  # flip for inverse alpha
        dat = df_human.groupby(["alpha", "pid"])[param].mean()
        y = [dat.loc[i].mean() for i in dat.index.levels[0]]
        err = np.array([errors(dat.loc[i]) for i in dat.index.levels[0]]).T
        plt.errorbar(range(len(y)), np.flip(y), yerr=np.flip(err), color="#1f77b4", lw=8)  # flip
        plt.ylim(ylim); plt.yticks(fontsize=ft_ticks); plt.grid(True)
        if last:
            plt.xlabel(r"Dispersion [$\alpha^{-1}$]", fontsize=ft_labels)
            plt.xticks(np.arange(len(y)),
                       [r"$10^{-1.0}$", r"$10^{-0.5}$", r"$10^{0.0}$", r"$10^{0.5}$", r"$10^{1.0}$"],
                       fontsize=ft_ticks)
            plt.tick_params(axis="y", which="both", left=False, labelleft=False)
        else:
            plt.xticks(np.arange(len(y)), dat.index.levels[0], fontsize=ft_ticks)
            plt.tick_params(axis="both", which="both", left=False, labelleft=False,
                            bottom=False, labelbottom=False)

        # ----- cost -----
        plt.sca(plt.subplot(gs[3 * r + 2]))
        plt.plot(df_model.groupby("cost")[param].mean().values, color="#17becf", lw=8)
        dat = df_human.groupby(["cost", "pid"])[param].mean()
        y = [dat.loc[i].mean() for i in dat.index.levels[0]]
        err = np.array([errors(dat.loc[i]) for i in dat.index.levels[0]]).T
        plt.errorbar(range(len(y)), y, yerr=err, color="#1f77b4", lw=8)
        plt.xticks(np.arange(len(y)), dat.index.levels[0], fontsize=ft_ticks)
        plt.ylim(ylim); plt.grid(True)
        if last:
            plt.xlabel(r"Cost [$\lambda$]", fontsize=ft_labels)
            plt.tick_params(axis="y", which="both", left=False, labelleft=False)
        else:
            plt.tick_params(axis="both", which="both", left=False, labelleft=False,
                            bottom=False, labelbottom=False)

        ax = plt.subplot(gs[3 * r + 2])
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        if r == 0:
            ax.legend(["Model", "Participants"], fontsize=ft_legend,
                      loc="lower left", bbox_to_anchor=(1, 0.5))

    plt.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="w")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="overwrite ../../new-figs/ instead of writing to a temp dir")
    args = ap.parse_args()

    dest = os.path.join(REPO_ROOT, "new-figs")
    out_dir = dest if args.write else tempfile.mkdtemp(prefix="new-figs-")
    os.makedirs(out_dir, exist_ok=True)

    for fname, rows in FIGURES.items():
        path = os.path.join(out_dir, fname)
        make_figure(rows, path)
        print(f"wrote {path}")

    if not args.write:
        print(f"\n(regenerated in {out_dir}; not overwriting committed new-figs/.")
        print(" pass --write to overwrite, or compare manually — bitmaps are not")
        print(" byte-identical across matplotlib/font versions.)")


if __name__ == "__main__":
    main()
