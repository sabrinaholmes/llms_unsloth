"""Shared plotting helpers for llms_unsloth task scripts.

Import from any task subdirectory via the same sys.path convention already
used for get_models.py:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import plotting_utils

Deliberately independent of get_models.py (no unsloth/transformers/torch
imports), so plotting-only scripts stay cheap to import.
"""
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt

# Serif everywhere, matching LaTeX's default text/math typefaces. Set at
# import time (rather than only inside set_dynamic_fontsize, below) so every
# script gets it just by doing `import plotting_utils`, regardless of whether
# it happens to call set_dynamic_fontsize -- composite figures that always
# pass their own `ax=` into rl_waltmann's plot_* helpers never hit that
# function's "standalone" branch, which is the only place it used to be set.
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})


# --- Fontsize -----------------------------------------------------------

_FONT_ROLE_MULTIPLIERS = {
    'font.size':        1.0,
    'axes.titlesize':   1.25,
    'axes.labelsize':   1.05,
    'xtick.labelsize':  1.0,
    'ytick.labelsize':  1.0,
    'legend.fontsize':  1.0,
}


def get_dynamic_fontsize(multiplier=1.0, fig_width=12, base_font=14, min_font=7, max_font=32):
    """
    Pure version of set_dynamic_fontsize's scale+clamp formula (no rcParams side
    effect) -- for one-off `fontsize=N` call sites that don't automatically
    inherit rcParams the way standard Axes artists do (e.g. fig.text/fig.suptitle,
    which fall back to the flat 'font.size' rather than 'axes.titlesize' etc.),
    so those can still track the same width-based scale instead of a hardcoded
    literal.
    """
    # Gentle scaling: sqrt rather than linear, so doubling width doesn't double font size
    scale = np.sqrt(fig_width / 12)
    return max(min_font, min(max_font, base_font * scale * multiplier))


def get_dynamic_labelpad(fig_width=12, base_pad=3, min_pad=2, max_pad=30):
    """
    Scale an axis label's `labelpad` with figure width, using the same gentle
    sqrt scaling get_dynamic_fontsize uses for font sizes. `base_pad` is the
    pad you want at the 12-inch-wide reference size; pass the *actual*
    figure's width (ax.figure.get_size_inches()[0], not a hardcoded default)
    so labels tuned for a large standalone figure don't look oversized when
    the same plotting function is reused on a small embedded panel.
    """
    scale = np.sqrt(fig_width / 16)
    return max(min_pad, min(max_pad, base_pad * scale))


def set_dynamic_fontsize(fig_width=12, base_font=11, min_font=7, max_font=32):
    """
    base_font: target label size at your *reference* figure width — pick this to
    match where the figure will actually be viewed (paper column, slide, poster),
    not an arbitrary constant.
    min_font/max_font: hard clamps so extreme fig_width values can't blow up sizing.
    """
    plt.rcParams.update({
        role: get_dynamic_fontsize(mult, fig_width, base_font, min_font, max_font)
        for role, mult in _FONT_ROLE_MULTIPLIERS.items()
    })
    plt.rcParams.update({
        'mathtext.fontset': 'cm',
        'font.family': 'serif',
    })


# --- Grid / spines ----------------------------------------------------------

def style_y_gridlines(ax):
    """Horizontal-only gridlines at the major y-ticks, drawn behind the data.

    Shared so every task's plots use the same light-grey/thin look rather than
    each script picking its own grid styling.
    """
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#d9d9d9', linewidth=0.8, alpha=0.9, zorder=0)
    ax.grid(axis='x', visible=False)


def remove_bar_frame(ax):
    """Hide top and right spines around a bar plot -- gridlines carry the scale
    instead of a box/frame."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# --- Colors ---------------------------------------------------------------

CENTAUR_ORANGE = ['#D55E00', '#E69F00']  # dark, light
CENTAUR_LILA = ['#5E3C99', '#B39DDB']    # dark, light
LLAMA_COLORS = ['#0072B2', '#56B4E9']    # dark, light
HUMAN_COLORS = ['#000000', '#999999']    # rl_waltmann's 'human' family
DOMAIN_SPECIFIC_COLORS = ['#CC79A7', '#999999']  # predictive_plots' 'domain-specific' (RW) family

CENTAUR_COLOR_OPTIONS = {'orange': CENTAUR_ORANGE, 'lila': CENTAUR_LILA}

DEFAULT_COLOR_MAP = {
    'centaur': CENTAUR_ORANGE,
    'llama': LLAMA_COLORS,
    'human': HUMAN_COLORS,
    'domain-specific': DOMAIN_SPECIFIC_COLORS,
}


def define_colors_for_families(family_mapping, centaur_color_mode='orange', color_map=None):
    """
    Define a color mapping for each model family.

    Parameters
    ----------
    family_mapping : dict
        A dictionary mapping family names to lists of (model_name, DataFrame[, size_label]) tuples.
    centaur_color_mode : str
        'orange' (default) or 'lila' — selects which shade pair centaur is drawn in.
    color_map : dict, optional
        Full override of DEFAULT_COLOR_MAP.

    Returns
    -------
    dict
        A dictionary mapping family names to a list of hex colors, falling back
        to ['#000000'] for any family not present in the map.
    """
    base = dict(color_map or DEFAULT_COLOR_MAP)
    base['centaur'] = CENTAUR_COLOR_OPTIONS.get(centaur_color_mode, CENTAUR_ORANGE)
    return {family: base.get(family, ['#000000']) for family in family_mapping.keys()}


# --- Trend aggregation ------------------------------------------------------

TrendStats = namedtuple('TrendStats', ['mean', 'error'])


def aggregate_trend(df, trial_col, value_col, ci_multiplier=1.0, compute_error=True):
    """
    Group df by trial_col, compute the mean of value_col per trial, and
    (unless compute_error=False) an error band = ci_multiplier * SEM.

    ci_multiplier=1.0  -> plain SEM
    ci_multiplier=1.96 -> ~95% CI
    compute_error=False -> skips .std()/.count() entirely, returns error=None
                           (cheaper than ci_multiplier=0.0 for mean-only callers)

    Returns TrendStats(mean: pd.Series indexed by trial_col, error: pd.Series | None)
    """
    grouped = df.groupby(trial_col)[value_col]
    mean = grouped.mean()
    if not compute_error:
        return TrendStats(mean, None)
    error = ci_multiplier * grouped.std() / np.sqrt(grouped.count())
    return TrendStats(mean, error)
