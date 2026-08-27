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
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Serif everywhere, matching LaTeX's default text/math typefaces. Set at
# import time (rather than only inside set_dynamic_fontsize, below) so every
# script gets it just by doing `import plotting_utils`, regardless of whether
# it happens to call set_dynamic_fontsize -- composite figures that always
# pass their own `ax=` into rl_waltmann's plot_* helpers never hit that
# function's "standalone" branch, which is the only place it used to be set.
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'hatch.linewidth': 0.5
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


# Reference width (inches) at which scale == 1, i.e. font.size == base_font.
_REF_WIDTH = 12
# Added to both sides of the fig_width/_REF_WIDTH ratio before the sqrt. At
# fig_width == _REF_WIDTH the offset cancels out (scale is still exactly 1),
# but it flattens the curve for narrow figures (e.g. journal-column panels a
# couple inches wide) so they don't collapse straight to min_font regardless
# of base_font -- without the offset, sqrt(1.7/12) ~= 0.38 already undercuts
# every role's min_font=7 floor at typical base_font values, making base_font
# effectively a no-op for those figures.
_WIDTH_SOFTENING = 4


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
    scale = np.sqrt((fig_width + _WIDTH_SOFTENING) / (_REF_WIDTH + _WIDTH_SOFTENING))
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
    scale = np.sqrt((fig_width + _WIDTH_SOFTENING) / (_REF_WIDTH + _WIDTH_SOFTENING))
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
    ax.grid(axis='y', color='#d9d9d9', linewidth=0.8, alpha=0.5, zorder=0)
    ax.grid(axis='x', visible=False)

def style_ticks(ax):
    "shared yticks and xticks styling"
    ax.tick_params(axis='both',length=3, color='#888888', labelcolor='#666666')

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


def family_and_size_label(name):
    """
    Classify a model/participant name into (family_label, size_label), the
    shared basis for the 'size over family' two-tier bar labels (e.g. '70B'
    over 'Centaur', '70B' over 'Llama', lone 'Human') used by both
    rl_waltmann/transition_analysis.py's plot_wsls and
    horizon/generate_gen_plots.py's plot_fifth_trial_accuracy_bar.
    size_label is None for families (like human) that get a single label.
    """
    nl = name.lower()
    if 'centaur' in nl:
        return 'Centaur', '70B'
    if 'llama' in nl:
        return 'Llama', '70B'
    if 'human' in nl:
        return 'Human', None
    return name, None


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


# --- Saving -------------------------------------------------------------

def save_panel(fig, path, figsize, dpi=300):
    """
    Render `fig` onto a canvas of exactly `figsize` inches at `dpi`, instead of
    letting savefig's bbox_inches='tight' decide the final size.

    Panels destined for a fixed-width LaTeX slot (e.g. \\includegraphics[width=
    \\linewidth] inside a minipage) usually have their font sizes pre-scaled by
    set_dynamic_fontsize(fig_width=figsize[0], ...) to look right *at that
    physical width*. Plain bbox_inches='tight' crops each figure to its own
    content bbox by a different, unpredictable amount (depends on how much
    whitespace/annotation-bleed that particular figure happens to have) --
    LaTeX then re-stretches whatever comes out back up to fill \\linewidth, so
    the pre-scaled fonts end up magnified by a different factor per figure and
    no longer match across a row of panels sized from the same figsize.

    This still auto-trims interior whitespace (via an initial tight-bbox pass)
    but then centers that trimmed content on a fixed-size transparent canvas
    equal to `figsize`, so every panel saved this way is pixel-for-pixel the
    same physical size and \\linewidth never has to rescale it.

    Not a fit for figures that intentionally draw outside their own axes bbox
    to gain space (e.g. a legend anchored below the figure at a negative
    figure-fraction y) -- those need genuine extra canvas, not a fixed-size
    crop/pad. Save those with plain bbox_inches='tight' instead.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02,
                dpi=dpi, transparent=True)
    buf.seek(0)
    content = Image.open(buf).convert('RGBA')

    target_w, target_h = round(figsize[0] * dpi), round(figsize[1] * dpi)
    if content.width > target_w or content.height > target_h:
        # Content (incl. any clip_on=False annotation bleed) overflows the
        # target box -- shrink it to fit rather than let it hang off the edge.
        scale = min(target_w / content.width, target_h / content.height)
        content = content.resize(
            (max(1, round(content.width * scale)), max(1, round(content.height * scale))),
            Image.LANCZOS)

    canvas = Image.new('RGBA', (target_w, target_h), (0, 0, 0, 0))
    offset = ((target_w - content.width) // 2, (target_h - content.height) // 2)
    # canvas.paste(content, offset, content) would be wrong here: PIL's paste()
    # with a mask does a naive per-channel lerp (dst*(1-a) + src*a) rather than
    # proper "over" alpha compositing, so every semi-transparent pixel (any
    # anti-aliased text/line edge) gets its color *and* alpha both cut roughly
    # in half against this fully-transparent canvas -- e.g. (200,50,50,128)
    # becomes (100,25,25,64) instead of staying (200,50,50,128). That fades
    # every edge, which reads as lighter/washed-out ("more white") once
    # composited onto any real background. alpha_composite does the correct
    # Porter-Duff "over" operation instead.
    canvas.alpha_composite(content, offset)
    canvas.save(path, dpi=(dpi, dpi))
