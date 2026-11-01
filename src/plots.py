import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots (dataviz reference palette, light mode), one per model family
FAMILY_COLORS = {'Empirical': '#eb6834', 'Mechanistic': '#2a78d6', 'ML': '#1baf7a', 'Hybrid': '#6250d6'}
TEXT, TEXT_2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

EMPIRICAL = ('Gilbert', 'Ros', 'Baxendell', 'Achong')

def family(model):
    if model.startswith('Perkins'):
        return 'Mechanistic'
    if model.startswith('Hybrid'):
        return 'Hybrid'
    if model in EMPIRICAL:
        return 'Empirical'
    return 'ML'

def uses_tests(model):
    """Models that use the last periodic well test (shown hatched or dashed)."""
    return 'well tests' in model or 'test calibration' in model

def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)

def plot_parity(pred, split, output_path):
    """Small multiples of observed vs predicted liquid rate, one panel per model (log-log)."""
    models = list(pred.Model.unique())
    ncols = 4
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.3 * nrows), sharex=True, sharey=True)
    lo, hi = pred[['Q_obs', 'Q_pred']].clip(lower=1).min().min() * 0.8, pred[['Q_obs', 'Q_pred']].max().max() * 1.2

    for ax, model in zip(axes.flat, models):
        g = pred[pred.Model == model]
        aapre = np.mean(np.abs(g.Q_pred - g.Q_obs) / g.Q_obs) * 100
        ax.scatter(g.Q_obs, g.Q_pred.clip(lower=lo), s=8, alpha=0.35, color=FAMILY_COLORS[family(model)], edgecolors='none')
        ax.plot([lo, hi], [lo, hi], color=TEXT_2, lw=1, ls='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{model}\nAAPRE {aapre:.1f}%", fontsize=10, color=TEXT)
        _style(ax)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    fig.supxlabel('Observed liquid rate (STB/d)', color=TEXT_2, fontsize=10)
    fig.supylabel('Predicted liquid rate (STB/d)', color=TEXT_2, fontsize=10)
    fig.suptitle(f"Volve parity, {split.lower()} split", color=TEXT, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_error_by_subset(metrics, metric, output_path, subsets=('All', 'Choke-controlled', 'Settled')):
    """Horizontal bars of an error metric per model, small multiples by split (rows) and data subset (columns)."""
    splits = list(metrics.Split.unique())
    n_models = metrics.Model.nunique()
    fig, axes = plt.subplots(len(splits), len(subsets), figsize=(14, 0.28 * n_models * len(splits) + 1.2), sharex=True)
    shown = metrics[metrics.Subset.isin(subsets)]
    xmax = min(shown[metric].max(), 150) * 1.18
    label = {'AAPRE_%': 'Mean absolute % error (AAPRE)', 'MedAPE_%': 'Median absolute % error'}[metric]

    for i, split in enumerate(splits):
        for j, subset in enumerate(subsets):
            ax = axes[i, j]
            m = metrics[(metrics.Split == split) & (metrics.Subset == subset)].iloc[::-1]
            colors = [FAMILY_COLORS[family(x)] for x in m.Model]
            bars = ax.barh(m.Model, m[metric].clip(upper=xmax), color=colors, height=0.7, edgecolor='white', linewidth=1.5)
            for bar, model in zip(bars, m.Model):
                bar.set_hatch('///' if uses_tests(model) else '')
            for y, v in enumerate(m[metric]):
                ax.text(min(v, xmax) + xmax * 0.01, y, f"{v:.1f}%", va='center', fontsize=7, color=TEXT)
            ax.set_xlim(0, xmax)
            n_rows = int(m.n.iloc[0]) if len(m) else 0
            ax.set_title(f"{split} · {subset} (n = {n_rows})", fontsize=10, color=TEXT, loc='left')
            if j:
                ax.set_yticklabels([])
            ax.tick_params(axis='y', labelsize=8)
            _style(ax)
            ax.grid(axis='y', visible=False)
    for ax in axes[-1]:
        ax.set_xlabel(f"{label}, bars capped at {xmax / 1.18:.0f}%", color=TEXT_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=TEXT_2, hatch='///'))
    fig.legend(handles, list(FAMILY_COLORS) + ['Uses last well test'], loc='upper center', ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_feature_importance(importance, output_path):
    """Permutation importance: how much the error grows when each input is shuffled."""
    imp = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(imp) + 1.2))
    ax.barh(imp.Feature, imp.Importance.clip(lower=0), xerr=imp.Std, color=FAMILY_COLORS['ML'], height=0.7,
            error_kw={'ecolor': TEXT_2, 'lw': 1})
    for y, v in enumerate(imp.Importance):
        ax.text(max(v, 0) + imp.Importance.max() * 0.02, y, f"{v:.1f}", va='center', fontsize=8, color=TEXT)
    ax.set_xlabel('Increase in mean absolute % error when shuffled (points)', color=TEXT_2, fontsize=9)
    ax.set_title('GB + well tests: what the model relies on (chronological test set)', fontsize=10, color=TEXT, loc='left')
    _style(ax)
    ax.grid(axis='y', visible=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_prediction_intervals(intervals, coverage, output_path):
    """Observed rate vs the 80% conformal prediction interval over the test period, one panel per well."""
    wells = list(intervals.Well.unique())
    fig, axes = plt.subplots(len(wells), 1, figsize=(10, 1.9 * len(wells) + 0.8), sharex=False)
    for ax, well in zip(np.atleast_1d(axes), wells):
        g = intervals[intervals.Well == well].sort_values('Date')
        # Break lines and bands across data gaps instead of drawing across them
        gaps = g.Date[g.Date.diff().dt.days > 3] - pd.Timedelta(days=1)
        g = pd.concat([g, pd.DataFrame({'Date': gaps.values})]).sort_values('Date')
        ax.fill_between(g.Date, g.Q_p10, g.Q_p90, color=FAMILY_COLORS['ML'], alpha=0.25, lw=0, label='80% interval')
        ax.plot(g.Date, g.Q_pred, color=FAMILY_COLORS['ML'], lw=1.2, label='Predicted')
        ax.scatter(g.Date, g.Q_obs, s=6, color=TEXT, label='Observed', zorder=3)
        ax.set_title(well, fontsize=9, color=TEXT, loc='left')
        ax.set_ylabel('STB/d', color=TEXT_2, fontsize=8)
        _style(ax)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, ncol=3, loc='upper right', labelcolor=TEXT)
    fig.suptitle(f"GB + well tests, chronological test period: 80% conformal interval covers "
                 f"{coverage['Conformal_coverage_%']:.0f}% of days", fontsize=11, color=TEXT, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_error_cdf(pred, split, models, output_path):
    """Share of predictions within a given absolute percentage error, one line per model."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model in models:
        g = pred[pred.Model == model]
        err = np.sort(np.abs(g.Q_pred - g.Q_obs) / g.Q_obs * 100)
        share = np.arange(1, len(err) + 1) / len(err) * 100
        color = FAMILY_COLORS[family(model)]
        ls = '--' if uses_tests(model) else '-'
        within_25 = np.mean(err <= 25) * 100
        ax.plot(err, share, color=color, lw=2, ls=ls, label=f"{model} ({within_25:.0f}% within 25%)")
    ax.legend(loc='lower right', frameon=False, fontsize=8, labelcolor=TEXT)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Absolute percentage error (%)', color=TEXT_2, fontsize=9)
    ax.set_ylabel('Predictions within error (%)', color=TEXT_2, fontsize=9)
    ax.set_title(f"Error distribution, Volve {split.lower()} split", color=TEXT, fontsize=11, loc='left')
    _style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_all():
    pred = pd.read_csv("outputs/volve_predictions.csv")
    metrics = pd.read_csv("outputs/volve_metrics.csv")
    for split, tag in [('Leave-one-well-out', 'lowo'), ('Chronological', 'chrono')]:
        plot_parity(pred[pred.Split == split], split, f"outputs/parity_{tag}.png")
        plot_error_cdf(pred[pred.Split == split], split,
                       ['Gilbert', 'Perkins', 'Gradient Boosting', 'Hybrid (Perkins + GB)',
                        'Perkins + test calibration', 'GB + well tests', 'Hybrid + well tests'],
                       f"outputs/error_cdf_{tag}.png")
    plot_error_by_subset(metrics, 'AAPRE_%', "outputs/aapre_by_subset.png")
    plot_error_by_subset(metrics, 'MedAPE_%', "outputs/medape_by_subset.png")
    plot_feature_importance(pd.read_csv("outputs/feature_importance.csv"), "outputs/feature_importance.png")
    plot_prediction_intervals(pd.read_csv("outputs/prediction_intervals.csv", parse_dates=['Date']),
                              pd.read_csv("outputs/prediction_interval_coverage.csv").iloc[0],
                              "outputs/prediction_intervals.png")
    print("Saved plots to 'outputs/'.")

if __name__ == "__main__":
    plot_all()
