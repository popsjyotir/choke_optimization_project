import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.plots import FAMILY_COLORS, TEXT, TEXT_2, GRID, family, _style

def plot_choke_response(summary, output_path):
    """Direction accuracy (with 95% CI) and rate-change error per model and mode in the choke-response test."""
    models = summary[summary.Mode != '-'].copy()
    models['Label'] = models.Model + ' · ' + models.Mode
    models = models.iloc[::-1]
    baseline = summary[summary.Model == 'Rate follows choke direction'].iloc[0]
    no_change = summary[summary.Model == 'No-change baseline'].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(models) + 1.8), sharey=True)
    colors = [FAMILY_COLORS[family(m)] for m in models.Model]
    hatches = ['' if mode == 'Nodal' else '///' for mode in models.Mode]
    y = np.arange(len(models))

    ax = axes[0]
    bars = ax.barh(y, models['Direction_correct_%'], color=colors, height=0.7, edgecolor='white', linewidth=2)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.errorbar(models['Direction_correct_%'], y,
                xerr=[models['Direction_correct_%'] - models.Direction_CI_low, models.Direction_CI_high - models['Direction_correct_%']],
                fmt='none', ecolor=TEXT_2, lw=1, capsize=3)
    ax.axvline(baseline['Direction_correct_%'], color=TEXT_2, ls='--', lw=1)
    ax.text(baseline['Direction_correct_%'] + 1, len(models) - 0.4, 'rate follows choke', fontsize=8, color=TEXT_2)
    ax.axvline(50, color=GRID, lw=1, zorder=0)
    ax.set_xlim(0, 100)
    ax.set_yticks(y, [f"{lbl} (n = {n})" for lbl, n in zip(models.Label, models.n)])
    ax.set_xlabel('Direction of rate change correct (%)', color=TEXT_2, fontsize=9)
    ax.set_title('Does the model get the direction right?', fontsize=10, color=TEXT, loc='left')

    ax = axes[1]
    bars = ax.barh(y, models['dQ_error_%_of_rate'], color=colors, height=0.7, edgecolor='white', linewidth=2)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for yi, v in zip(y, models['dQ_error_%_of_rate']):
        ax.text(v + 0.2, yi, f"{v:.1f}%", va='center', fontsize=8, color=TEXT)
    ax.axvline(no_change['dQ_error_%_of_rate'], color=TEXT_2, ls='--', lw=1)
    ax.text(no_change['dQ_error_%_of_rate'] + 0.2, len(models) - 0.4, 'no-change baseline', fontsize=8, color=TEXT_2)
    ax.set_xlabel('Error in predicted rate change (% of rate)', color=TEXT_2, fontsize=9)
    ax.set_title('How far off is the size of the change?', fontsize=10, color=TEXT, loc='left')
    for ax in axes:
        _style(ax)
        ax.grid(axis='y', visible=False)
    fig.text(0.01, -0.02, 'Hatched = fixed WHP, solid = nodal (choke + tubing + reservoir). Error bars: 95% Wilson interval.',
             fontsize=8, color=TEXT_2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_rate_vs_choke(nodal, fixed, well, output_path):
    """Rate vs choke opening with WHP held fixed vs solved by the nodal system, plus nodal WHP."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    ax = axes[0]
    ax.plot(fixed.Choke_pct, fixed.Q_liquid_STB, color=FAMILY_COLORS['Mechanistic'], lw=2, ls='--', label='Fixed WHP')
    ax.plot(nodal.Choke_pct, nodal.Q_liquid_STB, color=FAMILY_COLORS['Hybrid'], lw=2, label='Nodal (WHP solved)')
    ax.set_ylabel('Liquid rate (STB/d)', color=TEXT_2, fontsize=9)
    ax.set_title(f'{well}: rate vs choke opening', fontsize=10, color=TEXT, loc='left')
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
    ax = axes[1]
    ax.plot(nodal.Choke_pct, nodal.WHP_psig, color=FAMILY_COLORS['Hybrid'], lw=2)
    ax.set_ylabel('Wellhead pressure (psig)', color=TEXT_2, fontsize=9)
    ax.set_title('Nodal WHP falls as the choke opens', fontsize=10, color=TEXT, loc='left')
    for ax in axes:
        ax.set_xlabel('Choke opening (%)', color=TEXT_2, fontsize=9)
        _style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def plot_extensions():
    plot_choke_response(pd.read_csv("outputs/extensions/choke_response_summary.csv"), "outputs/extensions/choke_response.png")
    plot_rate_vs_choke(pd.read_csv("outputs/extensions/rate_vs_choke_F-14H_nodal.csv"),
                       pd.read_csv("outputs/extensions/rate_vs_choke_F-14H_fixed.csv"), 'F-14 H',
                       "outputs/extensions/rate_vs_choke_F-14H.png")
    print("Saved extension plots to 'outputs/extensions/'.")

if __name__ == "__main__":
    plot_extensions()
