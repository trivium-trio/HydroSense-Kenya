"""
visualization.py
Reusable scientific visualization functions for HydroSense-Kenya.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_rainfall_moisture(weather, soil, params, save_path=None):
    """Plot rainfall bars, soil moisture by zone, and ET on a 3-panel figure."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('HydroSense-Kenya — Weather and Soil Moisture Overview',
                 fontsize=15, fontweight='bold', y=0.98)

    colors = {'Zone_A': '#22c55e', 'Zone_B': '#f59e0b', 'Zone_C': '#8b5cf6'}
    labels = {'Zone_A': 'Zone A (Tomato)', 'Zone_B': 'Zone B (Kale)', 'Zone_C': 'Zone C (Maize)'}

    # Panel 1: Rainfall
    ax = axes[0]
    ax.bar(weather['date'], weather['rainfall_mm'], color='#3b82f6',
           edgecolor='#1e40af', width=0.8, alpha=0.85)
    ax.set_ylabel('Rainfall (mm)')
    ax.set_title('Daily Rainfall', loc='left')
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Soil moisture
    ax = axes[1]
    for zone_id in ['Zone_A', 'Zone_B', 'Zone_C']:
        zd = soil[soil['zone_id'] == zone_id]
        ax.plot(zd['timestamp'], zd['soil_moisture_pct'], 'o-', markersize=3,
                linewidth=1.5, color=colors[zone_id], label=labels[zone_id])
    for _, r in params.iterrows():
        ax.axhline(r['min_moisture_pct'], linestyle='--', color=colors[r['zone_id']],
                   alpha=0.5, linewidth=0.9)
    ax.set_ylabel('Soil Moisture (%)')
    ax.set_title('Soil Moisture by Zone (dashed = min threshold)', loc='left')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: ET
    if 'ET' in weather.columns:
        ax = axes[2]
        ax.fill_between(weather['date'], weather['ET'], alpha=0.3, color='#f97316')
        ax.plot(weather['date'], weather['ET'], color='#ea580c', linewidth=1.5,
                marker='s', markersize=3)
        ax.set_ylabel('ET (mm-equiv.)')
        ax.set_title('Estimated Daily Evapotranspiration', loc='left')
        ax.grid(axis='y', alpha=0.3)

    axes[-1].set_xlabel('Date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_simulation_comparison(day_range, euler_results, rk4_results, params,
                                zone_names=None, save_path=None):
    """Plot Euler vs RK4 simulation results for all zones."""
    colors = {'Zone_A': '#22c55e', 'Zone_B': '#f59e0b', 'Zone_C': '#8b5cf6'}
    if zone_names is None:
        zone_names = {'Zone_A': 'Zone A (Tomato)', 'Zone_B': 'Zone B (Kale)',
                      'Zone_C': 'Zone C (Maize)'}

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Soil Moisture Simulation: Euler vs. Runge-Kutta',
                 fontsize=14, fontweight='bold')

    for idx, zone in enumerate(['Zone_A', 'Zone_B', 'Zone_C']):
        ax = axes[idx]
        zp = params[params['zone_id'] == zone].iloc[0]
        ax.plot(day_range, euler_results[zone], '-', linewidth=2,
                color=colors[zone], label='Euler')
        ax.plot(day_range, rk4_results[zone], '--', linewidth=2,
                color='#1e293b', label='RK4')
        ax.axhline(zp['min_moisture_pct'], linestyle=':', color='red', alpha=0.6)
        ax.axhline(zp['target_moisture_pct'], linestyle=':', color='blue', alpha=0.4)
        ax.set_ylabel('Moisture (%)')
        ax.set_title(zone_names[zone], loc='left', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Day')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_monte_carlo_fan(day_range, mc_moisture, observed, min_threshold,
                          title='Monte Carlo Uncertainty', save_path=None):
    """Plot Monte Carlo fan chart with percentile bands."""
    mc_mean = mc_moisture.mean(axis=0)
    p5 = np.percentile(mc_moisture, 5, axis=0)
    p25 = np.percentile(mc_moisture, 25, axis=0)
    p75 = np.percentile(mc_moisture, 75, axis=0)
    p95 = np.percentile(mc_moisture, 95, axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(day_range, p5, p95, alpha=0.15, color='#3b82f6', label='5th–95th %ile')
    ax.fill_between(day_range, p25, p75, alpha=0.3, color='#3b82f6', label='25th–75th %ile')
    ax.plot(day_range, mc_mean, '-', color='#1d4ed8', linewidth=2, label='Mean')
    ax.plot(day_range, observed, '--', color='black', linewidth=1.5, label='Observed rainfall')
    ax.axhline(min_threshold, linestyle=':', color='red', alpha=0.7, label='Min threshold')
    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Soil Moisture (%)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df, columns, labels=None, title='Correlation Matrix',
                              save_path=None):
    """Plot a correlation heatmap with annotated values."""
    corr = df[columns].corr()
    if labels is None:
        labels = columns

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if abs(corr.iloc[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    plt.colorbar(im, label='Pearson Correlation', shrink=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
