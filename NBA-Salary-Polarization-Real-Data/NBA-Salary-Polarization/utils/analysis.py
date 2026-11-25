"""Shared data cleaning plus the inequality and polarization measures.

Used by both ``models/train.py`` and ``app.py`` so the web app and the training
script always see identical features and identical metric definitions.

How polarization is measured
----------------------------
Inequality and polarization are different things:

* Inequality (Gini, top-10% share) asks how unevenly the salary pool is split.
* Polarization asks whether players are pulling apart into separate groups at
  the ends of the distribution, with fewer players in the middle.

So each season gets both kinds of measure:

* ``gini``: 0 = everyone paid the same, 1 = one player earns everything.
* ``top_10_pct_share``: share of total payroll earned by the best-paid 10%.
* ``wolfson``: the Wolfson (1994) bipolarization index. It rises when the
  bottom and top halves move away from the median.
* ``middle_share``: share of players paid 50%-150% of that season's median
  salary (the "disappearing middle" view of polarization).

Gini, top-10% share and Wolfson do not change when every salary is multiplied
by the same number, so cap growth does not affect them. Comparing them across
seasons is fair even though salaries are nominal dollars.
"""
import numpy as np
import pandas as pd

# Per-game performance and context features used by the salary models.
FEATURES = [
    'age', 'games', 'games_started', 'minutes', 'points', 'rebounds',
    'assists', 'steals', 'blocks', 'fg_pct', 'three_pct', 'three_pa',
]

# Human-readable labels for charts and form inputs.
FEATURE_LABELS = {
    'age': 'Age', 'games': 'Games played', 'games_started': 'Games started',
    'minutes': 'Minutes per game', 'points': 'Points per game',
    'rebounds': 'Rebounds per game', 'assists': 'Assists per game',
    'steals': 'Steals per game', 'blocks': 'Blocks per game',
    'fg_pct': 'Field goal % (0-1)', 'three_pct': 'Three-point % (0-1)',
    'three_pa': 'Three-point attempts per game',
}

REQUIRED = ['player', 'season', 'salary', 'salary_cap', 'cap_share'] + FEATURES


def clean_data(df):
    """Validate columns, coerce numeric types, and drop unusable rows.

    Raises ValueError if the processed dataset is missing a required column
    (run ``python scripts/prepare_data.py`` to rebuild it).
    """
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing)
                         + '. Run python scripts/prepare_data.py to rebuild the dataset.')
    extra = [c for c in ['position', 'team', 'season_label'] if c in df.columns]
    out = df[REQUIRED + extra].copy()
    out['player'] = out['player'].astype(str).str.strip()
    out['season'] = out['season'].astype(str).str.strip()
    for col in ['salary', 'salary_cap', 'cap_share'] + FEATURES:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.dropna(subset=['player', 'season', 'salary', 'cap_share'])
    out = out[(out.salary > 0) & (out.player != '') & (out.season != '')]
    # Distinct players can share a name and season (Chris Johnson 2013, Tony
    # Mitchell 2014), so never deduplicate on name/season; only exact copies.
    return out.drop_duplicates()


def gini(values):
    """Gini coefficient of non-negative values (0 = perfect equality)."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a) & (a >= 0)]
    if not len(a) or a.sum() == 0:
        return 0.0
    a.sort()
    n = len(a)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * a.sum()))


def lorenz_at(values, p):
    """Share of the total held by the poorest fraction ``p`` (linear interpolation)."""
    a = np.sort(np.asarray(values, dtype=float))
    x = np.arange(len(a) + 1) / len(a)
    y = np.concatenate([[0.0], np.cumsum(a)]) / a.sum()
    return float(np.interp(p, x, y))


def wolfson(values):
    """Wolfson (1994) polarization index: P = 2 * (2T - G) * mean / median,
    where T = 0.5 - L(0.5) and L(0.5) is the bottom half's share of the total.
    0 = no polarization; larger = the two halves are farther from the middle."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a) & (a >= 0)]
    if not len(a) or a.sum() == 0 or np.median(a) == 0:
        return 0.0
    t = 0.5 - lorenz_at(a, 0.5)
    return float(2 * (2 * t - gini(a)) * a.mean() / np.median(a))


def middle_share(values, low=0.5, high=1.5):
    """Fraction of players paid between ``low`` and ``high`` times the median."""
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return 0.0
    m = np.median(a)
    return float(((a >= low * m) & (a <= high * m)).mean())


def inequality(df):
    """One row per season with inequality and polarization measures."""
    rows = []
    for season, group in df.groupby('season'):
        salaries = group.salary.dropna().sort_values(ascending=False)
        n = len(salaries)
        top_n = max(1, int(np.ceil(n * .1)))
        rows.append({
            'season': season,
            'players': n,
            'gini': gini(salaries),
            'top_10_pct_share': float(salaries.iloc[:top_n].sum() / salaries.sum()),
            'wolfson': wolfson(salaries),
            'middle_share': middle_share(salaries),
            'median_salary': float(salaries.median()),
            'median_cap_share': float(group.cap_share.median()),
        })
    return pd.DataFrame(rows).sort_values('season')
