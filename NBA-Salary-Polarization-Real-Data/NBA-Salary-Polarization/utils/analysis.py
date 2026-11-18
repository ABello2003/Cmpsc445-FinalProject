import numpy as np
import pandas as pd

FEATURES = ['age', 'games', 'minutes', 'points', 'rebounds', 'assists', 'steals', 'blocks', 'fg_pct', 'three_pct']
REQUIRED = ['player', 'season', 'salary'] + FEATURES

def clean_data(df):
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    out = df[REQUIRED + [c for c in ['position', 'team'] if c in df.columns]].copy()
    out['player'] = out['player'].astype(str).str.strip()
    out['season'] = out['season'].astype(str).str.strip()
    for col in ['salary'] + FEATURES:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out = out.dropna(subset=['player', 'season', 'salary'])
    out = out[(out.salary >= 0) & (out.player != '') & (out.season != '')]
    # Distinct players can share a name and season (e.g. Chris Johnson).
    # Preserve all source records; only exact duplicate rows may be removed.
    out = out.drop_duplicates()
    return out

def gini(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a) & (a >= 0)]
    if not len(a) or a.sum() == 0:
        return 0.0
    a.sort()
    n = len(a)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * a.sum()))

def inequality(df):
    rows = []
    for season, group in df.groupby('season'):
        salaries = group.salary.dropna().sort_values(ascending=False)
        n = len(salaries)
        rows.append({'season': season, 'players': n, 'gini': gini(salaries),
                     'median_salary': float(salaries.median()),
                     'top_10_pct_share': float(salaries.iloc[:max(1, int(np.ceil(n * .1)))].sum() / salaries.sum()) if salaries.sum() else 0.0})
    return pd.DataFrame(rows).sort_values('season')
