"""Build the processed dataset (data/nba_players.csv) from the raw source file.

Run from the project folder:
    python scripts/prepare_data.py

Steps
-----
1. Read the raw CSV (it starts with a UTF-8 byte-order mark, hence ``utf-8-sig``).
2. Keep the columns the project uses and rename them to snake_case.
3. Fill undefined shooting percentages. The raw file leaves FG% / 3P% blank when a
   player attempted no shots of that type (e.g. a center who never shoots threes).
   We set those to 0.0 and keep the attempt volume (``three_pa``) as a feature, so
   the model can tell "never shoots threes" apart from "shoots threes badly".
   Previously these blanks were median-imputed during training, which credited
   585 non-shooters with a ~33% three-point percentage.
4. Join the official NBA salary cap for each season and compute ``cap_share`` =
   salary / cap. Salaries are nominal dollars and the cap rose from $57.7M
   (2009-10) to $140.6M (2024-25), so raw dollars are not comparable across
   seasons. Cap share is the standard way to compare NBA pay across eras.

No rows are removed. Very small salaries (10-day / partial contracts) and players
with few games are real league payroll and are kept so the inequality numbers
describe the whole league; see DATASET_REVIEW.md for their counts.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'NBA Player Stats and Salaries_2010-2025.csv'
CAP = ROOT / 'data' / 'salary_cap.csv'
OUT = ROOT / 'data' / 'nba_players.csv'

# Raw column name -> project column name. "Year" is the calendar year the season
# ends in (Year 2010 = the 2009-10 season).
COLUMNS = {
    'Player': 'player', 'Year': 'season', 'Salary': 'salary', 'Age': 'age',
    'G': 'games', 'GS': 'games_started', 'MP': 'minutes', 'PTS': 'points',
    'TRB': 'rebounds', 'AST': 'assists', 'STL': 'steals', 'BLK': 'blocks',
    'FG%': 'fg_pct', '3P%': 'three_pct', '3PA': 'three_pa',
    'Pos': 'position', 'Team': 'team',
}


def prepare(raw_path=RAW, cap_path=CAP):
    """Return the processed player-season table built from the raw CSV."""
    raw = pd.read_csv(raw_path, encoding='utf-8-sig')
    df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].copy()

    # Undefined percentages (zero attempts) -> 0.0; attempts stay as a feature.
    df['fg_pct'] = df['fg_pct'].fillna(0.0)
    df['three_pct'] = df['three_pct'].fillna(0.0)

    caps = pd.read_csv(cap_path)[['season', 'season_label', 'salary_cap']]
    df = df.merge(caps, on='season', how='left', validate='many_to_one')
    if df['salary_cap'].isna().any():
        missing = sorted(df.loc[df['salary_cap'].isna(), 'season'].unique())
        raise ValueError(f'No salary cap listed for seasons: {missing}')
    df['cap_share'] = df['salary'] / df['salary_cap']
    return df


if __name__ == '__main__':
    data = prepare()
    data.to_csv(OUT, index=False)
    print(f'Wrote {len(data):,} rows x {data.shape[1]} columns to {OUT.relative_to(ROOT)}')
