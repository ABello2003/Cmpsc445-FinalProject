from pathlib import Path
import numpy as np
import pandas as pd
rng = np.random.default_rng(42)
rows = []
for season in range(2018, 2026):
    for i in range(180):
        age = int(rng.integers(19, 38))
        games = int(rng.integers(15, 83))
        minutes = float(rng.uniform(8, 38))
        points = float(max(0, rng.normal(minutes * .52, 4)))
        rebounds = float(rng.uniform(1, 12))
        assists = float(rng.uniform(.5, 9))
        steals = float(rng.uniform(.1, 2))
        blocks = float(rng.uniform(0, 2))
        fg_pct = float(rng.uniform(35, 62))
        three_pct = float(rng.uniform(20, 45))
        salary = float(np.clip(500000 + points * 550000 + assists * 250000 + rng.normal(0, 5e6), 500000, 55e6))
        rows.append(dict(player=f'Demo Player {i:03}', season=f'{season}-{str(season+1)[-2:]}', salary=salary, age=age, games=games, minutes=minutes, points=points, rebounds=rebounds, assists=assists, steals=steals, blocks=blocks, fg_pct=fg_pct, three_pct=three_pct))
root = Path(__file__).resolve().parents[1] / 'data'
root.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(root / 'nba_players.csv', index=False)
(root / 'DEMO_ONLY.txt').write_text('Synthetic data generated for UI testing; not real NBA records.\n')
print('Created synthetic demonstration dataset.')
