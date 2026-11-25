# Dataset review

## Source

- Raw file: `data/raw/NBA Player Stats and Salaries_2010-2025.csv`
- **TODO: original source, URL and license.** (Add where the file was downloaded from, e.g. the Kaggle page, author, and license, before submission.)
- Salary cap figures: `data/salary_cap.csv`, from NBA press releases (pr.nba.com) and salary-cap trackers; each row lists its source. The 2012-13 figure is marked to verify.

## Size and metadata

- 7,298 player-season rows, 31 raw columns, 16 seasons, 1,658 distinct player names.
- `Year` is the year the season ends: `2010` = the 2009-10 season, `2025` = 2024-25.
- 402-540 players per season.
- Statistics are per-game averages (except `G`/`GS`, which are game counts).
- Salaries are nominal US dollars. The cap rose from $57.7M (2009-10) to $140.6M (2024-25).

## Processing (`scripts/prepare_data.py`)

1. Rename 17 used columns to snake_case (the other raw columns are unused).
2. **Undefined shooting percentages**: FG% is blank in 20 rows and 3P% in 585 rows, always because the player attempted zero such shots (all 20 blank-FG% rows have FGA = 0). They are set to 0.0, and three-point attempts per game (`three_pa`) is kept as a feature so the model can tell non-shooters apart from poor shooters. Before this change, training median-imputed them, which gave non-shooters a ~33% three-point percentage.
3. Join the salary cap per season and compute `cap_share = salary / salary_cap`.

No rows are removed, so the processed file has 7,298 rows.

## Known data quirks (documented, not filtered)

| Quirk | Count | Handling / note |
|---|---|---|
| Salary under $100K (10-day / partial contracts, waived players) | 178 rows | Kept: they are real payroll and part of the inequality picture, but they push the lower tail down. |
| Salary under $500K | 651 rows | Same as above. |
| Fewer than 10 games played | 591 rows | Kept: per-game stats from tiny samples are noisy. |
| Same name, same season, different players | 2 (Chris Johnson 2013, Tony Mitchell 2014) | Never deduplicate on name/season. |
| Traded players | — | One row per player-season under a single team; the file does not say which stint. |
| Hybrid positions (e.g. `SF-SG`) | 129 rows | Position is not used as a model feature. |
| No stable player ID | — | Players are matched by name only. |

## Modeling notes

- Salaries depend on the contract-signing year and on prior seasons. Same-season stats therefore *describe* pay; they do not predict future contracts.
- Evaluation uses a chronological holdout: train on 2010-2024, test on 2025. A random split would leak a player's other seasons into training.
- Gini, top-10% share and Wolfson are unchanged when all salaries are scaled by the same factor, so cap growth does not bias them across seasons. Dollar-based comparisons (medians, model targets) use `cap_share`.
