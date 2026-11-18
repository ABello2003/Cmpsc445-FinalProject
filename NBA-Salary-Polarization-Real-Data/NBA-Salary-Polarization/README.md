# NBA Salary Polarization

A Streamlit machine-learning capstone exploring economic inequality in NBA player salaries and the relationship between salaries and on-court performance.

## Research question
How unequal are NBA salaries over time, and how much of the observed variation is associated with player performance?

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Real NBA dataset included
The uploaded `NBA Player Stats and Salaries_2010-2025.csv` is preserved in `data/raw/`. The converted `data/nba_players.csv` contains 7,298 rows from 2010 through 2025. Run `python models/train.py` then `streamlit run app.py`. The dataset source/provenance and reuse permissions have not been independently verified. Two player-year combinations share names but represent different players; these are intentionally retained. Shooting percentages have missing values and are imputed by the ML pipeline. The latest season (2025) is held out for evaluation.

## Optional synthetic demonstration data (overwrites real processed CSV!)
```powershell
python scripts/generate_demo.py  # Optional only: replaces real processed dataset
python models/train.py
streamlit run app.py
```
The generated data are **synthetic** and MUST NOT be used as actual NBA findings.

## Input schema / importing another NBA dataset
Create `data/nba_players.csv` with one row per player per season and columns:
`player,season,salary,age,games,minutes,points,rebounds,assists,steals,blocks,fg_pct,three_pct`.
Salary is annual USD; minutes/points/rebounds/assists/steals/blocks are per-game values; shooting percentages are 0–100. Keep season labels consistent, e.g. `2023-24`. Align salary and stats to the same season and document sources, merge keys, missing data, duplicates, currency and inclusion rules. Delete `data/DEMO_ONLY.txt` once real data are loaded. Re-run training.

Potential sources: https://www.basketball-reference.com/ , https://www.nba.com/stats/ and public salary datasets. Verify licensing, data completeness and access before use. Prefer downloading validated CSVs and preserving original files separately. Never present demonstration values as NBA facts.

## Methods
- Gini coefficient and top-decile salary share quantify inequality.
- K-Means groups player-season rows using salary and performance features (descriptive, not predictive).
- Ridge regression and Random Forest estimate same-season salary from performance statistics. They do not predict future contract negotiations.
- Train on earlier seasons; hold out the most recent season. Evaluate MAE, RMSE and R² on the holdout.

## Important limitations
- Salaries are affected by contracts, tenure, salary cap, injuries and collective bargaining rules; performance alone cannot explain them.
- Current-season performance may occur after a contract is signed, so do not interpret model relationships as causal or forward-looking.
- Salary levels change over time. For a rigorous multi-season study, add salary-cap adjustment or inflation normalization.
- Gini inequality does not by itself establish a bimodal or polarized distribution. Examine histograms, quantiles and clusters too.
- Ensure consistent inclusion of two-way contracts, minimum salaries and players with limited appearances.

## Tests
```powershell
python -m pytest -q
```

## Suggested team split
Partner 1: data collection, cleaning, inequality analysis and source documentation.
Partner 2: ML experiments, Streamlit interface and visualizations. Both: testing, interpretation, presentation and report.

## Submission checklist
- [ ] Replace demo data with verified NBA data
- [ ] Document sources and preprocessing
- [ ] Explain polarization operationally
- [ ] Show Gini trends and salary distribution
- [ ] Evaluate clustering and regression on real data
- [ ] Discuss bias, limitations and model errors
- [ ] Provide screenshots, code, README and AI-use disclosure as required
