# Dataset suitability review

Source file supplied by user: `NBA Player Stats and Salaries_2010-2025.csv`. External provenance and reuse license not verified.

- 7,298 rows, 31 source columns, years 2010–2025 (16 seasons), 1,658 distinct player names.
- No missing salary values, no nonpositive salaries.
- Missing shooting percentages: FG% 20, 3P% 585, 2P% 45, eFG% 20, FT% 202.
- Two duplicate player-year name pairs correspond to distinct players (Chris Johnson in 2013; Tony Mitchell in 2014). Do not deduplicate on name/year.
- Salaries are nominal USD. Extremely small salaries warrant investigation of partial-year and ten-day contracts.
- Team abbreviations vary historically. No stable player ID.
- Performance features are per-game averages, not totals.
- Gini/top-decile salary shares are useful inequality measures; polarization requires separate interpretation of groups or gaps.
- Same-season model estimates are descriptive, not predictions of future contracts.
- Test on held-out 2025 season; avoid random train/test split across player-seasons.
- Verify original dataset provenance and licensing before public publication.
