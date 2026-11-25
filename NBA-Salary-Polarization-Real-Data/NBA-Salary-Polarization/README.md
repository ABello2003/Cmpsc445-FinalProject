# NBA Salary Polarization

Machine learning web application analyzing NBA salary inequality and polarization, 2009-10 to 2024-25.
See the [main README](../../README.md) for the full description, methodology, results, and live demo link.

## Quick start (from this folder, Python 3.11+)

```bash
python -m pip install -r requirements.txt
python scripts/prepare_data.py   # optional: rebuild data/nba_players.csv from data/raw/
python models/train.py           # optional: retrain models into models/saved/
python -m pytest                 # optional: run tests
python -m streamlit run app.py   # open http://localhost:8501
```

Data source, processing steps, and known data quirks: [DATASET_REVIEW.md](DATASET_REVIEW.md).
