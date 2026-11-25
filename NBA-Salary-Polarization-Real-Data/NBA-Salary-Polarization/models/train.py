"""Train the salary models and the player-segmentation clusters.

Run from the project folder (after ``python scripts/prepare_data.py``):
    python models/train.py

Design
------
Supervised (salary estimation)
    * Target: log(cap_share), where cap_share = salary / that season's salary cap.
      Nominal salaries doubled over 2010-2025 because the cap grew, so a model
      trained on raw dollars from 2010-2024 badly underestimates 2025 pay.
      Cap share removes that drift; the log handles the heavy right skew.
    * Split: chronological. Train on 2010-2024, test on the held-out 2025 season.
      A random split would leak the same player's other seasons into training.
    * Models: Ridge regression (regularized linear baseline) and Random Forest
      (non-linear ensemble). A "median" baseline predicts the training-set median
      cap share for everyone, which shows how much the models actually add.
    * Metrics: MAE / RMSE / R^2 in dollars on 2025 (predicted cap share x 2025 cap).
      The nominal-dollar target is also trained, only for comparison, so the
      report can show the effect of the cap adjustment.
    * Feature importance: permutation importance on the held-out season (drop in
      R^2 when one feature is shuffled). It is computed on unseen data and is less
      biased toward correlated features than the forest's built-in importances.

Unsupervised (segmentation)
    * K-Means on the standardized performance features plus cap_share. Salary is
      included on purpose: the clusters describe pay-and-performance tiers.
    * k is chosen by the best silhouette score over k = 2..7; the full table is
      saved so the choice can be shown and discussed.
    * Clusters are renamed by median cap share (Tier 1 = lowest paid), so the
      labels stay the same when the model is retrained.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.analysis import FEATURES, clean_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models' / 'saved'
K_RANGE = range(2, 8)
TIER_NAMES = {2: ['Lower tier', 'Upper tier'],
              3: ['Lower tier', 'Middle tier', 'Upper tier']}


def make_models():
    """Fresh, unfitted model pipelines. Imputation is a safety net for app inputs."""
    return {
        'Ridge': Pipeline([('impute', SimpleImputer(strategy='median')),
                           ('scale', StandardScaler()),
                           ('model', Ridge(alpha=10))]),
        'Random Forest': Pipeline([('impute', SimpleImputer(strategy='median')),
                                   ('model', RandomForestRegressor(
                                       n_estimators=200, min_samples_leaf=3,
                                       random_state=42, n_jobs=-1))]),
    }


def dollar_scores(name, target, actual, predicted):
    return {'model': name, 'target': target,
            'MAE': mean_absolute_error(actual, predicted),
            'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'R2': r2_score(actual, predicted)}


def tier_names(k):
    return TIER_NAMES.get(k, [f'Tier {i + 1}' for i in range(k)])


def train(path=None):
    path = Path(path) if path else ROOT / 'data' / 'nba_players.csv'
    df = clean_data(pd.read_csv(path))
    seasons = sorted(df.season.unique())
    if len(seasons) < 3:
        raise ValueError('Need at least three seasons for a chronological train/test split.')
    test_season = seasons[-1]
    train_df, test_df = df[df.season != test_season], df[df.season == test_season]
    X_train, X_test = train_df[FEATURES], test_df[FEATURES]
    test_cap = float(test_df.salary_cap.iloc[0])

    # ---- Supervised: cap-adjusted target (the models the app uses) -------------
    y_train = np.log(train_df.cap_share)
    fitted, scores, predictions = {}, [], {'player': test_df.player.values,
                                           'team': test_df['team'].values if 'team' in test_df else None,
                                           'actual_salary': test_df.salary.values}
    baseline = np.full(len(test_df), train_df.cap_share.median() * test_cap)
    scores.append(dollar_scores('Median baseline', 'cap share', test_df.salary, baseline))
    for name, pipe in make_models().items():
        pipe.fit(X_train, y_train)
        pred = np.exp(pipe.predict(X_test)) * test_cap
        scores.append(dollar_scores(name, 'cap share', test_df.salary, pred))
        predictions[name] = pred
        fitted[name] = pipe

    # ---- Comparison only: the original nominal-dollar target ------------------
    for name, pipe in make_models().items():
        pipe.fit(X_train, np.log1p(train_df.salary))
        pred = np.expm1(pipe.predict(X_test)).clip(0)
        scores.append(dollar_scores(name, 'nominal $ (no cap adjustment)', test_df.salary, pred))

    # ---- Permutation importance on the held-out season ------------------------
    importance = []
    for name, pipe in fitted.items():
        result = permutation_importance(pipe, X_test, np.log(test_df.cap_share),
                                        scoring='r2', n_repeats=10, random_state=42, n_jobs=-1)
        for feature, mean, std in zip(FEATURES, result.importances_mean, result.importances_std):
            importance.append({'model': name, 'feature': feature,
                               'importance': mean, 'std': std})

    # ---- Unsupervised: K-Means segmentation -----------------------------------
    cluster_features = FEATURES + ['cap_share']
    cluster_pipeline = Pipeline([('impute', SimpleImputer(strategy='median')),
                                 ('scale', StandardScaler())])
    cluster_data = cluster_pipeline.fit_transform(df[cluster_features])
    selection, fits = [], {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(cluster_data)
        fits[k] = km
        selection.append({'k': k, 'silhouette': silhouette_score(cluster_data, km.labels_),
                          'inertia': km.inertia_})
    selection = pd.DataFrame(selection)
    best_k = int(selection.loc[selection.silhouette.idxmax(), 'k'])
    km = fits[best_k]

    # Relabel clusters by median cap share so Tier 1 is always the lowest paid.
    raw_labels = pd.Series(km.labels_, index=df.index)
    order = df.cap_share.groupby(raw_labels).median().sort_values().index
    names = tier_names(best_k)
    label_map = {old: new for new, old in enumerate(order)}
    df['cluster'] = raw_labels.map(label_map)
    df['cluster_name'] = df.cluster.map(dict(enumerate(names)))

    # ---- Save everything the app needs ----------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({'models': fitted, 'cluster_pipeline': cluster_pipeline, 'kmeans': km,
                 'cluster_label_map': label_map, 'cluster_names': names,
                 'features': FEATURES, 'test_season': test_season},
                MODEL_DIR / 'artifacts.joblib', compress=3)
    df.to_csv(MODEL_DIR / 'clustered_players.csv', index=False)
    scores = pd.DataFrame(scores)
    scores.to_csv(MODEL_DIR / 'metrics.csv', index=False)
    pd.DataFrame(importance).to_csv(MODEL_DIR / 'feature_importance.csv', index=False)
    selection.to_csv(MODEL_DIR / 'cluster_selection.csv', index=False)
    pd.DataFrame(predictions).to_csv(MODEL_DIR / 'test_predictions.csv', index=False)
    summary = {'test_season': test_season, 'train_seasons': [seasons[0], seasons[-2]],
               'train_rows': len(train_df), 'test_rows': len(test_df),
               'k': best_k, 'silhouette': float(selection.silhouette.max()),
               'sklearn_version': sklearn.__version__}
    (MODEL_DIR / 'summary.json').write_text(json.dumps(summary, indent=2))

    print(scores.to_string(index=False))
    print(selection.round(3).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    train()
