from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.analysis import FEATURES, clean_data

ROOT = Path(__file__).resolve().parents[1]

def train(path=None):
    path = Path(path) if path else ROOT / 'data' / 'nba_players.csv'
    df = clean_data(pd.read_csv(path))
    seasons = sorted(df.season.unique())
    if len(seasons) < 3:
        raise ValueError('Need at least three seasons for a chronological train/test split.')
    train_df, test_df = df[df.season.isin(seasons[:-1])], df[df.season == seasons[-1]]
    if len(train_df) < 30 or len(test_df) < 5:
        raise ValueError('Need at least 30 training and 5 test player-seasons.')
    # Same-season descriptive salary estimation, NOT prospective contract forecasting.
    X_train, X_test = train_df[FEATURES], test_df[FEATURES]
    y_train, y_test = np.log1p(train_df.salary), test_df.salary
    models = {
        'Ridge': Ridge(alpha=10),
        'Random Forest': RandomForestRegressor(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1)
    }
    fitted, scores = {}, []
    for name, estimator in models.items():
        pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()), ('model', estimator)])
        pipe.fit(X_train, y_train)
        predictions = np.expm1(pipe.predict(X_test)).clip(0)
        scores.append({'model': name, 'MAE': mean_absolute_error(y_test, predictions), 'RMSE': np.sqrt(mean_squared_error(y_test, predictions)), 'R2': r2_score(y_test, predictions)})
        fitted[name] = pipe
    # Salary included here intentionally: clusters describe economic/performance segmentation.
    cluster_features = FEATURES + ['salary']
    cluster_pipeline = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    cluster_data = cluster_pipeline.fit_transform(df[cluster_features])
    k = min(3, len(df) - 1)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(cluster_data)
    df['cluster'] = labels
    model_dir = ROOT / 'models' / 'saved'
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({'models': fitted, 'cluster_pipeline': cluster_pipeline, 'kmeans': km, 'features': FEATURES, 'test_season': seasons[-1]}, model_dir / 'artifacts.joblib')
    df.to_csv(model_dir / 'clustered_players.csv', index=False)
    pd.DataFrame(scores).to_csv(model_dir / 'metrics.csv', index=False)
    print(pd.DataFrame(scores).to_string(index=False))
    print('Silhouette:', round(silhouette_score(cluster_data, labels), 3) if len(set(labels)) > 1 else 'N/A')
    print('Held-out season:', seasons[-1])

if __name__ == '__main__':
    train()
