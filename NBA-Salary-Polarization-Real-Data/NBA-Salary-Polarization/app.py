"""NBA Salary Polarization: Streamlit web app.

Five tabs, built on the files written by ``scripts/prepare_data.py`` and
``models/train.py``:
  1. Salary inequality & polarization: league-level measures for each season.
  2. Player explorer: individual salaries, side-by-side comparison, and
     actual-vs-model salary for two players.
  3. Clustering: K-Means pay/performance tiers and how they change over time.
  4. Salary prediction: live inference from user-entered statistics.
  5. Model evaluation: holdout metrics, feature importance, cluster selection.
"""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.analysis import FEATURES, FEATURE_LABELS, clean_data, inequality

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'nba_players.csv'
SAVED = ROOT / 'models' / 'saved'
ART = SAVED / 'artifacts.joblib'

st.set_page_config(page_title='NBA Salary Polarization', page_icon='🏀', layout='wide')


# ---------------------------------------------------------------------------
# Cached loaders: the model file is large, so load it once per server process
# instead of on every widget interaction.
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return clean_data(pd.read_csv(DATA))


@st.cache_data
def load_csv(name):
    path = SAVED / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def load_artifacts():
    return joblib.load(ART) if ART.exists() else None


def predict_salary(model, rows, cap):
    """The models predict log(salary / cap); convert back to dollars for a season's cap."""
    return np.exp(model.predict(rows[FEATURES])) * cap


st.title('🏀 NBA Salary Polarization')
st.caption('Real NBA player salaries and per-game statistics, 2009-10 to 2024-25 seasons. '
           'Economic inequality, polarization, player segmentation, and salary modeling.')

if not DATA.exists():
    st.error('Missing data/nba_players.csv. Run python scripts/prepare_data.py.')
    st.stop()
try:
    df = load_data()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
if df.empty:
    st.error('Dataset contains no usable rows.')
    st.stop()

artifacts = load_artifacts()
summary = inequality(df)
caps = df.groupby('season').salary_cap.first()
season_labels = df.groupby('season').season_label.first() if 'season_label' in df else caps.index.to_series()

season = st.sidebar.selectbox('Season', sorted(df.season.unique(), reverse=True),
                              format_func=lambda s: season_labels[s])
selected = df[df.season == season]
st.sidebar.metric('Salary cap', f'${caps[season]:,.0f}')

with st.expander('What does "polarization" mean in this project?'):
    st.markdown(
        """
We study **economic polarization of NBA pay**: whether the league is splitting into a
small group of very highly paid players and a large group near the minimum, with
fewer players in between. We look at three levels:

* **League:** *inequality* (Gini, top-10% payroll share) and *polarization*
  (the Wolfson index and the share of players near the median salary).
* **Groups:** K-Means finds pay/performance tiers. The number of tiers is chosen
  by silhouette score, and the data favors **two** clear groups.
* **Individuals:** regression models estimate the salary a player's statistics
  are usually paid. The gap between that estimate and the actual salary shows how
  much of the pay divide performance explains.

Salaries are compared as a **share of each season's salary cap**, because the
cap more than doubled between 2010 and 2025 and nominal dollars are not
comparable across seasons. All results are associations, not causes.
        """
    )

tabs = st.tabs(['Inequality & polarization', 'Player explorer', 'Clustering',
                'Salary prediction', 'Model evaluation'])

# ---------------------------------------------------------------------------
# Tab 1: league-level inequality and polarization
# ---------------------------------------------------------------------------
with tabs[0]:
    stats = summary[summary.season == season].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Gini coefficient', f'{stats.gini:.3f}')
    c2.metric('Top 10% payroll share', f'{stats.top_10_pct_share:.1%}')
    c3.metric('Wolfson polarization', f'{stats.wolfson:.3f}')
    c4.metric('Players near median pay', f'{stats.middle_share:.1%}',
              help='Share of players paid 50%-150% of the season median salary.')
    c5.metric('Median salary', f'${stats.median_salary:,.0f}',
              help=f'{stats.median_cap_share:.1%} of the salary cap')

    hist = px.histogram(selected, x='cap_share', nbins=40,
                        title=f'Salary distribution, {season_labels[season]} (share of salary cap)',
                        labels={'cap_share': 'Salary as share of cap'})
    hist.update_xaxes(tickformat='.0%')
    st.plotly_chart(hist, width='stretch')

    trend = summary.assign(season_label=summary.season.map(season_labels))
    left, right = st.columns(2)
    ineq = px.line(trend, x='season_label', y=['gini', 'top_10_pct_share'], markers=True,
                   title='Inequality across seasons',
                   labels={'season_label': 'Season', 'value': 'Value', 'variable': 'Measure'})
    left.plotly_chart(ineq, width='stretch')
    polar = px.line(trend, x='season_label', y=['wolfson', 'middle_share'], markers=True,
                    title='Polarization across seasons',
                    labels={'season_label': 'Season', 'value': 'Value', 'variable': 'Measure'})
    right.plotly_chart(polar, width='stretch')

    median_fig = px.line(trend, x='season_label', y='median_cap_share', markers=True,
                         title='Median salary as a share of the cap',
                         labels={'season_label': 'Season', 'median_cap_share': 'Median cap share'})
    median_fig.update_yaxes(tickformat='.1%')
    st.plotly_chart(median_fig, width='stretch')
    st.caption('Gini and top-10% share measure inequality. The Wolfson index rises when the lower '
               'and upper halves move away from the median, and a shrinking "near median" '
               'share means a thinner middle class. Gini, top-10% share and Wolfson are not '
               'affected by cap growth.')

# ---------------------------------------------------------------------------
# Tab 2: player explorer and side-by-side comparison
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader('Player Explorer')
    st.dataframe(selected.sort_values('salary', ascending=False),
                 width='stretch', hide_index=True)
    st.divider()

    st.subheader('Side-by-Side Player Comparison')
    st.caption('Compare player salaries and performance statistics from the selected season.')
    compare = selected.sort_values('salary', ascending=False).reset_index(drop=True)
    # Name + team + age keeps the two same-name/same-season players distinguishable.
    compare['display_name'] = (compare.player + ' | ' + compare.get('team', 'N/A').astype(str)
                               + ' | Age ' + compare.age.astype(int).astype(str))

    if len(compare) < 2:
        st.info('At least two players are required.')
    else:
        col1, col2 = st.columns(2)
        first_idx = col1.selectbox('Select Player 1', compare.index,
                                   format_func=lambda i: compare.loc[i, 'display_name'],
                                   key='compare_first')
        second_idx = col2.selectbox('Select Player 2', compare.index, index=1,
                                    format_func=lambda i: compare.loc[i, 'display_name'],
                                    key='compare_second')

        if first_idx == second_idx:
            st.warning('Please select two different players.')
        else:
            first, second = compare.loc[first_idx], compare.loc[second_idx]
            label1, label2 = first.player + ' (Player 1)', second.player + ' (Player 2)'
            salary_difference = abs(first.salary - second.salary)
            salary_ratio = max(first.salary, second.salary) / min(first.salary, second.salary)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(first.player, f'${first.salary:,.0f}', help=f'{first.cap_share:.1%} of cap')
            c2.metric(second.player, f'${second.salary:,.0f}', help=f'{second.cap_share:.1%} of cap')
            c3.metric('Salary Difference', f'${salary_difference:,.0f}')
            c4.metric('Salary Ratio', f'{salary_ratio:.2f}x')

            st.subheader('Performance Comparison')
            display_metrics = [('Salary ($)', 'salary'), ('Share of cap', 'cap_share')] + \
                [(FEATURE_LABELS[f], f) for f in FEATURES]
            comparison = pd.DataFrame({
                'Metric': [label for label, _ in display_metrics],
                label1: [first[col] for _, col in display_metrics],
                label2: [second[col] for _, col in display_metrics],
            })
            st.dataframe(comparison, width='stretch', hide_index=True)

            chart_data = pd.DataFrame({'Player': ['Player 1: ' + first.player, 'Player 2: ' + second.player],
                                       'Salary': [first.salary, second.salary]})
            st.plotly_chart(px.bar(chart_data, x='Player', y='Salary', color='Player',
                                   title='Actual Salary Comparison', text_auto='.3s'),
                            width='stretch')

            box_stats = ['points', 'rebounds', 'assists', 'steals', 'blocks']
            performance_data = pd.DataFrame({
                'Statistic': [FEATURE_LABELS[s] for s in box_stats],
                label1: [first[s] for s in box_stats],
                label2: [second[s] for s in box_stats],
            })
            st.plotly_chart(px.bar(performance_data, x='Statistic', y=[label1, label2],
                                   barmode='group', title='Player Performance Comparison'),
                            width='stretch')

            # Actual vs model-estimated salary
            if artifacts is None:
                st.info('Run python models/train.py to enable salary predictions.')
            else:
                st.subheader('Actual vs Predicted Salary')
                model_choice = st.selectbox('Select Comparison Model', list(artifacts['models']),
                                            key='comparison_model')
                players = pd.DataFrame([first, second])
                estimates = predict_salary(artifacts['models'][model_choice], players, caps[season])
                prediction_table = pd.DataFrame({
                    'Player': [first.player, second.player],
                    'Actual Salary ($)': [first.salary, second.salary],
                    'Predicted Salary ($)': estimates,
                    'Actual - Predicted ($)': [first.salary - estimates[0], second.salary - estimates[1]],
                })
                st.dataframe(prediction_table.style.format({c: '${:,.0f}' for c in prediction_table.columns[1:]}),
                             width='stretch', hide_index=True)
                st.plotly_chart(px.bar(prediction_table, x='Player',
                                       y=['Actual Salary ($)', 'Predicted Salary ($)'],
                                       barmode='group', title='Actual vs Predicted Salary'),
                                width='stretch')
                if season == artifacts['test_season']:
                    st.success(f'{season_labels[season]} is the held-out test season: these are '
                               'genuine out-of-sample estimates.')
                else:
                    st.warning(f'{season_labels[season]} was part of the training data, so these '
                               'estimates are in-sample and look more accurate than the model really is. '
                               f'Select {season_labels[artifacts["test_season"]]} for out-of-sample estimates.')
                st.caption('Predictions are statistical estimates of what similar statistics are usually '
                           'paid, not judgments of player worth.')

            st.divider()
            st.subheader('Salary Disparity Analysis')
            higher, lower = (first, second) if first.salary >= second.salary else (second, first)
            st.write(f'**{higher.player}** earns **{salary_ratio:.2f} times** the salary of '
                     f'**{lower.player}**, a salary gap of **${salary_difference:,.0f}**.')
            st.write('Comparing performance alongside salary helps show whether differences in '
                     'pay correspond to differences in measured performance.')
            st.caption('One pair of players says nothing about the whole league; see the first tab '
                       'for league-wide trends.')

# ---------------------------------------------------------------------------
# Tab 3: K-Means clustering
# ---------------------------------------------------------------------------
with tabs[2]:
    clustered = load_csv('clustered_players.csv')
    if clustered is None or 'cluster_name' not in clustered:
        st.info('Run python models/train.py to generate clusters.')
    else:
        clustered['season'] = clustered.season.astype(str)
        names = artifacts['cluster_names'] if artifacts else sorted(clustered.cluster_name.unique())
        this_season = clustered[clustered.season == season]
        scatter = px.scatter(this_season, x='points', y='cap_share', color='cluster_name',
                             category_orders={'cluster_name': names}, hover_name='player',
                             hover_data={'salary': ':$,.0f', 'minutes': True, 'age': True},
                             title=f'Pay / performance tiers, {season_labels[season]}',
                             labels={'points': 'Points per game', 'cap_share': 'Salary as share of cap',
                                     'cluster_name': 'Tier'})
        scatter.update_yaxes(tickformat='.0%')
        st.plotly_chart(scatter, width='stretch')

        profile = (clustered.groupby('cluster_name')
                   [['cap_share', 'salary', 'minutes', 'points', 'games_started', 'age']]
                   .median().reindex(names))
        profile.insert(0, 'players (all seasons)', clustered.cluster_name.value_counts().reindex(names))
        st.write('Median profile of each tier (all seasons)')
        st.dataframe(profile.style.format({'cap_share': '{:.1%}', 'salary': '${:,.0f}',
                                           'minutes': '{:.1f}', 'points': '{:.1f}',
                                           'games_started': '{:.0f}', 'age': '{:.0f}'}),
                     width='stretch')

        shares = (clustered.groupby('season').cluster_name.value_counts(normalize=True)
                  .rename('share').reset_index())
        shares['season_label'] = shares.season.map(season_labels)
        tier_trend = px.area(shares, x='season_label', y='share', color='cluster_name',
                             category_orders={'cluster_name': names},
                             title='Share of players in each tier by season',
                             labels={'season_label': 'Season', 'share': 'Share of players',
                                     'cluster_name': 'Tier'})
        tier_trend.update_yaxes(tickformat='.0%')
        st.plotly_chart(tier_trend, width='stretch')
        st.caption('Clusters use salary (as cap share) together with the performance statistics, '
                   'so they describe pay-and-performance tiers. Tier names are ordered by median '
                   'pay and are not value judgments.')

# ---------------------------------------------------------------------------
# Tab 4: live salary prediction (model inference)
# ---------------------------------------------------------------------------
with tabs[3]:
    if artifacts is None:
        st.info('Run python models/train.py to enable predictions.')
    else:
        st.write('Enter a player\'s per-game statistics to estimate the salary such statistics '
                 'are usually paid. The estimate is a share of the cap, shown in the dollars '
                 'of the season you choose.')
        defaults = df[df.season == artifacts['test_season']][FEATURES].median()
        with st.form('prediction'):
            top1, top2 = st.columns(2)
            model_name = top1.selectbox('Model', list(artifacts['models']))
            cap_season = top2.selectbox('Express in salary cap of season', sorted(caps.index, reverse=True),
                                        format_func=lambda s: f'{season_labels[s]} (${caps[s] / 1e6:,.1f}M cap)')
            cols = st.columns(3)
            inputs = {}
            for i, col in enumerate(FEATURES):
                bounds = {'min_value': 0.0, 'max_value': float(df[col].max())}
                if col in ('fg_pct', 'three_pct'):
                    bounds['max_value'] = 1.0
                step = 1.0 if col in ('age', 'games', 'games_started') else 0.1
                step = 0.01 if col in ('fg_pct', 'three_pct') else step
                inputs[col] = cols[i % 3].number_input(FEATURE_LABELS[col], value=float(round(defaults[col], 2)),
                                                       step=step, **bounds)
            submitted = st.form_submit_button('Estimate salary')
        if submitted:
            row = pd.DataFrame([inputs])
            share = float(np.exp(artifacts['models'][model_name].predict(row[FEATURES]))[0])
            c1, c2 = st.columns(2)
            c1.metric('Estimated salary', f'${share * caps[cap_season]:,.0f}')
            c2.metric('Share of salary cap', f'{share:.2%}')
            st.caption('An associational estimate based on same-season statistics: not a contract '
                       'recommendation or a forecast of future pay.')

# ---------------------------------------------------------------------------
# Tab 5: model evaluation
# ---------------------------------------------------------------------------
with tabs[4]:
    metrics = load_csv('metrics.csv')
    if metrics is None:
        st.info('Run python models/train.py to calculate evaluation metrics.')
    else:
        info_path = SAVED / 'summary.json'
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        test_label = season_labels.get(str(info.get('test_season', '')), info.get('test_season', ''))
        st.subheader('Salary regression: chronological holdout')
        st.write(f'Trained on {info.get("train_rows", "?"):,} player-seasons from earlier seasons, tested on '
                 f'{info.get("test_rows", "?"):,} players from the held-out {test_label} season.')
        st.dataframe(metrics.style.format({'MAE': '${:,.0f}', 'RMSE': '${:,.0f}', 'R2': '{:.3f}'}),
                     width='stretch', hide_index=True)
        st.caption('MAE/RMSE are in dollars (lower is better); R² is higher-is-better. "Median baseline" '
                   'predicts the same typical salary for everyone. The "nominal $" rows train on '
                   'unadjusted dollars and are shown only to show the effect of the cap adjustment.')

        preds = load_csv('test_predictions.csv')
        if preds is not None and artifacts:
            model_pick = st.selectbox('Model', list(artifacts['models']), key='eval_model')
            scatter = px.scatter(preds, x='actual_salary', y=model_pick, hover_name='player',
                                 title=f'Actual vs predicted salary, {test_label} (unseen data)',
                                 labels={'actual_salary': 'Actual salary ($)', model_pick: 'Predicted salary ($)'})
            top = max(preds.actual_salary.max(), preds[model_pick].max())
            scatter.add_shape(type='line', x0=0, y0=0, x1=top, y1=top, line=dict(dash='dash', color='gray'))
            st.plotly_chart(scatter, width='stretch')

        importance = load_csv('feature_importance.csv')
        if importance is not None:
            importance['label'] = importance.feature.map(FEATURE_LABELS)
            fig = px.bar(importance.sort_values('importance'), x='importance', y='label', color='model',
                         barmode='group', orientation='h', error_x='std',
                         title='Permutation feature importance on the held-out season',
                         labels={'importance': 'Drop in R² when the feature is shuffled', 'label': ''})
            st.plotly_chart(fig, width='stretch')

        st.subheader('Clustering')
        selection = load_csv('cluster_selection.csv')
        if selection is not None:
            st.metric(f'Silhouette score (k = {info.get("k", "?")})', f'{info.get("silhouette", float("nan")):.3f}')
            st.plotly_chart(px.line(selection, x='k', y='silhouette', markers=True,
                                    title='Silhouette score by number of clusters (higher = better separated)'),
                            width='stretch')
            st.caption('k is chosen by the highest silhouette score. A score around 0.3 means the '
                       'groups are real but overlap: players form a continuum with two dense regions, '
                       'not two completely separate populations.')
