from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.analysis import FEATURES, clean_data, inequality

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'nba_players.csv'
ART = ROOT / 'models' / 'saved' / 'artifacts.joblib'
st.set_page_config(page_title='NBA Salary Polarization', page_icon='🏀', layout='wide')
st.title('🏀 NBA Salary Polarization')
st.caption('Real historical dataset: 2010–2025. Economic inequality, player segmentation, and descriptive salary modeling')
if not DATA.exists():
    st.error('Missing data/nba_players.csv. Follow README.md to add real data or generate synthetic demo data.')
    st.stop()
try:
    df = clean_data(pd.read_csv(DATA))
except ValueError as exc:
    st.error(str(exc)); st.stop()
if df.empty:
    st.error('Dataset contains no usable rows.'); st.stop()
if (ROOT / 'data' / 'DEMO_ONLY.txt').exists():
    st.warning('SYNTHETIC DEMONSTRATION DATA — not actual NBA players or salaries.')
season = st.sidebar.selectbox('Season', sorted(df.season.unique(), reverse=True))
selected = df[df.season == season]
summary = inequality(df)
tabs = st.tabs(['Salary inequality', 'Player explorer', 'Clustering', 'Salary prediction', 'Model evaluation'])
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    stats = summary[summary.season == season].iloc[0]
    c1.metric('Gini coefficient', f'{stats.gini:.3f}')
    c2.metric('Top 10% salary share', f'{stats.top_10_pct_share:.1%}')
    c3.metric('Median salary', f'${stats.median_salary:,.0f}')
    st.plotly_chart(px.histogram(selected, x='salary', nbins=25, title='Salary distribution'), use_container_width=True)
    st.plotly_chart(px.line(summary, x='season', y='gini', markers=True, title='Gini coefficient across seasons'), use_container_width=True)
    st.caption('Gini measures inequality (0 = equal salaries); inequality alone does not prove two distinct salary groups. Nominal salaries and small contract values require care when comparing seasons.')
with tabs[1]:
    st.dataframe(selected.sort_values('salary', ascending=False), use_container_width=True, hide_index=True)
    players = sorted(selected.player.unique())
    if players:
        player = st.selectbox('Player', players)
        st.write(selected[selected.player == player].T)
        st.caption('Some player names are shared by different individuals; check age and team.')
with tabs[2]:
    cluster_path = ROOT / 'models' / 'saved' / 'clustered_players.csv'
    if cluster_path.exists():
        clustered = pd.read_csv(cluster_path)
        clustered = clustered[clustered.season.astype(str) == str(season)]
        st.plotly_chart(px.scatter(clustered, x='points', y='salary', color=clustered.cluster.astype(str), hover_name='player', title='Performance / salary clusters'), use_container_width=True)
        st.dataframe(clustered.groupby('cluster')[['salary', 'points', 'assists', 'rebounds']].median(), use_container_width=True)
        st.caption('Clusters include salary as an input. Cluster numbers are arbitrary, not value judgments.')
    else:
        st.info('Run python models/train.py to generate clusters.')
with tabs[3]:
    if not ART.exists():
        st.info('Run python models/train.py to enable predictions.')
    else:
        artifacts = joblib.load(ART)
        model_name = st.selectbox('Model', list(artifacts['models']))
        with st.form('prediction'):
            inputs = {col: st.number_input(col.replace('_', ' ').title(), value=float(df[col].median()) if pd.notna(df[col].median()) else 0.0, min_value=0.0) for col in FEATURES}
            submitted = st.form_submit_button('Estimate salary')
        if submitted:
            import numpy as np
            prediction = np.expm1(artifacts['models'][model_name].predict(pd.DataFrame([inputs]))[0])
            st.metric('Estimated salary', f'${max(0, prediction):,.0f}')
            st.caption('Associational estimate based on season statistics; not a contract recommendation or future forecast.')
with tabs[4]:
    metrics_path = ROOT / 'models' / 'saved' / 'metrics.csv'
    if metrics_path.exists():
        st.write('Chronological holdout evaluation (latest season)')
        st.dataframe(pd.read_csv(metrics_path), use_container_width=True, hide_index=True)
        st.caption('MAE/RMSE are in dollars; lower is better. R² higher is better.')
    else:
        st.info('Run python models/train.py to calculate evaluation metrics.')
