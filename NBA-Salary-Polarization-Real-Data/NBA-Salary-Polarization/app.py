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
    st.subheader("Player Explorer")

    st.dataframe(
        selected.sort_values("salary", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # Side-by-side player comparison
    st.subheader("Side-by-Side Player Comparison")

    st.caption(
        "Compare player salaries and performance statistics "
        "from the selected NBA season."
    )

    compare = selected.reset_index(drop=True).copy()

    # Create unique player labels
    compare["display_name"] = compare.apply(
        lambda r: (
            f"{r['player']} | "
            f"{r.get('team', 'N/A')} | "
            f"Age {r['age']} | "
            f"Record {r.name + 1}"
        ),
        axis=1
    )

    if len(compare) < 2:
        st.info("At least two players are required.")

    else:
        col1, col2 = st.columns(2)

        # Select first player
        with col1:
            first_idx = st.selectbox(
                "Select Player 1",
                options=list(compare.index),
                format_func=lambda i: compare.loc[i, "display_name"],
                key="compare_first"
            )

        # Select second player
        with col2:
            second_idx = st.selectbox(
                "Select Player 2",
                options=list(compare.index),
                index=1,
                format_func=lambda i: compare.loc[i, "display_name"],
                key="compare_second"
            )

        if first_idx == second_idx:
            st.warning("Please select two different players.")

        else:
            first = compare.loc[first_idx]
            second = compare.loc[second_idx]

            # Calculate salary differences
            salary_difference = abs(
                first["salary"] - second["salary"]
            )

            salary_ratio = (
                max(first["salary"], second["salary"])
                / min(first["salary"], second["salary"])
            )

            # Display salary metrics
            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                first["player"],
                f"${first['salary']:,.0f}"
            )

            c2.metric(
                second["player"],
                f"${second['salary']:,.0f}"
            )

            c3.metric(
                "Salary Difference",
                f"${salary_difference:,.0f}"
            )

            c4.metric(
                "Salary Ratio",
                f"{salary_ratio:.2f}x"
            )

            # Compare performance statistics
            st.subheader("Performance Comparison")

            display_metrics = [
                ("Salary ($)", "salary"),
                ("Age", "age"),
                ("Games Played", "games"),
                ("Minutes", "minutes"),
                ("Points", "points"),
                ("Rebounds", "rebounds"),
                ("Assists", "assists"),
                ("Steals", "steals"),
                ("Blocks", "blocks"),
                ("Field Goal %", "fg_pct"),
                ("Three-Point %", "three_pct")
            ]

            comparison = pd.DataFrame({
                "Metric": [
                    label for label, col in display_metrics
                ],
                first["player"] + " (Player 1)": [
                    first[col] for label, col in display_metrics
                ],
                second["player"] + " (Player 2)": [
                    second[col] for label, col in display_metrics
                ]
            })

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            # Salary comparison chart
            chart_data = pd.DataFrame({
                "Player": [
                    "Player 1: " + first["player"],
                    "Player 2: " + second["player"]
                ],
                "Salary": [
                    first["salary"],
                    second["salary"]
                ]
            })

            fig = px.bar(
                chart_data,
                x="Player",
                y="Salary",
                color="Player",
                title="Actual Salary Comparison",
                text_auto=".3s"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Performance comparison chart
            performance_data = pd.DataFrame({
                "Statistic": [
                    "Points",
                    "Rebounds",
                    "Assists",
                    "Steals",
                    "Blocks"
                ],
                first["player"] + " (Player 1)": [
                    first["points"],
                    first["rebounds"],
                    first["assists"],
                    first["steals"],
                    first["blocks"]
                ],
                second["player"] + " (Player 2)": [
                    second["points"],
                    second["rebounds"],
                    second["assists"],
                    second["steals"],
                    second["blocks"]
                ]
            })

            performance_fig = px.bar(
                performance_data,
                x="Statistic",
                y=[
                    first["player"] + " (Player 1)",
                    second["player"] + " (Player 2)"
                ],
                barmode="group",
                title="Player Performance Comparison"
            )

            st.plotly_chart(
                performance_fig,
                use_container_width=True
            )

            # Actual vs predicted salary
            if ART.exists():

                import numpy as np

                artifacts = joblib.load(ART)

                st.subheader("Actual vs Predicted Salary")

                model_choice = st.selectbox(
                    "Select Comparison Model",
                    list(artifacts["models"]),
                    key="comparison_model"
                )

                model = artifacts["models"][model_choice]

                player_inputs = pd.DataFrame(
                    [first, second]
                )[FEATURES]

                estimates = np.expm1(
                    model.predict(player_inputs)
                ).clip(0)

                prediction_table = pd.DataFrame({
                    "Player": [
                        first["player"],
                        second["player"]
                    ],
                    "Actual Salary ($)": [
                        first["salary"],
                        second["salary"]
                    ],
                    "Predicted Salary ($)": estimates,
                    "Salary Difference ($)": [
                        first["salary"] - estimates[0],
                        second["salary"] - estimates[1]
                    ]
                })

                st.dataframe(
                    prediction_table.style.format({
                        "Actual Salary ($)": "${:,.0f}",
                        "Predicted Salary ($)": "${:,.0f}",
                        "Salary Difference ($)": "${:,.0f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                # Actual vs predicted salary chart
                prediction_fig = px.bar(
                    prediction_table,
                    x="Player",
                    y=[
                        "Actual Salary ($)",
                        "Predicted Salary ($)"
                    ],
                    barmode="group",
                    title="Actual vs Predicted Salary"
                )

                st.plotly_chart(
                    prediction_fig,
                    use_container_width=True
                )

                st.caption(
                    "Predictions are statistical estimates, "
                    "not determinations of player worth. "
                    "Estimates for training seasons are "
                    "in-sample predictions."
                )

            else:
                st.info(
                    "Run python models/train.py "
                    "to enable salary predictions."
                )

            # Connection to polarization
            st.divider()
            st.subheader("Salary Disparity Analysis")

            higher_paid = (
                first if first["salary"] >= second["salary"]
                else second
            )

            lower_paid = (
                second if first["salary"] >= second["salary"]
                else first
            )

            st.write(
                f"**{higher_paid['player']}** earns "
                f"**{salary_ratio:.2f} times** the salary of "
                f"**{lower_paid['player']}**, representing "
                f"a salary gap of **${salary_difference:,.0f}**."
            )

            st.write(
                "Comparing player performance alongside salary "
                "helps investigate whether differences in "
                "compensation correspond to differences in "
                "measured performance."
            )

            st.caption(
                "Individual salary disparities do not independently "
                "establish league-wide polarization. Refer to the "
                "Salary Inequality tab for historical trends "
                "and overall salary distributions."
            )

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
