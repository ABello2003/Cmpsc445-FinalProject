
# NBA Salary Polarization

NBA Salary Polarization is a machine learning web application that analyzes NBA player salaries and performance statistics from 2010–2025. The project investigates economic inequality, identifies player groups, and examines the relationship between player performance and salary disparities.

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nba-salary-polarization.streamlit.app/)

**Application URL:** https://nba-salary-polarization.streamlit.app/

The application is hosted through Streamlit Community Cloud and can be accessed from any device with a compatible web browser and an internet connection. No local installation or Python setup is required to use the hosted application.

## Features

- Interactive web application built with Streamlit.
- Historical NBA salary and performance analysis from 2010–2025.
- Salary inequality analysis using the Gini coefficient and top-10% payroll share.
- Salary polarization analysis using the Wolfson polarization index and the share of players paid near the median.
- Salaries compared as a share of each season's salary cap, so different eras are comparable.
- Salary distribution and historical trend visualizations.
- Player explorer with individual player statistics.
- Side-by-side player comparison of salaries and performance.
- K-Means clustering to identify pay/performance tiers, with the number of clusters chosen by silhouette score.
- Ridge Regression and Random Forest Regression for salary prediction.
- Actual versus predicted salary comparisons, labeled as in-sample or out-of-sample.
- Model evaluation using MAE, RMSE, R², a median baseline, permutation feature importance, and Silhouette Score.

## Technologies Used

- **Python:** Primary programming language.
- **Streamlit:** Interactive web application and user interface.
- **Pandas:** Data manipulation and preprocessing.
- **NumPy:** Numerical calculations.
- **Scikit-learn:** Machine learning, preprocessing, and model evaluation.
- **Plotly:** Interactive charts and data visualizations.
- **Joblib:** Saving and loading trained machine learning models.
- **GitHub:** Version control and source code management.
- **Streamlit Community Cloud:** Web application hosting and deployment.

## Machine Learning Models

| Model | Learning Type | Purpose |
|---|---|---|
| K-Means Clustering | Unsupervised Learning | Identify player groups based on salary (as cap share) and performance. |
| Ridge Regression | Supervised Learning | Estimate player salaries using regularized linear regression. |
| Random Forest Regression | Supervised Learning | Estimate player salaries using an ensemble of decision trees. |

**Inputs:** 12 per-game and context statistics (age, games, games started, minutes, points, rebounds, assists, steals, blocks, FG%, 3P%, 3PA).
**Output:** salary as a share of the season's salary cap (trained on its logarithm), converted back to dollars with that season's cap.
**Evaluation:** chronological holdout. The models train on 2009-10 to 2023-24 and are tested on the unseen 2024-25 season.

| Model (2024-25 holdout) | MAE | RMSE | R² |
|---|---|---|---|
| Median baseline | $8.30M | $13.61M | -0.196 |
| Ridge | $5.53M | $8.99M | 0.479 |
| Random Forest | $4.15M | $6.38M | 0.738 |

Training on nominal dollars instead of cap share drops Random Forest to R² 0.566 and Ridge to 0.385, because the salary cap more than doubled over the period. Full results are in `models/saved/metrics.csv`.

## Polarization Analysis

Inequality and polarization are treated as different things:

- **Inequality:** how unevenly the payroll is split (Gini coefficient, top-10% payroll share).
- **Polarization:** whether players are separating into a low-paid group and a high-paid group with a thinning middle (Wolfson polarization index; share of players paid 50%-150% of the median).

The project examines this at three levels:

- **League Level:** Inequality and polarization measures for every season, 2009-10 to 2024-25.
- **Group Level:** K-Means clustering of pay and performance. Silhouette score is highest at k = 2, giving a lower tier and an upper tier.
- **Individual Level:** Actual versus model-estimated salary, showing how much of the pay gap measured performance explains.

The objective is to investigate salary disparities and determine the extent to which player performance can explain differences in compensation.

## Installation

Local installation is optional. Users can access the hosted application directly through the Live Demo link above.

Follow the instructions below to run or modify the project locally.

### Prerequisites

- Python 3.11 or newer (the pinned pandas 3 release requires it)
- Git
- Visual Studio Code (recommended)

1. Clone the repository:

   ```bash
   git clone https://github.com/ABello2003/Cmpsc445-FinalProject.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Cmpsc445-FinalProject
   cd NBA-Salary-Polarization-Real-Data/NBA-Salary-Polarization
   ```

3. Create a Python virtual environment:

   ```powershell
   python -m venv .venv
   ```

4. Activate the virtual environment on Windows:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

5. Install the required dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

6. (Optional) Rebuild the processed dataset from the raw CSV:

   ```bash
   python scripts/prepare_data.py
   ```

7. (Optional) Retrain the models if the saved models are missing or the dataset has changed:

   ```bash
   python models/train.py
   ```

8. (Optional) Run the tests:

   ```bash
   python -m pytest
   ```

## Usage

### Option 1: Hosted Application (Recommended)

Open the following link in your web browser:

https://nba-salary-polarization.streamlit.app/

No installation, terminal commands, or local server is required.

### Option 2: Run Locally

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the following URL in your web browser:

http://localhost:8501

### Application Navigation

The application contains five interactive tabs:

- **Inequality & Polarization:** Explore salary distributions and season-by-season Gini, top-10% share, Wolfson polarization, near-median share, and median salary as a share of the cap.
- **Player Explorer:** View individual player statistics and compare two players' salaries, performance, salary differences, salary ratios, and predicted salaries.
- **Clustering:** Visualize K-Means pay/performance tiers, their median profiles, and how the share of players in each tier changes over time.
- **Salary Prediction:** Enter player statistics to estimate a salary (as a share of the cap and in dollars for a chosen season) using Ridge Regression or Random Forest Regression.
- **Model Evaluation:** Compare models against a baseline using MAE, RMSE and R², view actual-vs-predicted results and permutation feature importance on the held-out season, and see the silhouette score for each number of clusters.

Use the season selector to explore NBA player data from different years.

### Stopping the Local Application

To stop the application, press `Ctrl + C` in the terminal.

To restart the application, run:

```bash
python -m streamlit run app.py
```

## Project Structure

```text
Cmpsc445-FinalProject/
├── README.md
└── NBA-Salary-Polarization-Real-Data/
    └── NBA-Salary-Polarization/
        ├── app.py                      # Streamlit web application
        ├── README.md
        ├── requirements.txt            # Pinned dependency versions
        ├── DATASET_REVIEW.md           # Data source, processing, and known quirks
        ├── data/
        │   ├── raw/                    # Original source CSV
        │   ├── salary_cap.csv          # NBA salary cap by season (with sources)
        │   └── nba_players.csv         # Processed dataset (built by prepare_data.py)
        ├── scripts/
        │   └── prepare_data.py         # Raw CSV -> processed dataset
        ├── models/
        │   ├── train.py                # Trains regression and clustering models
        │   └── saved/
        │       ├── artifacts.joblib        # Trained models
        │       ├── clustered_players.csv   # Players with cluster tier labels
        │       ├── cluster_selection.csv   # Silhouette score for k = 2..7
        │       ├── feature_importance.csv  # Permutation importance (holdout)
        │       ├── metrics.csv             # Holdout MAE / RMSE / R²
        │       ├── summary.json            # Split sizes, chosen k, versions
        │       └── test_predictions.csv    # Holdout actual vs predicted
        ├── tests/
        │   ├── test_analysis.py
        │   └── test_app.py
        └── utils/
            └── analysis.py             # Cleaning + inequality/polarization measures
```

## Deployment

The application is deployed using Streamlit Community Cloud and connected to the GitHub repository.

Changes committed and pushed to the main branch are generally reflected automatically in the hosted application.

The hosted application runs independently of the developers' local computers.

## Limitations

- Salary predictions are statistical estimates and do not determine a player's actual value or future contract.
- Player salaries may be influenced by factors outside the dataset, including injuries, contract negotiations, experience, and market conditions.
- Salaries are compared as a share of the salary cap rather than adjusted for general inflation. Gini, top-10% share and Wolfson do not depend on the salary scale.
- Same-season statistics describe pay; they do not predict future contracts, because contracts are usually signed based on earlier seasons.
- Very small salaries (10-day and partial contracts) and players with few games are kept, which lowers the bottom of the distribution. See `DATASET_REVIEW.md`.
- The best clustering has a silhouette score of only about 0.29, so the two tiers overlap and players form a continuum rather than two fully separate populations.
- The hosted application may temporarily enter sleep mode after inactivity and require restarting through the website.

## Data Source

- Player statistics and salaries: `data/raw/NBA Player Stats and Salaries_2010-2025.csv`. **TODO: add the original source URL and license.**
- Salary cap by season: `data/salary_cap.csv` (sources listed per row).
