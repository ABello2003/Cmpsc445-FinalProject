
# NBA Salary Polarization

NBA Salary Polarization is a machine learning web application that analyzes NBA player salaries and performance statistics from 2010–2025. The project investigates economic inequality, identifies player groups, and examines the relationship between player performance and salary disparities.

## Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nba-salary-polarization.streamlit.app/)

**Application URL:** https://nba-salary-polarization.streamlit.app/

The application is hosted through Streamlit Community Cloud and can be accessed from any device with a compatible web browser and an internet connection. No local installation or Python setup is required to use the hosted application.

## Features

- Interactive web application built with Streamlit.
- Historical NBA salary and performance analysis from 2010–2025.
- Salary inequality analysis using the Gini coefficient.
- Salary distribution and historical trend visualizations.
- Player explorer with individual player statistics.
- Side-by-side player comparison of salaries and performance.
- K-Means clustering to identify salary and performance groups.
- Ridge Regression and Random Forest Regression for salary prediction.
- Actual versus predicted salary comparisons.
- Model evaluation using MAE, RMSE, R², and Silhouette Score.

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
| K-Means Clustering | Unsupervised Learning | Identify player groups based on salary and performance. |
| Ridge Regression | Supervised Learning | Estimate player salaries using regularized linear regression. |
| Random Forest Regression | Supervised Learning | Estimate player salaries using an ensemble of decision trees. |

## Polarization Analysis

The project examines economic polarization within the NBA at three levels:

- **League Level:** Analyze historical salary inequality, salary distributions, and salary concentration.
- **Group Level:** Identify distinct player groups using K-Means clustering.
- **Individual Level:** Compare player salaries, performance statistics, and actual versus predicted salaries.

The objective is to investigate salary disparities and determine the extent to which player performance can explain differences in compensation.

## Installation

Local installation is optional. Users can access the hosted application directly through the Live Demo link above.

Follow the instructions below to run or modify the project locally.

### Prerequisites

- Python 3.10 or newer, compatible with the project dependencies
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

6. Train the machine learning models if the saved models are missing or the dataset has been modified:

   ```bash
   python models/train.py
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

- **Salary Inequality:** Explore historical salary distributions, Gini coefficients, median salaries, and salary concentration.
- **Player Explorer:** View individual player statistics and compare two players' salaries, performance, salary differences, salary ratios, and predicted salaries.
- **Clustering:** Visualize player groups generated using K-Means clustering.
- **Salary Prediction:** Enter player performance statistics to estimate salaries using Ridge Regression or Random Forest Regression.
- **Model Evaluation:** Compare machine learning models using MAE, RMSE, R², and clustering Silhouette Score.

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
        ├── app.py
        ├── README.md
        ├── requirements.txt
        ├── DATASET_REVIEW.md
        ├── data/
        │   ├── raw/
        │   └── nba_players.csv
        ├── models/
        │   ├── train.py
        │   └── saved/
        │       ├── artifacts.joblib
        │       ├── clustered_players.csv
        │       └── metrics.csv
        ├── scripts/
        ├── tests/
        │   └── test_analysis.py
        └── utils/
            └── analysis.py
```

## Deployment

The application is deployed using Streamlit Community Cloud and connected to the GitHub repository.

Changes committed and pushed to the main branch are generally reflected automatically in the hosted application.

The hosted application runs independently of the developers' local computers.

## Limitations

- Salary predictions are statistical estimates and do not determine a player's actual value or future contract.
- Player salaries may be influenced by factors outside the dataset, including injuries, contract negotiations, experience, and market conditions.
- Historical salaries are expressed in nominal dollars and are not adjusted for inflation or changes in the NBA salary cap.
- Salary inequality and individual salary disparities do not independently establish the existence of distinct polarized groups.
- The hosted application may temporarily enter sleep mode after inactivity and require restarting through the website.

## Important Note

The project uses real historical NBA data. Do not run the demonstration data generator, as it may overwrite the real dataset with synthetic player records.

