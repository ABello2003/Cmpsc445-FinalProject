
# NBA Salary Polarization

NBA Salary Polarization is a machine learning web application that analyzes NBA player salaries and performance statistics from 2010–2025. The project investigates economic inequality, identifies player groups, and examines the relationship between player performance and salary disparities.

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

## Installation

### Prerequisites

- Python 3.10 or newer
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

6. Train the machine learning models:

   ```bash
   python models/train.py
   ```

## Usage

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the following URL in your web browser:

http://localhost:8501

The application contains five interactive tabs:

- **Salary Inequality:** Explore historical salary distributions, Gini coefficients, median salaries, and salary concentration.
- **Player Explorer:** View individual player statistics and compare two players' salaries, performance, and predicted salaries.
- **Clustering:** Visualize player groups generated using K-Means clustering.
- **Salary Prediction:** Enter player performance statistics to estimate salaries using Ridge Regression or Random Forest Regression.
- **Model Evaluation:** Compare machine learning models using evaluation metrics.

Use the season selector to explore NBA player data from different years.

To stop the application, press `Ctrl + C` in the terminal.

To restart the application, run:

```bash
python -m streamlit run app.py
```

**Note:** The project uses real historical NBA data. Do not run the demonstration data generator, as it may overwrite the real dataset with synthetic player records.

