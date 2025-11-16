# KC House Price Prediction

## Project Overview
- This repository analyzes the Kaggle `kc_house_data` dataset with the goal of predicting housing prices in King County, WA (Seattle and surrounding cities) using Multiple Linear Regression (MLR).
- The notebook `Final_Group_Project.ipynb` walks through the full lifecycle: cleaning 21,613 real transactions (May 2014–May 2015), engineering interpretable features, performing exploratory analysis, and fitting an interpretable log-price regression model.
- The study focuses on surfaces larger than \$650K to identify the amenities and locations that command premium valuations and to provide data-driven pricing guidance for agents and investors.

## Dataset
- Source: Kaggle – [House Sales in King County, USA](https://www.kaggle.com/datasets/shivachandel/kc-house-data).
- Size: 21,613 rows × 21 original columns (id, date, price, bedrooms, bathrooms, living/lot square footage, floors, waterfront, view, grade, lat/long, etc.).
- Added features inside the notebook:
  - `building_age = 2015 - yr_built` to capture depreciation.
  - `dummy_basement` to indicate whether a property has a basement.
  - `log_sqft_living` and `log_price` to stabilize skewed distributions.
  - `city` derived from ZIP codes via `uszipcode.SearchEngine`.

## Methodology
1. **Data preparation**
   - Missing data only appears in `sqft_above` and is imputed with the column mean.
   - Outliers for price, bedrooms, bathrooms, living area, and building age are trimmed using a quantile-based `remove_outliers` helper to retain representative samples.
2. **Feature engineering**
   - Core features: bedrooms, bathrooms, log living area, basement indicator, grade, building age, condition, view, waterfront.
   - One-hot encoding for `city` (drop-first to avoid the dummy variable trap) yields 32 total input columns.
3. **Exploratory Data Analysis**
   - Matplotlib, Seaborn, and Plotly visualizations illustrate how age, rooms, area, and geography influence price.
   - A correlation heatmap validates that `log_sqft_living` and `grade` maintain strong linear relationships with log-price.
4. **Modeling**
   - Target: `log_price`, improving residual normality and homoscedasticity.
   - Split: `train_test_split(test_size=0.3, random_state=100)`.
   - Estimator: `sklearn.linear_model.LinearRegression`.
   - Metrics: training R² ≈ 0.709. Residual histograms and actual vs. predicted scatter plots confirm stable fit on both splits.
   - Significance: `f_regression` p-values show most predictors are strongly associated with log-price (p < 0.05).
5. **Insights**
   - Square footage, grade, bedrooms, and property condition dominate price variance.
   - Waterfront, strong views, and affluent cities (e.g., Bellevue, Kirkland, Mercer Island) are key levers for \$650K+ listings.
   - Room count trends stay relatively flat over time; location and overall build quality drive short-term swings.

## Repository Structure
- `Final_Group_Project.ipynb` – main notebook with code, visuals, and commentary.
- `kc_house_data.csv` – raw dataset needed for replication.
- `Project Proposal.pdf` – early-stage proposal and problem framing.
- `Final_Group_Project.md` & `Final_Group_Project_files/` – static export generated via `nbconvert` (optional reference).

## Quick Start
1. **Clone**
   ```bash
   git clone https://<your-clone-url>/kc-house-price-prediction-datascience.git
   cd kc-house-price-prediction-datascience
   ```
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -U pip
   pip install numpy pandas matplotlib seaborn plotly scikit-learn uszipcode geopandas shapely jupyter
   ```
   > `uszipcode` will download a local database the first time it runs; allow a minute for initialization.
4. **Launch the notebook**
   ```bash
   jupyter lab  # or jupyter notebook
   ```
   Open `Final_Group_Project.ipynb` to rerun the workflow or browse `Final_Group_Project.md` for a static view.

## Reproducing Key Results
| Stage | Description | Output |
| --- | --- | --- |
| Cleaning | Mean imputation + quantile-based trimming of extreme values | 13,946 training / 5,977 testing samples |
| Feature set | Log transforms, engineered features, city dummies | 32 explanatory variables |
| Estimation | `LinearRegression` on log-price | Training R² ≈ 0.709 |
| Diagnostics | Residual histograms, actual vs. predicted scatterplots | Errors centered near 0; predictions align with 45° line |

## Next Steps
1. Evaluate regularized linear models (Ridge/Lasso/ElasticNet) and tree ensembles (Random Forest, XGBoost) for better generalization.
2. Enrich the dataset with macroeconomic indicators (interest rates, CPI) or neighborhood-level signals (school scores, walkability).
3. Automate feature selection and hyperparameter tuning pipelines to streamline A/B testing.

## Acknowledgments
- Data: Kaggle – `House Sales in King County, USA`.
- ZIP-to-city mapping: `uszipcode`.
- Visualization: Matplotlib, Seaborn, Plotly.

Contributions via Issues or Pull Requests are welcome. Happy modeling!
