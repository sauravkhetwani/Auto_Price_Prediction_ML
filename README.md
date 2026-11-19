# 🚗 Automobile Price Prediction

An end-to-end machine learning pipeline that predicts a car's **price** from its technical and categorical specifications, using the UCI Automobile (Imports-85) dataset.

## Highlights

- Full workflow: data loading → EDA → cleaning → feature engineering → preprocessing → model selection → hyperparameter tuning → evaluation → export
- Leakage-free `scikit-learn` `Pipeline` + `ColumnTransformer` (imputation, scaling and one-hot encoding are fit on training data only)
- Compares 11 regression algorithms with 5-fold cross-validation
- Tunes models with `GridSearchCV` and `Optuna`
- Engineered features: `power_to_weight`, `avg_mpg`, `engine_size_per_cyl`, `volume_proxy`, `length_to_wheelbase`

## Dataset

`auto_imports.csv` — 200 rows, 26 columns (UCI Automobile dataset). The file has **no header row** and missing values are encoded as `?`; the notebook handles both. Target variable: `price`.

## Results

Cross-validated model comparison (5-fold R², training set):

| Model | CV R² |
|---|---|
| ExtraTrees | 0.870 |
| GradientBoosting | 0.861 |
| RandomForest | 0.858 |
| Ridge | 0.847 |
| XGBoost | 0.841 |

Held-out test set (20% split) after tuning:

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| **Gradient Boosting (GridSearchCV)** | **0.952** | $1,592 | $2,433 | 9.55% |
| Extra Trees (GridSearchCV) | 0.944 | $1,582 | $2,612 | 8.99% |
| Extra Trees (Optuna) | 0.942 | $1,613 | $2,667 | 8.90% |

The final exported model is the tuned Gradient Boosting pipeline.

## Project Structure

```
.
├── auto_price_prediction_final.ipynb   # Main notebook (full pipeline)
├── auto_imports.csv                    # Dataset
├── requirements.txt                    # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook auto_price_prediction_final.ipynb
```

Running the notebook end-to-end saves the trained pipeline as `price_prediction_pipeline.pkl`. Note: the Optuna search (400 trials) and the grid searches can take a while.

## Using the Saved Model

```python
import joblib
import pandas as pd

model = joblib.load("price_prediction_pipeline.pkl")
prediction = model.predict(new_car_dataframe)  # DataFrame with the same columns/engineered features used in training
```

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · Optuna · Matplotlib · Seaborn


