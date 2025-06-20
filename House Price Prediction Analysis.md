# House Price Prediction Analysis

## Data Preparation and Feature Selection

First, we import the necessary libraries and prepare our dataset by selecting relevant features:

```python
import pandas as pd

df = pd.read_csv("AmesHousing.csv")
features_to_keep = ['Lot Area', 'Lot Shape', 'House Style', 'Neighborhood', 
                   'Year Built', 'Bsmt Cond', 'Central Air', 'Overall Cond',
                   'Full Bath', 'TotRms AbvGrd', 'Fireplaces', 'Garage Area',
                   'Yr Sold', 'SalePrice']
df = df[features_to_keep]

# Handle missing values
df["Bsmt Cond"] = df["Bsmt Cond"].fillna("None")
df["Garage Area"] = df["Garage Area"].fillna(0)
```

## Model Setup

We set up our machine learning pipeline using scikit-learn components:

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Separate features and target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define feature types
categorical_cols = ["Lot Shape", "House Style", "Neighborhood", "Bsmt Cond", 
                   "Central Air", "Overall Cond"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)

# Create full pipeline
model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("regressor", LinearRegression())
])
```

## Initial Model Performance

We first tried a basic model with raw sale prices:

```python
from sklearn.metrics import mean_squared_error, r2_score
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.4f}")
```

## Improved Model with Log Transformation

To handle the skewed nature of house prices, we applied a log transformation:

```python
import numpy as np

# Log-transform the target
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Fit model on log-transformed target
model.fit(X_train, y_train_log)

# Predict and inverse-transform
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)

# Calculate final RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (after log transformation): {rmse:.2f}")
```

## Visualization

We can visualize the model's predictions using scatter plots. The red dashed line represents perfect predictions, while the points show actual vs. predicted values.

[Note: Visualizations would appear here when rendered. You'll need to save the plots as images and include them in your markdown with:

```markdown
![Predicted vs Actual Prices](path_to_image.png)
```
]