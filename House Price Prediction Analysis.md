# House Price Prediction Analysis Using Ames Housing Dataset

## Data Preparation and Feature Selection

First, we import the necessary libraries and prepare our dataset by selecting relevant features:
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
```
Next, we take a look at the features that are included in this dataset.
```python
df.columns
```
```output
Index(['Order', 'PID', 'MS SubClass', 'MS Zoning', 'Lot Frontage', 'Lot Area',
       'Street', 'Alley', 'Lot Shape', 'Land Contour', 'Utilities',
       'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1',
       'Condition 2', 'Bldg Type', 'House Style', 'Overall Qual',
       'Overall Cond', 'Year Built', 'Year Remod/Add', 'Roof Style',
       'Roof Matl', 'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type',
       'Mas Vnr Area', 'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual',
       'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1',
       'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF',
       'Heating', 'Heating QC', 'Central Air', 'Electrical', '1st Flr SF',
       '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath',
       'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr',
       'Kitchen AbvGr', 'Kitchen Qual', 'TotRms AbvGrd', 'Functional',
       'Fireplaces', 'Fireplace Qu', 'Garage Type', 'Garage Yr Blt',
       'Garage Finish', 'Garage Cars', 'Garage Area', 'Garage Qual',
       'Garage Cond', 'Paved Drive', 'Wood Deck SF', 'Open Porch SF',
       'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Pool QC',
       'Fence', 'Misc Feature', 'Misc Val', 'Mo Sold', 'Yr Sold', 'Sale Type',
       'Sale Condition', 'SalePrice'],
      dtype='object')
```
To simplify our study, we only select the features that can have the most impact in a house sale price. We expect neighborhood, condition, and the size of the house including square footage and number of bedrooms to have the biggest impact. On that note, we select the following features to be included in our study. 

```python

df = pd.read_csv("AmesHousing.csv")
features_to_keep = ['Bsmt Half Bath','Bsmt Full Bath','Total Bsmt SF','Lot Area','Lot Shape','House Style','Neighborhood', 'Year Built','Bsmt Cond','Central Air','Overall Cond','Full Bath','TotRms AbvGrd','Fireplaces','Garage Area','Yr Sold','SalePrice']
df = df[features_to_keep]
```
Next, we check to see if the features we have selected include any NaN values
```python
print(df.isnull().sum())
```
```output
Bsmt Half Bath    2
Bsmt Full Bath    2
Total Bsmt SF     1
Lot Area          0
Lot Shape         0
House Style       0
Neighborhood      0
Year Built        0
Bsmt Cond         0
Central Air       0
Overall Cond      0
Full Bath         0
TotRms AbvGrd     0
Fireplaces        0
Garage Area       1
Yr Sold           0
SalePrice         0
dtype: int64
```
We see that Bsmt Cond has 80 such values while Garage Area has one. We choose to turn the Basement Condition ones to None since we do not have information about the basement of those houses and we fill 0 for the Garage Area that is missing.
```python 
# Handle missing values
df["Garage Area"] = df["Garage Area"].fillna(0)
df["Total Bsmt SF"] = df["Total Bsmt SF"].fillna(0)
df["Bsmt Full Bath"] = df["Bsmt Full Bath"].fillna(0)
df["Bsmt Half Bath"] = df["Bsmt Half Bath"].fillna(0)
```
## Visualizing and analyzing the data
We check the type of data each feature includes. This helps us separate the features between categorical and numerical later on.
```python
df.info()
```
```output
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2930 entries, 0 to 2929
Data columns (total 17 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Bsmt Half Bath  2930 non-null   float64
 1   Bsmt Full Bath  2930 non-null   float64
 2   Total Bsmt SF   2930 non-null   float64
 3   Lot Area        2930 non-null   int64  
 4   Lot Shape       2930 non-null   object 
 5   House Style     2930 non-null   object 
 6   Neighborhood    2930 non-null   object 
 7   Year Built      2930 non-null   int64  
 8   Bsmt Cond       2930 non-null   object 
 9   Central Air     2930 non-null   object 
 10  Overall Cond    2930 non-null   int64  
 11  Full Bath       2930 non-null   int64  
 12  TotRms AbvGrd   2930 non-null   int64  
 13  Fireplaces      2930 non-null   int64  
 14  Garage Area     2930 non-null   float64
 15  Yr Sold         2930 non-null   int64  
 16  SalePrice       2930 non-null   int64  
dtypes: float64(4), int64(8), object(5)
memory usage: 389.3+ KB
```
We can also check the correlation heatmap for the numerical features that we have to get an idea how related the features are.
```python
num_df = df.select_dtypes(include=['int64','float64'])
correlation_matrix = num_df.corr()
plt.figure(figsize=(13,7))
sns.heatmap(correlation_matrix, cmap ='coolwarm', fmt =".2f", annot=True)
plt.title("Correlation Heatmap for Numerical Features", fontsize=15, fontweight='bold')
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()
```
![Correlation Heat Map](images/correlation_heatmap.png)

## Model Setup

We set up our machine learning pipeline using scikit-learn, starting by separating the data into training and testing sets, 80% and 20% respectively.

```python
# Separate features and target
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Split the data into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define feature types
categorical_cols = ["Lot Shape", "House Style", "Neighborhood", "Bsmt Cond", 
                   "Central Air"]
numeric_cols = [col for col in X.columns if col not in categorical_cols]

# Transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
    ]
)
```
## Training and Obtaining the Results

Now, we train three different models to see which of them performs better. We will use Linear Regression, Random Forest Regressor, and XGBRegressor. We will run them all with their default hyperparameter values, but later on, we will show how one can do a search for the hyperparameters through cross validation.

```python
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100,random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

results = []

for name, reg in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),   # same preprocessor for all
        ("regressor", reg)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    mpe = np.mean((y_test - y_pred) / y_test) * 100

    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "MAPE (%)": mape,
        "MPE (%)": mpe
    })

# Convert to DataFrame for comparison
results_df = pd.DataFrame(results).sort_values(by="RMSE")
print(results_df)
```
```output
               Model     RMSE     MAE     R²      MAPE (%)   MPE (%)
1      Random Forest  30090.05  17823.73  0.89      9.67    -2.38
2            XGBoost  33198.76  19154.70  0.86     10.05    -2.00
0  Linear Regression  37866.82  23799.43  0.82     12.66    -0.80

```

We can see that XGBoost and Random Forest perform almost identically with all the metrics being close between the two models. Linear regression seems to do relatively worse which indicates nonlinearities in the data.

## Visualizing the Results

We now visualize the difference between the three models used with some graphs.

```python
results_df = pd.DataFrame(results)

# Set figure size and create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot RMSE
sns.barplot(ax=axes[0], data=results_df, x="Model", y="RMSE", palette="Blues_d")
axes[0].set_title("RMSE by Model")

# Plot MAE
sns.barplot(ax=axes[1], data=results_df, x="Model", y="MAE", palette="Oranges_d")
axes[1].set_title("MAE by Model")

# Plot R²
sns.barplot(ax=axes[2], data=results_df, x="Model", y="R²", palette="Greens_d")
axes[2].set_title("R² Score by Model")

# Clean up layout
for ax in axes:
    ax.set_xlabel("Model")
    ax.set_ylabel("")

plt.tight_layout()
plt.show()
```
![Model comparison](images/model_comparison.png)

Finally, we visualize the predicted sale price versus the actual sale price for the houses for all three models. 
![Model comparison](images/sale_price_comparison.png)
