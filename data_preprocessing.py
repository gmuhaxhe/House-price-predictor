# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import random
import numpy as np
import torch
import os
import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_preprocessed_data(csv_path="AmesHousing.csv"):
    df = pd.read_csv(csv_path)

    features_to_keep = [
        'Bsmt Half Bath', 'Bsmt Full Bath', 'Total Bsmt SF', 'Lot Area',
        'Lot Shape', 'House Style', 'Neighborhood', 'Year Built',
        'Bsmt Cond', 'Central Air', 'Overall Cond', 'Full Bath',
        'TotRms AbvGrd', 'Fireplaces', 'Garage Area', 'Yr Sold', 'SalePrice'
    ]
    df = df[features_to_keep]

    # Fill missing values
    df["Bsmt Cond"] = df["Bsmt Cond"].fillna("None")
    df["Garage Area"] = df["Garage Area"].fillna(0)
    df["Total Bsmt SF"] = df["Total Bsmt SF"].fillna(0)
    df["Bsmt Full Bath"] = df["Bsmt Full Bath"].fillna(0)
    df["Bsmt Half Bath"] = df["Bsmt Half Bath"].fillna(0)

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    categorical_cols = ["Lot Shape", "House Style", "Neighborhood", "Central Air"]
    ordinal_cols = ["Bsmt Cond"]
    numerical_cols = [col for col in X.columns if col not in (categorical_cols + ordinal_cols)]

    bsmt_cond_order = [['Po','None','Fa','Gd','TA', 'Ex']]

    # Define transformers
    numeric_transformer = Pipeline([('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
    ordinal_transformer = Pipeline([('ordinal', OrdinalEncoder(categories=bsmt_cond_order))])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
        ("ord", ordinal_transformer, ordinal_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, preprocessor
