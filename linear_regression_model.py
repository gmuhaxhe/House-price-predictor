# linear_model.py

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd

def run_linear_model(X_train, X_test, y_train, y_test, preprocessor):
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results = {
        "Model": "Linear Regression",
        "RMSE": rmse,
        "R²": r2,
        "MAE": mae,
        "MAPE (%)": mape
    }

    return results
