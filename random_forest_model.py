from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def run_random_forest(X_train, X_test, y_train, y_test, preprocessor):
    rf_pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    results = {
        "Model": "Random Forest",
        "RMSE": rmse,
        "R²": r2,
        "MAE": mae,
        "MAPE (%)": mape,
        "Best Params": grid_search.best_params_
    }

    return results
