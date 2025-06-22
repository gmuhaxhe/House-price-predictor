from data_preprocessing import get_preprocessed_data
from data_preprocessing import set_seed
from linear_regression_model import run_linear_model
from random_forest_model import run_random_forest
from xgboost_model import run_xgboost
from neural_net import run_neural_net
import pandas as pd


def main():
    set_seed(42)
    X_train, X_test, y_train, y_test, preprocessor = get_preprocessed_data("AmesHousing.csv")

    results = []
    results.append(run_linear_model(X_train, X_test, y_train, y_test, preprocessor))
    results.append(run_random_forest(X_train, X_test, y_train, y_test, preprocessor))
    results.append(run_xgboost(X_train, X_test, y_train, y_test, preprocessor))
    results.append(run_neural_net(X_train, X_test, y_train, y_test, preprocessor))

    df_results = pd.DataFrame(results)
    print("\nModel Comparison Results:")
    print(df_results.sort_values(by="RMSE"))


if __name__ == "__main__":
    main()
