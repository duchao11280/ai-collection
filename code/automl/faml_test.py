import os
from sklearn.datasets import fetch_california_housing

from lightgbm import early_stopping, log_evaluation
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import matplotlib.pyplot as plt
import time
import pandas as pd

from flaml import AutoML
from flaml.automl.model import LGBMEstimator

SEED = 42
housing = fetch_california_housing(as_frame=True)
x_train, x_temp, y_train, y_temp = train_test_split(housing.data, housing.target, test_size=0.3, random_state=SEED)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=SEED)

if __name__ == "__main__":
    start_time = time.perf_counter()
    predictor = AutoML()
    settings = {
        "time_budget": 1000,
        "metric": "mae",
        "estimator_list": ["lgbm"],
        "task": "regression",
        "log_file_name": "experiment.log",
        "seed": SEED,
        "eval_method": "holdout", # "cv" or "holdout"
    }
    predictor.fit(X_train=x_train, y_train=y_train, X_val=x_valid, y_val=y_valid, **settings)
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.3f}s")
    print("Best hyperparmeter config:", predictor.best_config)
    config = predictor.best_config.copy()
    config['metric'] = 'mae'
    print(config)
    y_pred = predictor.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: ", mae)


    model = lgb.LGBMRegressor(**config)

    # Train the model using the training data.
    model.fit(x_train, y_train, eval_set=[(x_valid,y_valid)])
    check = model.predict(x_test)
    print("new mae",mean_absolute_error(y_test, check))

    ax = lgb.plot_metric(model)
    plt.show()
