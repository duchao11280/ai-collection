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
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.core.learning_curves.plot_curves import plot_curves

SEED = 42
housing = fetch_california_housing(as_frame=True)
x_train, x_temp, y_train, y_temp = train_test_split(housing.data, housing.target, test_size=0.3, random_state=SEED)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=SEED)
train_data = pd.concat([x_train, y_train], axis=1)
valid_data = pd.concat([x_valid, y_valid], axis=1)
print(train_data.dtypes)
if __name__ == "__main__":
    start_time = time.perf_counter()
    predictor = TabularPredictor(
        label="MedHouseVal",
        eval_metric="mean_absolute_error",
        problem_type='regression').fit(
            train_data=train_data,
            tuning_data=valid_data, # if None, model will automatically hold out some random validation example from train_data
            time_limit=1000,
            included_model_types=['GBM'],
            keep_only_best=True,
            fit_weighted_ensemble=True,
            learning_curves={
                "metrics": ["mean_absolute_error"],
            },

    )
    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.3f}s")
    y_pred = predictor.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE: ", mae)
    print(predictor.model_best)

    fig = plot_curves(predictor.learning_curves(), "LightGBM", "mean_absolute_error", return_fig = True)
    fig.savefig("learning_curve_autogluon.png")

    # # print("best params:", predictor)

    # model_file = os.path.join("./", f"model_autogluon.pkl")
    # pickle.dump(predictor, open(model_file, "wb"))
    # model = lgb.LGBMRegressor(**predictor.params)

    # # Train the model using the training data.
    # model.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_valid,y_valid)], callbacks=[early_stopping(100), log_evaluation(100)])
    # check = model.predict(x_test)
    # print("new mae",mean_absolute_error(y_test, check))

    # ax = lgb.plot_metric(model)
    # plt.show()
