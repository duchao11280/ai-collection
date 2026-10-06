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
import optuna.integration.lightgbm as optuna_lgb
from flaml import AutoML
from flaml.automl.model import LGBMEstimator
from autogluon.core.learning_curves.plot_curves import plot_curves
from autogluon.tabular import TabularPredictor, TabularDataset

SEED = 42
housing = fetch_california_housing(as_frame=True)
x_train, x_temp, y_train, y_temp = train_test_split(housing.data, housing.target, test_size=0.3, random_state=SEED)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=SEED)
train_data = pd.concat([x_train, y_train], axis=1)
valid_data = pd.concat([x_valid, y_valid], axis=1)

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Optuna
lgb_learn = optuna_lgb.Dataset(x_train, y_train)
lgb_valid = optuna_lgb.Dataset(x_valid, y_valid)

param = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'random_seed': SEED,
    'verbosity': -1,
}

predictor = optuna_lgb.train(
    param,
    lgb_learn,
    valid_sets=[lgb_learn, lgb_valid],
    callbacks=[early_stopping(100), log_evaluation(100)],
    time_budget=1000,
    optuna_seed=SEED,
)
end_time = time.perf_counter()

y_pred = predictor.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)
print("best params:", predictor.params)
model_file = os.path.join("./", f"model_optuna.pkl")
pickle.dump(predictor, open(model_file, "wb"))
model = lgb.LGBMRegressor(**predictor.params)

# Train the model using the training data.
model.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_valid,y_valid)], callbacks=[early_stopping(100), log_evaluation(100)])
check = model.predict(x_test)
print("new mae",mean_absolute_error(y_test, check))

ax1 = lgb.plot_metric(model, ax=axs[0])
axs[0].set_title("Optuna")


## Autogluon
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

y_pred = predictor.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)
print(predictor.model_best)

plot_curves(predictor.learning_curves(), "LightGBM", "mean_absolute_error", return_fig = True, ax=axs[1])
axs[1].set_title("Autogluon Learning Curve")

config = {'n_estimators': 2921, 'num_leaves': 150, 'min_child_samples': 2, 'learning_rate': 0.008194291128432538,
          'log_max_bin': 10, 'colsample_bytree': 0.747147429822089, 'reg_alpha': 0.13384572125432315, 'reg_lambda': 0.014781508385816664, 'metric': 'mae'}
# FAML
model = lgb.LGBMRegressor(**config)

# Train the model using the training data.
model.fit(x_train, y_train, eval_set=[(x_valid,y_valid)])
check = model.predict(x_test)
print("new mae",mean_absolute_error(y_test, check))

ax3 = lgb.plot_metric(model,ax=axs[2])

plt.show()
