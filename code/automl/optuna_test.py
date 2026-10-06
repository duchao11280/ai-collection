import os
from sklearn.datasets import fetch_california_housing
import optuna.integration.lightgbm as optuna_lgb
from lightgbm import early_stopping, log_evaluation
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve
import time
SEED = 42
housing = fetch_california_housing(as_frame=True)
print(housing.DESCR)
x_train, x_temp, y_train, y_temp = train_test_split(housing.data, housing.target, test_size=0.3, random_state=SEED)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=SEED)

if __name__ == "__main__":
    start_time = time.perf_counter()
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
    print(f"Time taken: {end_time - start_time:.3f}s")
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

    ax = lgb.plot_metric(model)
    plt.show()
