import pickle
from datetime import datetime
import calendar
import time
import requests
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor, TabularDataset
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
# load trainer pkl

with open('./AutogluonModels/ag-20250201_153941/models/WeightedEnsemble_L2/model.pkl', 'rb') as f:
    predictor = pickle.load(f)

time_now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
print(time_now)
def convert_datetime_to_long(datetime):
    # convert '2025-12-31 23:59:59' to 1893455999000

    ts =calendar.timegm(time.strptime(datetime, '%Y-%m-%dT%H:%M:%S.%fZ'))
    return ts
COIN = 'ACHUSDT'
INTERVAL = '5m'

# interval = 1 minute

response = requests.get(f'https://api.binance.com/api/v3/klines?interval={INTERVAL}&symbol={COIN}')
data = response.json()
for idx, e in enumerate(data[-1]):
    try:
        data[-1][idx] = float(e)
    except:
        pass

print(data[-1][4])
del data[-1][4]
array = np.array([data[-1]], dtype=float)
# data[-1].insert(0, COIN)
cols = ['open_time', 'open', 'high', 'low', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
df = pd.DataFrame(array, columns=cols)
data_test = TabularDataset(df)
print(df.dtypes)
# auto_ml_pipeline_feature_generator = AutoMLPipelineFeatureGenerator()
# df = auto_ml_pipeline_feature_generator.fit_transform(X=data_test)
# print(df.head())
prediction = predictor.predict(data_test, transform_features=True)

print(prediction)