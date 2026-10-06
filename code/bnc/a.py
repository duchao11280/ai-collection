import websocket
import json
import threading
import requests
import calendar
import time
import csv
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from autogluon.core.learning_curves.plot_curves import plot_curves
# def on_message(ws, message):
#     data = json.loads(message)
#     if 'stream' in data:
#         # you can pipeline this data to your function, analysis or backtesting
#         print(f"Symbol: {data['data']['s']}, Price: {data['data']['c']}, Time: {data['data']['E']}")
#     else:
#         print(f"Received message: {message}")

# def on_error(ws, error):
#     print(f"Error: {error}")

# def on_close(ws, close_status_code, close_msg):
#     print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

# def on_open(ws):
#     print("WebSocket connection opened")
#     # Subscribe to the ticker stream for BTCUSDT
#     subscribe_message = {
#         "method": "SUBSCRIBE",
#         "params": ["achusdt@trade", ],
#         "id": 1
#     }
#     ws.send(json.dumps(subscribe_message))

# def on_ping(ws, message):
#     print(f"Received ping: {message}")
#     ws.send(message, websocket.ABNF.OPCODE_PONG)
#     print(f"Sent pong: {message}")

# if __name__ == "__main__":
#     websocket.enableTrace(True)
#     socket = 'wss://stream.binance.com:9443/ws'
#     ws = websocket.WebSocketApp(socket,
#                                 on_message=on_message,
#                                 on_error=on_error,
#                                 on_close=on_close,
#                                 on_open=on_open,
#                                 on_ping=on_ping)
#     ws.run_forever()
def convert_datetime_to_long(datetime):
    # convert '2025-12-31 23:59:59' to 1893455999000

    ts =calendar.timegm(time.strptime(datetime, '%Y-%m-%dT%H:%M:%S.%fZ'))
    return ts

def write_csv(csv_path, data):
    """ write data into csv

    Args:
        csv_path (str): csv path
        data (List): List data
    """
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

s_date = '2025-01-30T05:00:00.000Z'
time_start = convert_datetime_to_long(s_date)
COIN = 'ACHUSDT'
INTERVAL = '15m'

response = requests.get(f'https://api.binance.com/api/v3/klines?interval={INTERVAL}&symbol={COIN}')

data = response.json()
cols = ['item_id', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
write_csv('data.csv', cols)

for kline in data:
    kline.insert(0, COIN)
    write_csv('data.csv', kline)
df = pd.read_csv("data.csv")
# time series
# train_data = TimeSeriesDataFrame.from_data_frame(
#     df,
#     id_column="item_id",
#     timestamp_column="open_time",
#     static_features_df=df,
# )

# predictor = TimeSeriesPredictor(
#     prediction_length=24,
#     path="0201",
#     target="close",
#     eval_metric="MASE",
# )

# predictor.fit(
#     train_data,
#     presets="medium_quality",
#     time_limit=600,
# )

# lgb
SEED = 42
x_train, x_temp, y_train, y_temp = train_test_split(df.drop(columns='close'), df['close'], test_size=0.3, random_state=SEED, shuffle=False)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=SEED, shuffle=False)
train_data = pd.concat([x_train, y_train], axis=1)
valid_data = pd.concat([x_valid, y_valid], axis=1)
predictor = TabularPredictor(
        label="close",
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
y_pred = predictor.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
print("MAE: ", mae)
print(predictor.model_best)

# fig = plot_curves(predictor.learning_curves(), "LightGBM", "mean_absolute_error", return_fig = True)
# fig.savefig("learning_curve_autogluon.png")
