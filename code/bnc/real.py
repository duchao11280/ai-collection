import websocket
import json
import threading
import requests
import calendar
import time
import csv
import os
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from autogluon.core.learning_curves.plot_curves import plot_curves
def on_message(ws, message):
    data = json.loads(message)
    if 'stream' in data:
        # you can pipeline this data to your function, analysis or backtesting
        # print(f"Symbol: {data['data']['s']}, Price: {data['data']['c']}, Time: {data['data']['E']}")
        pass
    else:
        # print(f"Received message: {message}")
        mess = json.loads(message)
        os.system('cls')
        print(mess.get('p'))

def on_error(ws, error):
    # print(f"Error: {error}")
    pass

def on_close(ws, close_status_code, close_msg):
    print(f"WebSocket connection closed: {close_status_code} - {close_msg}")

def on_open(ws):
    print("WebSocket connection opened")
    # Subscribe to the ticker stream for BTCUSDT
    subscribe_message = {
        "method": "SUBSCRIBE",
        "params": ["achusdt@trade", ],
        "id": 1
    }
    ws.send(json.dumps(subscribe_message))

def on_ping(ws, message):
    # print(f"Received ping: {message}")
    ws.send(message, websocket.ABNF.OPCODE_PONG)
    # print(f"Sent pong: {message}")

if __name__ == "__main__":
    websocket.enableTrace(True)
    socket = 'wss://stream.binance.com:9443/ws'
    ws = websocket.WebSocketApp(socket,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close,
                                on_open=on_open,
                                on_ping=on_ping)
    ws.run_forever()
