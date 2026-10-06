import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

random.seed(42)

def random_sampling(data):
    # Assuming 'data' is a pandas DataFrame
    random_sample = data.sample(n=100)  # Randomly select 100 data points
    return random_sample

def stratified_sampling(data, test_size=0.2):
    # Assuming 'data' is a pandas DataFrame and 'label' is the column with subgroups
    train_set, test_set = train_test_split(data, test_size, stratify=data['label'])
    return train_set, test_set

def systematic_sampling(data, interval = 10):
    # Select every 10th data point
    systematic_sample = data.iloc[::interval]
    return systematic_sample

def cluster_sampling(data, n_clusters):
    # Assuming 'data' is a pandas DataFrame and we are dividing data into 'n_clusters'

    clusters = data.groupby(np.arange(len(data)) // n_clusters)
    selected_clusters = np.random.choice(a=clusters.ngroups, size=5, replace=False)
    cluster_sample = clusters.apply(lambda x: x.sample(frac=1)).loc[selected_clusters]
    return cluster_sample

def oversampling(X, y):
    # Assuming 'X' and 'y' are features and labels
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res

def oversampling(X, y):
    # Assuming 'X' and 'y' are features and labels
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    return X_res, y_res

def reservoir_sampling(stream, k):
    # Assuming a stream of data 'stream' and a size 'k' for the reservoir
    reservoir = []
    for i, element in enumerate(stream):
        if i+1<= k:
            reservoir.append(element)
        else:
            probability = k/(i+1)
            if random.random() < probability:
                reservoir[random.choice(range(0, k))] = element
    return reservoir