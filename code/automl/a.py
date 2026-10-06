import pickle

# load trainer pkl
with open('../automl/AutogluonModels/ag-20250114_145726/models/LightGBM/model.pkl', 'rb') as f:
    predictor = pickle.load(f)
    print(predictor.params)