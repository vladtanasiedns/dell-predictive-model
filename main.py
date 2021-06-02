from dellai import SlowDispatchPredictionModel
import pandas as pd

csv = pd.read_csv("./dell_data.csv")

network = SlowDispatchPredictionModel(model_path="model_recall_0.7809606986899563",
                                      encoder_path="missing_data_encoder.joblib")
# network.encode()

network.loadDataFromBigQuery()
network.predictSequentially(output_csv=True)
# preds_dict = network.predictSlowDispatchBatch(return_dict=True)
# print(preds_dict)

# print(csv[0:1])
# print(csv[1:2])
# print(csv[2:3])
