from dellai import SlowDispatchPrediction
import pandas as pd

csv = pd.read_csv("./dell_data.csv")

network = SlowDispatchPrediction(model_path="model_recall_0.7809606986899563",
                                 encoder_path="missing_data_encoder.joblib",
                                 data_path="./dell_data.csv")
# network.encode()

# predictions = network.predictBatch()
# print(predictions)
# print(network)
# network.printDataInfo()

network.predictSequentially(output_csv=True)

# print(csv[0:1])
# print(csv[1:2])
# print(csv[2:3])
