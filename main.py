from dellai import SlowDispatchPrediction
import pandas as pd

csv = pd.DataFrame(pd.read_csv("./dell_data.csv"))

data = csv.loc[1:2]

network = SlowDispatchPrediction(model_path="model_recall_0.7809606986899563",
                                 encoder_path="missing_data_encoder.joblib")

network.loadData(data)
network.sanitize()
network.encode()

predictions = network.predictBatch()

print(predictions)
# print(network)
