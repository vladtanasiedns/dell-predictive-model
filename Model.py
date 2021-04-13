# TODO: create Model class
    # Import necessary libraries
    # Import model
    # Import encoder
    # Encode data
    # Predict on data
    # Return results as dict
    # __repr__ to print results formatted


class Model:
    def __init__(self, model_path=None, encoder_path=None, data=None):
        """
            model_path=None - or the string path of the model
            encoder_path=None - or the string path of the encoder
            data=None - or the string path of the data, currently accepted only csv, json
            
            The model, encoder and data can be loaded at a later time using the support functions
            - loadModel
            - loadEncoder
            - loadData

            To encode the data use the `encode` function
            To make a prediction use the `predict` function
        """
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.data = data
    
    def loadModel(self, model):
        self.model = model

    def loadEncoder(self, encoder):
        self.encoder = encoder

    def loadData(self, data):
        self.data = data

    def encode():
        # Encode the data
        pass

    def predict(self, dict=False)
        # Return predictions
        pass

    def __repr__(self):
        # Print Model
        pass