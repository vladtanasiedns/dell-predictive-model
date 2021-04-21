from tensorflow import keras
from joblib import load
import pandas as pd
import numpy as np
import os


# TODO: Refactor docstrings for the class, add better explanations
# TODO: Add better error handling for the functions
# TODO: Output a requirements.txt file
# TODO: Plan and implement a way to update/train the model remotely
# TODO: Implement loading data to a dataframe from sql

class SlowDispatchPrediction:
    def __init__(self, model_path=None, encoder_path=None, data_path=None):
        """
            model_path=None - or the string path of the model
            encoder_path=None - or the string path of the encoder
            data=None - the pandas DataFrame object of the data
            
            The model, encoder and data can be loaded at a later time using the support functions
            - loadModel
            - loadEncoder
            - loadData

            To encode the data use the `encode` function
            To make a prediction use the `predict` function

            Usage:
                Initiate the object, pass the path values of your data, encoder and model and then use the predictBatch
                and predictSequentially functions to get your predictions.
                Currently supported file types are csv, json, xlsx, xls, odf, ods, odt
                If the paths are not available at the time of instantiation, call the, loadModel, loadData and
                loadEncoder functions by passing the required objects,
                then call the sanitize, encode and predict functions.
        """
        if model_path is not None:
            self._model = keras.models.load_model(model_path)

        if encoder_path is not None:
            self._encoder = load(encoder_path)

        if data_path is not None:
            name, extension = os.path.splitext(data_path)
            if extension == ".csv":
                self.data = pd.read_csv(data_path)
            elif extension == ".json":
                self.data = pd.read_json(data_path)
            elif extension == ".xlsx":
                self.data = pd.read_excel(data_path)
            elif extension == ".xls":
                self.data = pd.read_excel(data_path, engine="xlrd")
            elif extension == ".odf" or ".ods" or ".odt":
                self.data = pd.read_excel(data_path, engine="odf")

        # The categorical columns that the model needs
        self.categorical_columns = ['HEADER_SVC_BU_ID', 'HEADER_DSPCH_ORD_BU_ID', 'TECH_DRCT_FLG', 'B2C_FLG', 'B2D_FLG',
                                    'DUMMY_SVC_TAG_FLG', 'NEW_PART_SHIP_FLG', 'SHIP_SVC_BU_ID', 'ITM_QTY',
                                    'ITM_COMDTY_ID', 'PART_ORD_SEQ_NBR', 'PART_ORD_LN_NBR', 'NEW_PART_FLG',
                                    'SUBSTT_FLG', 'TMZN_LOC_ID', 'DW_LD_GRP_VAL', 'ITM_COST', 'ISS_CODE', 'COST',
                                    'TWO_WEEK_TR', 'SPMD_FBSC', 'OneCost']

        # The numerical columns that the model needs
        self.numerical_columns = ['COUNTRY', 'SOURCE', 'DISPATCH_TYPE', 'CALL_TYPE', 'REASON_CODE', 'STATUS',
                                  'PARTS_STATUS', 'LOCAL_CHANNEL', 'SVC_TYPE', 'SVC_LEVEL', 'SVC_OPTIONS',
                                  'PROD_TYPE_DESC', 'PROD_GRP_DESC', 'LOB_CODE', 'LOB_DESC', 'PROD_LN_DESC',
                                  'SYS_CLASS', 'ZIP_SHIPPING', 'KYHD', 'ACCIDENTAL_DAMAGE', 'COMPLETE_CARE',
                                  'TAX_XMPTN_FLG', 'CURRENCY_CODE', 'KYC', 'ITM_NBR', 'DSPCH_ORD_STAT_CD',
                                  'MOST_XPSV_PART_FLG', 'DSTRB_HUB_NM', 'PART_TYPE_CD', 'ITEM_NUM', 'ITEM_DESC',
                                  'AGE_CLASS', 'COMMODITY', 'COMMODITY_DESC', 'ASP_FAMILY', 'RFC', 'REPAIR',
                                  'RSL_ACTIVE_FLAG', 'BOM_Parent_Part', 'AMER_PLNR_NAME', 'EMEA_PLNR_NAME',
                                  'APJ_PLNR_NAME', 'FULFIL_CENTER', 'SPMD_ACTIVE_FLAG', 'SPMD_COMMODITY']

        # Private variable of the encoded version of the data
        self._encoded_data = None

        # Sanitize the data for prediction and encoding
        if data_path is not None:
            self._sanitize()

    def loadModel(self, model):
        """
        model - already loaded from tf.keras.model.load('model path')
        """
        self._model = model

    def loadEncoder(self, encoder):
        """
        encoder - the encoder loaded from joblib.load
        """
        self._encoder = encoder

    def loadData(self, data):
        """
        data - the data for our model loaded into a data frame from pd.DataFrame
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            print("Data needs to be instance of DataFrame")

    def _sanitize(self):
        """
        Creates the ideal data frame by selecting the needed columns for a prediction.
        Performs data sanitization by replacing missing values for categorical with "NULL" and "0" for numerical.
        """
        if self.data is not None:
            cat_obj = "object"
            categorical = self.data.select_dtypes(include=cat_obj)
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            numerical = self.data.select_dtypes(include=numerics)

            numerical = [col for col in numerical.columns]
            categorical = [col for col in categorical.columns]

            # Loop trough categorical columns and replace missing values
            for col in categorical:
                self.data[col].fillna(value="NULL", inplace=True)
            # Loop trough numerical columns and replace missing values
            for col in numerical:
                self.data[col].fillna(value=np.float(0), inplace=True)

            columns = [y for x in [self.categorical_columns, self.numerical_columns] for y in x]

            # makes the data property be composed only of the columns declared in the constructor
            self.data = self.data[columns]

    def _encode(self):
        """
        Encodes the data property of the object to numerical for making a prediction.
        WARNING! If the data that is predicted on contains any extra data that the model has not seen,
        it will throws an error
        """
        try:
            if self.data is not None:
                self._encoded_data = self._encoder.transform(self.data)
        except Exception as e:
            print("There was an error while encoding the data")
            print(e)

    def predictSlowDispatchBatch(self, return_dict=False):
        """
        This function predicts in batches, this can raise errors so best used in a try/except block

        :param return_dict: if True returns a dictionary with the predicted classes and the probabilities
                for the classes
        :return: array of predicted classes
        """
        if self.data is not None:
            self._sanitize()
            self._encode()

            if not return_dict:
                return (self._model.predict(self._encoded_data) > 0.5).astype("int32")
            else:
                return {
                    "classes": (self._model.predict(self._encoded_data) > 0.5).astype("int32"),
                    "probability": self._model.predict(self._encoded_data)
                }
        else:
            print("Data has not been passed to the object, please use loadData(data_path)")
            return None

    def predictSequentially(self, output_csv=False):
        """
            Predicts if a data sample will be a slow dispatch, it does this sequentially by looping through the data,
        this might be inefficient computationally but it has error handling for those cases where the encoder encounters
        data that it has not been trained to encode

        :param output_csv: if True will save a csv file with the predictions as a new column in the original data
        :return: dictionary of predicted classes, probabilities and data samples that had errors
        """

        results = {
            "rows": [],  # Contains the indexes of the rows that had errors
            "classes": [],
            "probability": []
        }

        # Sanitize data
        if self.data is not None:
            self._sanitize()

        slice_index = 0

        for index, row in self.data.iterrows():
            try:
                encoded = self._encoded_data = self._encoder.transform(self.data[slice_index:slice_index+1])
                results["classes"].append((self._model.predict(encoded) > 0.5).astype("int32")[0][0])
                results["probability"].append((self._model.predict(encoded))[0][0])
                results["rows"].append(index)
            except Exception as e:
                results["rows"].append(index)
                results["classes"].append("ERROR")
                results["probability"].append("ERROR")

            # Increment slice index in order to move to the next row of data
            slice_index += 1

        if output_csv is False:
            return results
        else:
            self.data["SLOW_DISPATCH"] = results["classes"]
            self.data["PROBABILITY"] = results["probability"]
            self.data.to_csv("./predictions.csv", index=False)

    def printDataInfo(self):
        print(f"Shape {self.data.shape}")
        print(self.data.head())

    def __str__(self):
        return str(self._model.summary())
