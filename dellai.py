from tensorflow import keras
from joblib import load
import pandas as pd
import numpy as np
import os


# TODO: Handle unseen values that get into the model (OOV - Out of Vocabulary values)
# TODO: Refactor docstrings for the class, add better explanations
# TODO: Add better error handling for the functions
# TODO: Refactor the name of the library and functions
# TODO: Output a requirements.txt file
# TODO: Separate DataFrame from encoded data


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
            self.model = keras.models.load_model(model_path)

        if encoder_path is not None:
            self.encoder = load(encoder_path)

        if data_path is not None:
            name, extension = os.path.splitext(data_path)

            if extension is ".csv":
                self.data = pd.read_csv(data_path)
            elif extension is ".json":
                self.data = pd.read_json(data_path)
            elif extension is ".xlsx":
                self.data = pd.read_excel(data_path)
            elif extension is ".xls":
                self.data = pd.read_excel(data_path, engine="xlrd")
            elif extension is ".odf" or ".ods" or ".odt":
                self.data = pd.read_excel(data_path, engine="odf")

        # The categorical columns that the model needs
        self.categorical_columns = ['COUNTRY', 'SOURCE', 'DISPATCH_TYPE', 'CALL_TYPE',
                                    'REASON_CODE', 'CREATOR_LOGIN', 'OWNER_LOGIN', 'STATUS', 'PARTS_STATUS',
                                    'LOCAL_CHANNEL', 'SVC_TYPE', 'SVC_LEVEL', 'SVC_OPTIONS', 'PROD_TYPE_DESC',
                                    'PROD_GRP_DESC', 'LOB_CODE', 'LOB_DESC', 'PROD_LN_DESC', 'SYS_CLASS',
                                    'ZIP_SHIPPING', 'KYHD', 'ACCIDENTAL_DAMAGE', 'COMPLETE_CARE', 'TAX_XMPTN_FLG',
                                    'CURRENCY_CODE', 'KYC', 'ITM_NBR', 'DSPCH_ORD_STAT_CD', 'MOST_XPSV_PART_FLG',
                                    'DSTRB_HUB_NM', 'PART_TYPE_CD', 'ITEM_NUM', 'ITEM_DESC', 'AGE_CLASS', 'COMMODITY',
                                    'COMMODITY_DESC', 'ASP_FAMILY', 'RFC', 'REPAIR', 'RSL_ACTIVE_FLAG',
                                    'BOM_Parent_Part', 'AMER_PLNR_NAME', 'EMEA_PLNR_NAME', 'APJ_PLNR_NAME',
                                    'FULFIL_CENTER', 'SPMD_ACTIVE_FLAG', 'SPMD_COMMODITY']

        # The numerical columns that the model needs
        self.numerical_columns = ['HEADER_SVC_BU_ID', 'HEADER_DSPCH_ORD_BU_ID', 'TECH_DRCT_FLG', 'B2C_FLG', 'B2D_FLG',
                                  'DUMMY_SVC_TAG_FLG', 'NEW_PART_SHIP_FLG', 'SHIP_SVC_BU_ID', 'ITM_QTY',
                                  'ITM_COMDTY_ID', 'PART_ORD_SEQ_NBR', 'PART_ORD_LN_NBR', 'NEW_PART_FLG', 'SUBSTT_FLG',
                                  'TMZN_LOC_ID', 'DW_LD_GRP_VAL', 'ITM_COST', 'ISS_CODE', 'COST', 'TWO_WEEK_TR',
                                  'SPMD_FBSC', 'OneCost']

        # Private variable of the encoded version of the data
        self._encoded_data = None

        # If there is a path for the encoder and for the data the class will sanitize and encode the data for the model
        if encoder_path is not None and data_path is not None:
            self.sanitize()
            self.encode()

    def loadModel(self, model):
        """
        model - already loaded from tf.keras.model.load('model path')
        """
        self.model = model

    def loadEncoder(self, encoder):
        """
        encoder - the encoder loaded from joblib.load
        """
        self.encoder = encoder

    def loadData(self, data):
        """
        data - the data for our model loaded into a data frame from pd.DataFrame
        """
        if isinstance(data, pd.DataFrame):
            self.data = data

    def sanitize(self):
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

    def encode(self):
        """
        Encodes the data property of the object to numerical for making a prediction.
        WARNING! If the data that is predicted on contains any extra data that the model has not seen,
        it will throws an error
        """
        try:
            if self.data is not None:
                self._encoded_data = self.encoder.transform(self.data)
        except Exception:
            print("There was an error while encoding the data")
            print(Exception)

    def predictBatch(self, return_dict=False):
        # TODO: Implement saving the predictions to file
        """
        Use this function if there were no errors during the encoding of the data

        :param return_dict: if True returns a dictionary with the predicted classes and the probabilities
                for the classes
        :return: array of predicted classes
        """
        if self.data is not None:
            if not return_dict:
                return (self.model.predict(self._encoded_data) > 0.5).astype("int32")
            else:
                return {
                    "classes": (self.model.predict(self._encoded_data) > 0.5).astype("int32"),
                    "probability": self.model.predict(self._encoded_data)
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

        # TODO: Implement predicting sequentially on the data and return predictions,
        #  for error sample save them separately
        # TODO: Implement saving the results to file and save in the same file predictions and error samples

        # Read file
        #   Loop trough data samples then encode and predict each row
        #       Handle possible errors and save the error rows for later
        #   Match prediction with the row in the data and save the row and the prediction in a dict
        #   Save predictions and errors in a new column on the data passed in a csv file
        pass

    def __str__(self):
        return str(self.model.summary())
