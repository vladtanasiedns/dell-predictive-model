from tensorflow import keras
from joblib import load
import pandas as pd
import numpy as np
import sys
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


class DellModelPrediction:
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
                Initiate the object, pass the path values of your data, encoder and model then use the predict model to get the prediction.
                If the paths are not available at the time of instantiation, call the, loadModel, loadData and loadEncoder functions by passing the required objects,
                then call the sanitize, encode and predict functions.
        """
        self.model = keras.models.load_model(model_path)
        self.encoder = load(encoder_path)
        self.data = pd.DataFrame(pd.read_csv(data_path))

        self.categorical_columns = ['COUNTRY', 'GBL_SVC_PRVDR_ID', 'SOURCE', 'DISPATCH_TYPE', 'CALL_TYPE',
                                    'REASON_CODE', 'CREATOR_LOGIN', 'OWNER_LOGIN', 'STATUS', 'PARTS_STATUS',
                                    'LOCAL_CHANNEL', 'SVC_TYPE', 'SVC_LEVEL', 'SVC_OPTIONS', 'PROD_TYPE_DESC',
                                    'PROD_GRP_DESC', 'LOB_CODE', 'LOB_DESC', 'PROD_LN_DESC', 'SYS_CLASS',
                                    'ZIP_SHIPPING', 'KYHD', 'ACCIDENTAL_DAMAGE', 'COMPLETE_CARE', 'TAX_XMPTN_FLG',
                                    'CURRENCY_CODE', 'KYC', 'ITM_NBR', 'DSPCH_ORD_STAT_CD', 'MOST_XPSV_PART_FLG',
                                    'DSTRB_HUB_NM', 'PART_TYPE_CD', 'ITEM_NUM', 'ITEM_DESC', 'AGE_CLASS', 'COMMODITY',
                                    'COMMODITY_DESC', 'ASP_FAMILY', 'RFC', 'REPAIR', 'RSL_ACTIVE_FLAG',
                                    'BOM_Parent_Part', 'AMER_PLNR_NAME', 'EMEA_PLNR_NAME', 'APJ_PLNR_NAME',
                                    'FULFIL_CENTER', 'SPMD_ACTIVE_FLAG', 'SPMD_COMMODITY']

        self.numerical_columns = ['HEADER_SVC_BU_ID', 'HEADER_DSPCH_ORD_BU_ID', 'TECH_DRCT_FLG', 'B2C_FLG', 'B2D_FLG',
                                  'DUMMY_SVC_TAG_FLG', 'NEW_PART_SHIP_FLG', 'SHIP_SVC_BU_ID', 'ITM_QTY',
                                  'ITM_COMDTY_ID', 'PART_ORD_SEQ_NBR', 'PART_ORD_LN_NBR', 'NEW_PART_FLG', 'SUBSTT_FLG',
                                  'TMZN_LOC_ID', 'DW_LD_GRP_VAL', 'ITM_COST', 'ISS_CODE', 'COST', 'TWO_WEEK_TR',
                                  'SPMD_FBSC', 'OneCost']

        if self.encoder is not None and self.data is not None:
            self.sanitize()
            self.encode()

    def loadModel(self, model):
        """
        model - already loaded from tf.keras.model.load
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

        cat_obj = "object"
        categorical = self.data.select_dtypes(include=cat_obj)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numerical = self.data.select_dtypes(include=numerics)

        numerical = [col for col in numerical.columns]
        categorical = [col for col in categorical.columns]

        # Loop trough categorical columns and replace missing valaues
        for col in categorical:
            self.data[col].fillna(value="NULL", inplace=True)
        # Loop trough numerical columns and replace missing valaues
        for col in numerical:
            self.data[col].fillna(value=np.float(0), inplace=True)

        columns = [y for x in [self.categorical_columns, self.numerical_columns] for y in x]

        # makes the data property be composed only of the columns declared in the constructor
        self.data = self.data[columns]

    def encode(self):
        """
        Encodes the data property of the object to numerical for making a prediction
        """
        self.data = self.encoder.transform(self.data)

    def predict(self, return_dict=False):
        if not return_dict:
            return (self.model.predict(self.data) > 0.5).astype("int32")
        else:
            pred_dict = {
                "classes": (self.model.predict(self.data) > 0.5).astype("int32"),
                "probability": self.model.predict(self.data)
            }

    def __str__(self):
        return self.model.summary()
