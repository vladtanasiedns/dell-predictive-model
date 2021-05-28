from tensorflow import keras
from joblib import load
import pandas as pd
import numpy as np
import os

import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage


# TODO: Refactor docstrings for the class, add better explanations
# TODO: Add better error handling for the functions
# TODO: Output a requirements.txt file
# TODO: Implement loading data to a dataframe from sql

class SlowDispatchPredictionModel:
    def __init__(self, model_path=None, encoder_path=None, data_path=None):
        """
        model_path=None - or the string path of the model encoder_path=None - or the string path of the encoder
        data_path=None - the path to the data file (csv, json, xlsx, xls, [odf, ods, odt] - Libre office formats)
        recommended to always use csv

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
        self._dispatch_nums = None

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
            cat_obj = 'object'
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
        This function can be used for faster predictions when batches are passed, this can fail if the encoder encounter
        unknown data, use this with caution, recommended using within try/except block

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

    def loadDataFromBigQuery(self, query_string=None):
        # Explicitly create a credentials object. This allows you to use the same
        # credentials for both the BigQuery and BigQuery Storage clients, avoiding
        # unnecessary API calls to fetch duplicate authentication tokens.

        if query_string is None:
            query_string = """
            select
            DISPATCH_NUM,
            HEADER_SVC_BU_ID,
            HEADER_DSPCH_ORD_BU_ID,
            TECH_DRCT_FLG,
            B2C_FLG,
            B2D_FLG,
            DUMMY_SVC_TAG_FLG,
            NEW_PART_SHIP_FLG,
            SHIP_SVC_BU_ID,
            ITM_QTY,
            ITM_COMDTY_ID,
            PART_ORD_SEQ_NBR,
            PART_ORD_LN_NBR,
            NEW_PART_FLG,
            SUBSTT_FLG,
            TMZN_LOC_ID,
            DW_LD_GRP_VAL,
            ITM_COST,
            ISS_CODE,
            COST,
            TWO_WEEK_TR,
            SPMD_FBSC,
            OneCost,
            COUNTRY,
            SOURCE,
            DISPATCH_TYPE,
            CALL_TYPE,
            REASON_CODE,
            STATUS,
            PARTS_STATUS,
            LOCAL_CHANNEL,
            SVC_TYPE,
            SVC_LEVEL,
            SVC_OPTIONS,
            PROD_TYPE_DESC,
            PROD_GRP_DESC,
            LOB_CODE,
            LOB_DESC,
            PROD_LN_DESC,
            SYS_CLASS,
            ZIP_SHIPPING,
            KYHD,
            ACCIDENTAL_DAMAGE,
            COMPLETE_CARE,
            TAX_XMPTN_FLG,
            CURRENCY_CODE,
            KYC,
            ITM_NBR,
            DSPCH_ORD_STAT_CD,
            MOST_XPSV_PART_FLG,
            DSTRB_HUB_NM,
            PART_TYPE_CD,
            ITEM_NUM,
            ITEM_DESC,
            AGE_CLASS,
            COMMODITY,
            COMMODITY_DESC,
            ASP_FAMILY,
            RFC,
            REPAIR,
            RSL_ACTIVE_FLAG,
            BOM_Parent_Part,
            AMER_PLNR_NAME,
            EMEA_PLNR_NAME,
            APJ_PLNR_NAME,
            FULFIL_CENTER,
            SPMD_ACTIVE_FLAG,
            SPMD_COMMODITY,

            if(coalesce(LENGTH(RCV_TRAN_DATE),0) < 3 and coalesce(LENGTH(X_ISP_RTNED_DTS),0) < 3, 1, 0) as not_returned, 
            DATE_DIFF(DATE(IF(coalesce(LENGTH(RCV_TRAN_DATE),0) < 3, 
                          IF(coalesce(LENGTH(X_ISP_RTNED_DTS),0) < 3, CURRENT_TIMESTAMP(), TIMESTAMP(X_ISP_RTNED_DTS)), 
                          PARSE_TIMESTAMP("%m/%d/%Y %I:%M:%S %p",REPLACE(RCV_TRAN_DATE,"'","")))),
                  DATE(DBR_CL_DATES_1_O), DAY) AS velocity,
            if(DATE_DIFF(DATE(IF(coalesce(LENGTH(RCV_TRAN_DATE),0) < 3, 
                          IF(coalesce(LENGTH(X_ISP_RTNED_DTS),0) < 3, CURRENT_TIMESTAMP(), TIMESTAMP(X_ISP_RTNED_DTS)), 
                          PARSE_TIMESTAMP("%m/%d/%Y %I:%M:%S %p",REPLACE(RCV_TRAN_DATE,"'","")))),
                     DATE(DBR_CL_DATES_1_O), DAY)   
            < 31, 0, 1) as slow_30_dispatch,
            if(DATE_DIFF(DATE(IF(coalesce(LENGTH(RCV_TRAN_DATE),0) < 3, 
                          IF(coalesce(LENGTH(X_ISP_RTNED_DTS),0) < 3, CURRENT_TIMESTAMP(), TIMESTAMP(X_ISP_RTNED_DTS)), 
                          PARSE_TIMESTAMP("%m/%d/%Y %I:%M:%S %p",REPLACE(RCV_TRAN_DATE,"'","")))),
                     DATE(DBR_CL_DATES_1_O), DAY)   
            < 41, 0, 1) as slow_40_dispatch,
            if(DATE_DIFF(DATE(IF(coalesce(LENGTH(RCV_TRAN_DATE),0) < 3, 
                          IF(coalesce(LENGTH(X_ISP_RTNED_DTS),0) < 3, CURRENT_TIMESTAMP(), TIMESTAMP(X_ISP_RTNED_DTS)), 
                          PARSE_TIMESTAMP("%m/%d/%Y %I:%M:%S %p",REPLACE(RCV_TRAN_DATE,"'","")))),
                     DATE(DBR_CL_DATES_1_O), DAY)   
            < 51, 0, 1) as slow_50_dispatch,
            if(DATE_DIFF(DATE(IF(coalesce(LENGTH(RCV_TRAN_DATE),0) < 3, 
                          IF(coalesce(LENGTH(X_ISP_RTNED_DTS),0) < 3, CURRENT_TIMESTAMP(), TIMESTAMP(X_ISP_RTNED_DTS)), 
                          PARSE_TIMESTAMP("%m/%d/%Y %I:%M:%S %p",REPLACE(RCV_TRAN_DATE,"'","")))),
                     DATE(DBR_CL_DATES_1_O), DAY)   
            < 61, 0, 1) as slow_60_dispatch

            FROM `dandsltd-warehouse.dellfloat.F_DISPATCH_HEADER` 
            INNER JOIN `dandsltd-warehouse.dellfloat.F_DISPATCH_SHIP` on DISPATCH_NUM = SVC_DSPCH_ID
            INNER JOIN `dandsltd-warehouse.dellfloat.F_ITEM_MASTER_NEW` on ITM_NBR = ITEM_NUM
            INNER JOIN `dandsltd-warehouse.warehouse.p_invoices` on DISPATCH_NUM = dbr_cli_ref_no and ITM_NBR = DBR_CL_MISC_1
            LEFT JOIN `dandsltd-warehouse.dellfloat.F_DISPATCH_RCV` on SVC_DSPCH_ID = X_ISP_DSPCH_NBR and ITEM_NUM = X_ISP_PART_NBR
            LEFT JOIN `dandsltd-warehouse.dellfloat.F_DATA_LAKE_FILE` ON SVC_DSPCH_ID = dps_number AND DBR_CL_MISC_1 = part_number


            where SHIPD_DT >= '2021-04-01' and SHIPD_DT < '2021-04-31'
            and `dandsltd-warehouse.warehouse.p_invoices`._PARTITIONTIME = (select max(_PARTITIONTIME) from `dandsltd-warehouse.warehouse.p_invoices`)
            and `dandsltd-warehouse.warehouse.p_invoices`.dbr_client in ('DELL68','DELL94','DELLT1')
            """
        else:
            query_string = query_string

        pd.set_option("display.max_columns", None)
        bqclient = bigquery.Client.from_service_account_json('./credentials.json')
        dataframe = bqclient.query(query_string).to_dataframe()
        self.data = pd.DataFrame(dataframe)
        self._dispatch_nums = self.data["DISPATCH_NUM"]

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
            "probability": [],
            "dispatch_nums": []
        }

        # Sanitize data
        if self.data is not None:
            self._sanitize()

        slice_index = 0

        for (index, row), dispatch in zip(self.data.iterrows(), self._dispatch_nums):
            print(dispatch)
            try:
                encoded = self._encoder.transform(self.data[slice_index:slice_index + 1])
                results["classes"].append((self._model.predict(encoded) > 0.5).astype("int32")[0][0])
                results["probability"].append((self._model.predict(encoded))[0][0])
                results["rows"].append(index)
                results["dispatch_nums"].append(dispatch)
            except Exception as e:
                results["rows"].append(index)
                results["classes"].append("ERROR")
                results["probability"].append("ERROR")
                results["dispatch_nums"].append(dispatch)

            # Increment slice index in order to move to the next row of data
            slice_index += 1

        if output_csv is False:
            return results
        else:
            self.data["SLOW_DISPATCH"] = results["classes"]
            self.data["PROBABILITY"] = results["probability"]
            self.data["DISPATCH_NUM"] = results["dispatch_nums"]
            self.data.to_csv("./predictions.csv", index=False)

    def printDataInfo(self):
        print(f"Shape {self.data.shape}")
        print(self.data.head())

    # Implement transformation from query result to dataframe
    # Save the query in this file

    def __str__(self):
        return str(self._model.summary())
