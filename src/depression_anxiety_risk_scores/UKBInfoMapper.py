"""Definition of HumaBiobankMapper Class."""

from __future__ import annotations

from typing import List, Optional

import os
from functools import partial
from google.cloud import bigquery
import pandas as pd  # type: ignore

FIELD_ID_REQUIRED = ["_eid_", "_31_", "_21022_", "_191_", "_53_", "_40000_"]

LIST_INFO_FIELDS = [
    "field_id",
    "field_name",
    "field_title",
    "category_title",
    "Processed_Data",
    "Huma_Data",
]

LIST_INFO_ENCODING = [
    "field_id",
    "field_name",
    "category_title",
    "field_id_processed",
    "field_name_processed",
    "encoding_meaning",
    "encoding_value",
    "Processed_Data",
    "Huma_Data",
]

LIST_INFO_PREPROCESSING = [
    "field_id",
    "field_name",
    "category_title",
    "field_id_processed",
    "encoding_meaning",
    "encoding_value",
    "encoding_id",
    "field_name_processed",
    "value_type",
    "num_participants",
    "Processed_Data",
    "Huma_Data",
    "preprocessing_replace_values_with",
    "preprocessing_steps",
]
FILE = "../../data/ukb_info.ftr"

class UKBInfoMapper:
    """Class definition."""

    def __init__(self: UKBInfoMapper, 
                 field_id_required: List[str] = FIELD_ID_REQUIRED,
                 list_info_fields_level: List[str] = LIST_INFO_FIELDS,
                 list_info_encoding_level: List[str] = LIST_INFO_ENCODING,
                 list_info_preprocessing_level: List[str] = LIST_INFO_PREPROCESSING
                 ) -> None:
        """Initialize class

        Args:
            field_id_required (List[str], optional): field required for modelling later. Defaults to FIELD_ID_REQUIRED.
            list_info_fields_level (List[str], optional): list of info needed when the info level 'field' is selected. Defaults to LIST_INFO_FIELDS.
            list_info_encoding_level (List[str], optional): list of info needed when the info level 'encoding' is selected. Defaults to LIST_INFO_ENCODING.
            list_info_preprocessing_level (List[str], optional): list of info needed when the info level 'preprocessing' is selected. Defaults to LIST_INFO_PREPROCESSING.
        """
        self.client = bigquery.Client()
        self.df = self._load_info_UKB_from_BQ()
        self._field_id_required = field_id_required
        self._list_info_fields_level = list_info_fields_level
        self._list_info_encoding_level = list_info_encoding_level
        self._list_info_preprocessing_level = list_info_preprocessing_level

    def _load_info_UKB_from_BQ(self: UKBInfoMapper) -> pd.DataFrame:
        """Load the UKB info from bigquery table.

        Returns:
            pd.DataFrame: dataframe containing UKB info.
        """

        query = "SELECT * FROM `uk-biobank-data.assessment.field_info` "
        df_info = self.client.query(query).to_dataframe()
        return df_info

    def _get_list_from_column(
        self: UKBInfoMapper,
        filter_column: str,
        col_return: str,
        filter_value: Optional[List[str]] = None,
    ) -> List:
        """Get list of values from the mapper info.

        Args:
            filter_value (List[str)], optional): list of values to return
            filter_column (str): column to look for filter values.
            col_return(str): column to return.

        Returns:
            List: list of values.
        """
        if filter_value:
            df_ = self.df.loc[self.df[filter_column].isin(filter_value), col_return]
        else:
            df_ = self.df[col_return]

        return df_.unique().tolist()

    def _get_info(
        self: UKBInfoMapper,
        filter_value: List[str],
        filter_column: str,
        info_level: str = None,
    ) -> pd.DataFrame:
        """Give all information on specific column for a given value.

        Args:
            filter_value (List[str)]): list of values to return
            filter_column (str): column to return
            info_level (str) : correspond to the degree of description used in the dataframe.Options are 'field','encoding','preprocessing'. Default to 'field'

        Returns:
            pd.DataFrame: dataframe with filtered data.

        """
        if info_level == "field":
            return_columns_list = [filter_column] + self._list_info_fields_level

        elif info_level == "encoding":
            return_columns_list = [filter_column] + self._list_info_encoding_level

        elif info_level == "preprocessing":
            return_columns_list = [filter_column] + self._list_info_preprocessing_level

        else:
            return_columns_list = [filter_column] + self.df.columns.tolist()

        df = self.df.loc[self.df[filter_column].isin(filter_value), return_columns_list]
        df = df.loc[:, ~df.columns.duplicated()]
        return df.drop_duplicates(
            subset=df.columns.difference(["Processed_Data", "Huma_Data"])
        )

    def _write_query_load_data_raw(self: UKBInfoMapper, list_field: List[str]) -> str:
        """Write UKB query to collect preprocessed data from UKB.

        Args:
            list_field (List[str]): list field of data of interest.

        Returns:
            str: query as string.
        """
        query = "hf._eid AS _eid_, "
        for field in list_field:
            query = query + f"hf.{field}, "
        query = (
            "SELECT "
            + query
            + "FROM `uk-biobank-data.assessment.assessment_centre` hf WHERE _31 IS NOT NULL AND _21022 IS NOT NULL AND hf._instance_id = 0"
        )
        return query

    def _write_query_load_data_preprocessed(
        self: UKBInfoMapper, list_field: List[str]
    ) -> str:
        """Write UKB query to collect preprocessed data from UKB.

        Args:
            list_field (List[str]): list field of data of interest.

        Returns:
            str: query as string.
        """

        list_fields_missing = [
            col
            for col in list_field
            if col
            not in self.df[self.df["Processed_Data"] == 1].field_id_processed.tolist()
        ]
        print("Missing Data: {}".format(list_fields_missing))
        list_field = list(set(list_field) - set(list_fields_missing))
        list_fields_processed = [s.replace("-", "_") for s in list_field]
        query = "_eid_, "
        for field_p in list_fields_processed:
            query = query + f"{field_p},"
        query = (
            "SELECT "
            + query
            + "FROM `uk-biobank-data.assessment.assessment_centre_processed`"
        )
        return query

    def _load_data_from_BQ(
        self: UKBInfoMapper,
        filter_value: List[str],
        filter_column: str,
        col_select: Optional[str] = "field_id_processed",
    ) -> pd.DataFrame:
        """Apply the fetching of data to dowload data from BQ.

        Args:
            filter_value (List[str)]): list of values to return
            filter_column (str): column to return
            col_select (str, optional): column in mapper info to use for selecting the data. Defaults to 'field_id_processed'.

        Returns:
            pd.DataFrame: new dataframe with only fetch fields
        """
        df_info = self.df.loc[self.df[filter_column].isin(filter_value)]
        df_info = df_info.drop_duplicates()
        list_fields = df_info[col_select].unique().tolist()

        if col_select == "field_id_processed":
            query = self._write_query_load_data_preprocessed(list_fields)
            df = self.client.query(query).to_dataframe()
        elif col_select == "field_id":
            list_fields = [col[:-1] for col in list_fields]
            query = self._write_query_load_data_raw(list_fields)
            df = self.client.query(query).to_dataframe()
            df.columns = df.columns + "_"
            df = df.rename(columns={"_eid__": "_eid_"})
        return df

    def _apply_filtering_on_data(
        self: UKBInfoMapper,
        df: pd.DataFrame,
        filter_value: List[str],
        filter_column: str,
        col_select: Optional[str] = "field_id_processed",
    ) -> pd.DataFrame:
        """Apply the fetching of data on dataframe containing the data.

        Args:
            df (pd.DataFrame): dataframe with data from UKB
            filter_value (List[str)]): list of values to return
            filter_column (str): column to return
            col_select (str, optional): column in mapper info to use for selecting the data. Defaults to 'field_id_processed'.

        Returns:
            pd.DataFrame: new dataframe with only fetch fields
        """
        df_info = self.df.loc[self.df[filter_column].isin(filter_value)]
        df_info = df_info.drop_duplicates()
        list_fields = self._field_id_required + df_info[col_select].unique().tolist()
        list_fields = [col for col in list_fields if col in df.columns.tolist()]
        df = df[list_fields].copy()
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def _apply_dropping_on_data(
        self: UKBInfoMapper,
        df: pd.DataFrame,
        filter_value: List[str],
        filter_column: str,
        col_select: Optional[str] = "field_id_processed",
        verbose: Optional[bool] = True,
    ) -> pd.DataFrame:
        """Apply the fetching of data on dataframe containing the data.

        Args:
            df (pd.DataFrame): dataframe with data from UKB
            filter_value (List[str)]): list of values to return
            filter_column (str): column to return
            col_select (str, optional): column in mapper info to use for selecting the data. Defaults to 'field_id_processed'.
            verbose (bool, optional): print info or not. Defaults to True.

        Returns:
            pd.DataFrame: new dataframe with only fetch fields
        """
        df_info = self.df.loc[self.df[filter_column].isin(filter_value)]
        df_info = df_info.drop_duplicates()
        df_info = df_info[~df_info.field_id_processed.isin(self._field_id_required)]
        list_col_drop = df_info[col_select].unique().tolist()
        list_col_drop = [col for col in list_col_drop if col in df.columns.to_list()]
        if verbose:
            print(
                "[FILTERING FEATURES] Drop {} features from unwanted features".format(
                    len(list_col_drop)
                )
            )
        return df.drop(columns=list_col_drop)

    def _rename_columns_on_data(
        self: UKBInfoMapper,
        df: pd.DataFrame,
        original_col: Optional[str] = "field_id_processed",
        new_col: Optional[str] = "field_name_processed",
    ) -> pd.DataFrame:
        """Rename columns by creating a dictionary from columns in mapper dataframe.

        Args:
            df (pd.DataFrame): dataframe with data from UKB
            original_col (str, optional): original name corresponding to values from a column of mapper info. Defaults to 'field_id_processed'.
            new_col (str, optional): new name corresponding to values from a column of mapper info. Defaults to 'field_name_processed'.

        Returns:
            pd.DataFrame: new dataframe with renamed columns.
        """
        mapping_dict = dict(zip(self.df[original_col], self.df[new_col]))
        return df.rename(columns=mapping_dict)

    def _add_preprocessing_pipeline_to_field(
        self: UKBInfoMapper, col_field: str, preprocessing_pipeline: str
    ):
        """Add preprocessing info on dataframe. (see jupyter notebook to see how to apply)

        Args:
            col_field (str):field where we want to specficy the preprocessing steps.
            preprocessing_pipeline (str): preprocessing pipeline
        """
        preprocessing_pipeline = preprocessing_pipeline.replace(" ", "")
        preprocessing_pipeline = preprocessing_pipeline.replace("\n", "")
        index_list = self.df[self.df.field_id == col_field].index.to_numpy()
        for index in index_list:
            self.df.at[index, "preprocessing_steps"] = preprocessing_pipeline

    def _preprocessing_pipeline(
        self: UKBInfoMapper,
        df: pd.DataFrame,
        field_id: str,
        preprocessing_steps: Optional[str] = None,
    ) -> pd.DataFrame:
        """Apply the preprocessing defined in the mapper dataframe.

        Args:
            df (pd.DataFrame): dataframe containing the data to preprocessed.
            field_id (str): field to preprocess.
            preprocessing_steps (Optional[str]): string for preprocessing. Default is None.

        Returns:
            pd.DataFrame: dataframe with field preprocessed.
        """
        import utils_preprocessing_series
        if not preprocessing_steps:
            preprocessing_steps = self.df.loc[
                self.df.field_id == field_id, "preprocessing_steps"
            ].unique()[0]
        preprocessing_steps = preprocessing_steps.replace("FIELD_ID", field_id)
        mapper_info = self
        for f in eval(preprocessing_steps):
            df = f(df)
        # To fit with mapper
        df.columns = [col.split(".")[0] for col in df.columns]
        df.columns = df.columns.str.replace("-", "_")
        return df

    def _get_advice_on_preprocessing(self: UKBInfoMapper, field_id: str):
        preprocessing_steps = self.df[
            self.df.field_id == field_id
        ].preprocessing_steps.unique()
        if preprocessing_steps[0]:
            print(
                "Found {} existing preprocessing pipeline".format(
                    len(preprocessing_steps)
                )
            )
        else:
            val_col = self.df[self.df.field_id == field_id].value_type.unique()[0]
            encoding_col = (
                self.df[self.df.field_id == field_id].encoding_id == 0
            ).unique()[0]
            data_type_col = self.df[self.df.field_id == field_id].data_type.unique()[0]
            preprocessing_steps = (
                self.df[
                    (self.df.value_type == val_col)
                    & ((self.df.encoding_id == 0) == encoding_col)
                    & (self.df.data_type == data_type_col)
                    & (~self.df.preprocessing_steps.isnull())
                ]
                .preprocessing_steps.unique()
                .tolist()
            )
            if preprocessing_steps:
                print(
                    "Found {} similar preprocessing pipeline".format(
                        len(preprocessing_steps)
                    )
                )
            else:
                print("No similar preprocessing pipeline found")
                print(field_id, val_col, encoding_col, data_type_col)
                print(
                    "[FIELD_ID: {}] Data Description: VALUE_TYPE: {}, ENCODING:{}, DATA_TYPE: {}".format(
                        field_id, val_col, encoding_col, data_type_col
                    )
                )
                preprocessing_steps = None
        return preprocessing_steps

    def _upload_mapper_dataframe_on_BQ(
        self: UKBInfoMapper,
        table_id: str = "uk-biobank-data.assessment.assessment_centre.field_info",
    ):
        """Modify the mapper dataframe on the bigquery space of Huma.

        Args:
            table_id (str, optional): name of the table to update the dataframe. Defaults to "hu-ai-sandbox-project.bastien_sandbox.ukb_info".
        """
        print("Writing BQ table with mapper_info updated dataframe on huma cloud ...")
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = self.client.load_table_from_dataframe(
            self.df, table_id, job_config=job_config
        )
        job.result()
        print("Updated mapper_info dataframe")
