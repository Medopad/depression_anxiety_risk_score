"""This is a module for the preprocessing of UKBiobank data."""
import ast
from typing import List

import numpy as np
import pandas as pd

from UKBInfoMapper import UKBInfoMapper

def remove_float_number_field(row: pd.Series) -> pd.Series:
    """Convert float number as string to integer as string.

    Args:
        row (pd.Series): row column of dataframe.

    Returns:
        pd.Series: row column with processed string.
    """
    if row:
        return row.split(".")[0]
    else:
        return row


def preprocess_string_ukb(row: str) -> str:
    """Preprocess string tro remove space and capitalize first letter on each word.

    Args:
        row (str): row of the dataframe to process.

    Returns:
        str: processed row without space and concatenation of words.
    """
    token_string = row.split(" ")
    capitalized = [t.title() for t in token_string]
    return "".join(capitalized)


def calculate_list_mean(row: str, ignore_values: List[str] = []) -> float:
    """Calculate mean of a list returned by BQ.

    Args:
        row (str): array formated as string.
        ignore_values (List[str]): ignore defined values found
                                in the array to average . Defaults to [].

    Returns:
        float: calculated mean from array formatted as string.
    """

    if row and len(row) > 1:
        row = ast.literal_eval(row)
        row = [
            x
            for i, x in enumerate(row)
            if (x not in ignore_values and not isinstance(x, str))
        ]
        return np.nanmean(row)
    else:
        return np.nan


def compute_average_over_list_data(
    df: pd.DataFrame,
    mapper_info: UKBInfoMapper,
    field_id: str,
    drop_encoded_values: bool = False,
) -> pd.DataFrame:
    """Calculate mean values for defined list of columns with array formatted as string.

    Args:
        df (pd.DataFrame): dataframe containing data to process.
        list_columns (List[str]): list of columns of the dataframe
            to apply the function on.
        drop_encoded_values (bool): ignore defined values found
                                in the mapper as encoding values. Defaults to True.

    Returns:
        pd.DataFrame: dataframe with processed columns.
    """

    ignore_values = []
    if drop_encoded_values:
        ignore_values = (
            mapper_info.df[mapper_info.df.field_id == field_id]
            .encoding_value.astype(float)
            .tolist()
        )
    df[field_id] = df[field_id].apply(calculate_list_mean, ignore_values)
    return df


def extract_column_values(row: pd.Series) -> pd.Series:
    """Extract values from arrays formatted as string.

    Args:
        row (pd.Series): row of dataframe with array formatted as string.

    Returns:
        pd.Series: array data extracted from string.
    """
    if type(row) == float or row is None:
        return np.nan
    else:
        return ast.literal_eval(row)


def replace_values_integer(
    df: pd.DataFrame, mapper_info: UKBInfoMapper, field_id: str
) -> pd.DataFrame:
    """Replace values from UKBMApper info table when encoded values on a given field.

    Args:
        df (pd.DataFrame): dataframe containing data to process
        mapper_info (UKBInfoMapper): mapper info table containing encoding values and replacement
        field_id (str): field to apply the function.

    Returns:
        pd.DataFrame: dataframe with replaced values for the defined field_id.
    """
    df_info = mapper_info.df[mapper_info.df.field_id == field_id].copy()
    df_info = df_info.dropna(subset="encoding_value")
    df_info = df_info[~df_info["encoding_value"].isnull()]
    encoding_values = (
        df_info[df_info.field_id == field_id].encoding_value.astype(float).to_numpy()
    )
    replace_values = df_info[
        df_info.field_id == field_id
    ].preprocessing_replace_values_with.to_numpy()
    dict_values = dict(zip(encoding_values, replace_values))
    df[field_id].replace(dict_values, inplace=True)
    if 21022 in dict_values.values():
        df[field_id] = np.where(df[field_id] == 21022, df["_21022_"], df[field_id])
    return df


def explode_str_columns(df: pd.DataFrame, col_field: str) -> pd.DataFrame:
    """Explode the columns with arrays to remove arrays.

    Args:
        df (pd.DataFrame):  dataframe containing data to process
        col_field (str): column to explode.

    Returns:
        pd.DataFrame: dataframe with exploded columns.
    """
    list_field = [col_field, "_eid_"]
    df_ = df[list_field].copy()
    df_.loc[:, col_field] = df_[col_field].apply(lambda x: extract_column_values(x))
    df_ = df_.explode(col_field)
    return df_


def one_hot_encoder_list(
    df: pd.DataFrame, agg_function: str = "sum", prefix_str: str = ""
) -> pd.DataFrame:
    """Create dummy variables.

    Args:
        df (pd.DataFrame):  dataframe containing data to process
        agg_function (str): aggregate function to use when grouping the data per _eid_. Default to 'sum'.

    Returns:
        pd.DataFrame: dataframe with the exploded columns.
    """
    df_dummy = []
    for col in df.columns:
        if col == "_eid_":
            df_dummy.append(df["_eid_"])
        else:
            df_dummy = [pd.get_dummies(df[col], prefix=col, prefix_sep=prefix_str)]
    df_dummy = pd.concat(df_dummy, axis=1)
    df_dummy = df_dummy.groupby("_eid_").agg(agg_function)
    return df_dummy


def process_multi_categorical_type_data(
    df: pd.DataFrame, field_id: str, prefix_str: str = ""
) -> pd.DataFrame:
    """Process categorical data with arrays in rows.

    Args:
        df (pd.DataFrame): dataframe containing data to process
        list_field (List[str]): list field defined as categorical to process.
        prefix_str(str, optionsl): string to add before encoding for naming the field.

    Returns:
        pd.DataFrame: dataframe with processed categorical fields.
    """
    df_ = explode_str_columns(df, field_id)
    df_ = one_hot_encoder_list(df_, "sum", prefix_str)
    df = df.drop(columns=field_id)
    df = df.merge(df_, how="left", on="_eid_")
    return df


def process_categorical_type_data(
    df: pd.DataFrame, field_id: str, prefix_str: str = ""
) -> pd.DataFrame:
    """Process categorical data by creating dummy variables.

    Args:
        df (pd.DataFrame): dataframe containing data to process
        list_categorical_field (List[str]): list field defined as categorical to process.
        prefix_str(str, optionsl): string to add before encoding for naming the field.

    Returns:
       pd.DataFrame: dataframe with processed categorical fields.
    """
    df = pd.get_dummies(df, columns=[field_id], prefix=field_id, prefix_sep=prefix_str)
    return df


def process_value_type_data(
    df: pd.DataFrame,
    field_id: str,
) -> pd.DataFrame:
    """Process integer or float data by replacing NaN values with  mean of columns.

    Args:
        df (pd.DataFrame): dataframe containing data to process
        list_integer_field (List[str]): list field defined as integer/float to process.

    Returns:
        pd.DataFrame: dataframe with processed float or integer fields.
    """
    df.loc[:, field_id] = df[field_id].fillna(df[field_id].mean())
    return df


def process_date_type_data(
    df: pd.DataFrame,
    field_id: str,
) -> pd.DataFrame:
    """Process date data.

    Args:
        df (pd.DataFrame):  dataframe containing data to process
        list_date_field (List[str]):  list field defined as date to process.

    Returns:
        pd.DataFrame: dataframe with processed date fields.
    """
    df[field_id] = pd.to_datetime(df[field_id], format="%Y-%m-%d")
    return df


def transform_datetime_to_bool_prior_assessment(
    df: pd.DataFrame,
    field_id: str,
) -> pd.DataFrame:
    """Change data to binary by looking if happening before DateOfAssessment.

    Args:
        df (pd.DataFrame):  dataframe containing data to process
        list_date_field (List[str]):  list field defined as date to process.

    Returns:
        pd.DataFrame: dataframe with  date fields as binary.
    """
    df.loc[:, field_id] = np.where(df[field_id] < df["_53_"], 1, 0)
    return df
