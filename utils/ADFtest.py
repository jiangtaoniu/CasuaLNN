import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import ADF
from typing import List

"""
This script provides utility functions for performing the Augmented Dickey-Fuller (ADF)
test on time series datasets to assess stationarity.
"""

def calculate_ADF(root_path: str, data_path: str) -> np.ndarray:
    """
    Calculates the ADF test result for all feature columns in a given CSV dataset
    using the `statsmodels` library.

    Args:
        root_path (str): The root directory of the dataset.
        data_path (str): The path to the CSV data file.

    Returns:
        np.ndarray: An array where each row contains the full ADF test result
                    tuple for a column.
    """
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    cols = [col for col in df_raw.columns if col != 'date']
    df_raw = df_raw[cols]

    adf_results = []
    for col_name in cols:
        series = df_raw[col_name].dropna() # Drop NaN values for ADF test
        adf_result = adfuller(series, maxlag=1)
        # print(f"ADF result for {col_name}: {adf_result}")
        adf_results.append(adf_result)

    return np.array(adf_results)


def calculate_target_ADF(root_path: str, data_path: str, target: str = 'OT') -> np.ndarray:
    """
    Calculates the ADF test result for specific target columns in a CSV dataset
    using the `statsmodels` library.

    Args:
        root_path (str): The root directory of the dataset.
        data_path (str): The path to the CSV data file.
        target (str, optional): A comma-separated string of target column names.
                                Defaults to 'OT'.

    Returns:
        np.ndarray: An array where each row contains the full ADF test result
                    tuple for a target column.
    """
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    target_cols = target.split(',')
    df_raw = df_raw[target_cols]

    adf_results = []
    for col_name in target_cols:
        series = df_raw[col_name].dropna() # Drop NaN values for ADF test
        adf_result = adfuller(series, maxlag=1)
        adf_results.append(adf_result)
        
    return np.array(adf_results)


def archADF(root_path: str, data_path: str) -> float:
    """
    Calculates the average ADF test statistic across all feature columns in a
    CSV dataset using the `arch` library implementation.

    Args:
        root_path (str): The root directory of the dataset.
        data_path (str): The path to the CSV data file.

    Returns:
        float: The average ADF test statistic across all columns.
    """
    df = pd.read_csv(os.path.join(root_path, data_path))
    cols = [col for col in df.columns if col != 'date']
    
    total_adf_stat = 0
    for target_col in cols:
        series = df[target_col].dropna().values
        adf_test = ADF(series)
        total_adf_stat += adf_test.stat

    return total_adf_stat / len(cols) if len(cols) > 0 else 0.0


if __name__ == '__main__':
    """
    Demonstration block to run the `archADF` function on various datasets
    and print the average ADF statistic.
    """

    # --- Expected results for various datasets ---
    # * Exchange - result: -1.902 | report: -1.889
    ADFmetric = archADF(root_path="./dataset/exchange_rate/", data_path="exchange.csv")
    print("Exchange ADF metric:", ADFmetric)

    # * Illness - result: -5.334 | report: -5.406
    ADFmetric = archADF(root_path="./dataset/illness/", data_path="national_illness.csv")
    print("Illness ADF metric:", ADFmetric)

    # * ETTm2 - result: -5.663 | report: -6.225
    ADFmetric = archADF(root_path="./dataset/ETT-small/", data_path="ETTm2.csv")
    print("ETTm2 ADF metric:", ADFmetric)

    # * Electricity - result: -8.444 | report: -8.483
    ADFmetric = archADF(root_path="./dataset/electricity/", data_path="electricity.csv")
    print("Electricity ADF metric:", ADFmetric)

    # * Traffic - result: -15.020 | report: -15.046
    ADFmetric = archADF(root_path="./dataset/traffic/", data_path="traffic.csv")
    print("Traffic ADF metric:", ADFmetric)

    # * Weather - result: -26.681 | report: -26.661
    ADFmetric = archADF(root_path="./dataset/weather/", data_path="weather.csv")
    print("Weather ADF metric:", ADFmetric)
