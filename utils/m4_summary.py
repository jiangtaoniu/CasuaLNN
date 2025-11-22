# This source code is adapted from the N-BEATS model implementation.
# The original license and copyright information are reproduced below for attribution.
#
# Original Source: Oreshkin et al., N-BEATS: Neural basis expansion analysis for
#                  interpretable time series forecasting, https://arxiv.org/abs/1905.10437
#
# Copyright 2020 Element AI Inc. All Rights Reserved.
# Licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.

"""
M4 Summary Evaluation

This script provides a class `M4Summary` to evaluate forecasting model performance
on the M4 competition dataset, adhering to the official M4 evaluation rules.
It computes standard M4 metrics like sMAPE, MASE, and OWA (Overall Weighted Average)
by comparing model forecasts against the ground truth and the Naive2 benchmark.
"""
from collections import OrderedDict

import numpy as np
import pandas as pd

from data_provider.m4 import M4Dataset
from data_provider.m4 import M4Meta
import os


def group_values(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    """
    Helper function to filter an array of time series data based on a specified group name.

    Args:
        values (np.ndarray): The array of time series data to filter.
        groups (np.ndarray): An array of group names corresponding to the time series.
        group_name (str): The name of the group to filter by (e.g., 'Yearly', 'Monthly').

    Returns:
        np.ndarray: A filtered array containing only the time series belonging to the specified group.
    """
    return np.array([v[~np.isnan(v)] for v in values[groups == group_name]], dtype=object)


def mase(forecast: np.ndarray, insample: np.ndarray, outsample: np.ndarray, frequency: int) -> float:
    """
    Calculates the Mean Absolute Scaled Error (MASE).

    MASE is a scale-free error metric that compares the forecast's mean absolute error
    to the mean absolute error of a naive seasonal forecast on the in-sample data.

    Args:
        forecast (np.ndarray): The forecasted values.
        insample (np.ndarray): The in-sample (training) time series data.
        outsample (np.ndarray): The out-sample (ground truth) time series data.
        frequency (int): The seasonal frequency of the time series.

    Returns:
        float: The MASE value.
    """
    # Denominator is the mean absolute error of a naive seasonal forecast
    scaling_factor = np.mean(np.abs(insample[:-frequency] - insample[frequency:]))
    # Numerator is the mean absolute error of the forecast
    mae = np.mean(np.abs(forecast - outsample))
    return mae / scaling_factor


def smape_2(forecast: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Calculates the Symmetric Mean Absolute Percentage Error (sMAPE) for evaluation.

    Formula: 200 * abs(forecast - target) / (abs(target) + abs(forecast))

    Args:
        forecast (np.ndarray): The forecasted values.
        target (np.ndarray): The ground truth values.

    Returns:
        np.ndarray: An array of sMAPE values for each time step.
    """
    denom = np.abs(target) + np.abs(forecast)
    # Handle case where both forecast and target are zero to avoid division by zero
    denom[denom == 0.0] = 1.0 # The numerator will be 0 in this case, so the result is 0
    return 200 * np.abs(forecast - target) / denom


def mape(forecast: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Calculates the Mean Absolute Percentage Error (MAPE) for evaluation.

    Formula: 100 * abs(forecast - target) / abs(target)

    Args:
        forecast (np.ndarray): The forecasted values.
        target (np.ndarray): The ground truth values.

    Returns:
        np.ndarray: An array of MAPE values for each time step.
    """
    denom = np.abs(target)
    # Handle case where target is zero to avoid division by zero
    denom[denom == 0.0] = 1.0 # The numerator will be 0 in this case, so the result is 0
    return 100 * np.abs(forecast - target) / denom


class M4Summary:
    """
    A class to evaluate forecasts on the M4 dataset according to official competition rules.

    This class loads model forecasts and the Naive2 benchmark, computes sMAPE and MASE
    for each seasonal pattern, and then calculates the Overall Weighted Average (OWA).
    """

    def __init__(self, file_path: str, root_path: str):
        """
        Initializes the M4Summary evaluator.

        Args:
            file_path (str): The directory path where model forecast CSV files are stored.
            root_path (str): The root directory of the M4 dataset files (Train and Test).
        """
        self.file_path = file_path
        self.training_set = M4Dataset.load(training=True, dataset_file=root_path)
        self.test_set = M4Dataset.load(training=False, dataset_file=root_path)
        self.naive_path = os.path.join(root_path, 'submission-Naive2.csv') # Path to the Naive2 benchmark forecasts

    def evaluate(self) -> tuple:
        """
        Evaluates the model's forecasts against the M4 test dataset and Naive2 benchmark.

        This method iterates through all M4 seasonal patterns, loads the corresponding
        forecasts, calculates sMAPE and MASE for both the model and Naive2, and then
        computes the final OWA score.

        Returns:
            tuple: A tuple containing four dictionaries with rounded scores:
                   (grouped_smapes, grouped_owa, grouped_mapes, grouped_model_mases)
        """
        grouped_owa = OrderedDict()

        # Load Naive2 benchmark forecasts, which are used for scaling in OWA calculation
        naive2_forecasts = pd.read_csv(self.naive_path).values[:, 1:].astype(np.float32)
        naive2_forecasts = np.array([v[~np.isnan(v)] for v in naive2_forecasts], dtype=object)

        # Dictionaries to store metrics for each group
        model_mases, naive2_smapes, naive2_mases, grouped_smapes, grouped_mapes = {}, {}, {}, {}, {}

        # Iterate through each seasonal pattern defined in M4Meta
        for group_name in M4Meta.seasonal_patterns:
            # Construct file path for the model's forecast for the current group
            file_name = os.path.join(self.file_path, f"{group_name}_forecast.csv")
            if os.path.exists(file_name):
                model_forecast = pd.read_csv(file_name).values[:, 1:] # Exclude ID column
            else:
                raise FileNotFoundError(f"Forecast file not found for group '{group_name}': {file_name}")

            # Filter Naive2 forecasts, ground truth (target), and in-sample data for the current group
            naive2_forecast = group_values(naive2_forecasts, self.test_set.groups, group_name)
            target = group_values(self.test_set.values, self.test_set.groups, group_name)
            insample = group_values(self.training_set.values, self.test_set.groups, group_name)
            # All time series within a group share the same frequency
            frequency = self.training_set.frequencies[self.test_set.groups == group_name][0]

            # Calculate and store metrics for the current group
            # MASE for the model's forecast
            model_mases[group_name] = np.mean([mase(forecast=model_forecast[i],
                                                    insample=insample[i],
                                                    outsample=target[i],
                                                    frequency=frequency) for i in range(len(model_forecast))])
            # MASE for the Naive2 benchmark
            naive2_mases[group_name] = np.mean([mase(forecast=naive2_forecast[i],
                                                     insample=insample[i],
                                                     outsample=target[i],
                                                     frequency=frequency) for i in range(len(model_forecast))])
            # sMAPE for Naive2 and the model
            naive2_smapes[group_name] = np.mean(smape_2(naive2_forecast, target))
            grouped_smapes[group_name] = np.mean(smape_2(forecast=model_forecast, target=target))
            # MAPE for the model
            grouped_mapes[group_name] = np.mean(mape(forecast=model_forecast, target=target))

        # Summarize and aggregate scores according to M4 competition rules
        grouped_smapes = self.summarize_groups(grouped_smapes)
        grouped_mapes = self.summarize_groups(grouped_mapes)
        grouped_model_mases = self.summarize_groups(model_mases)
        grouped_naive2_smapes = self.summarize_groups(naive2_smapes)
        grouped_naive2_mases = self.summarize_groups(naive2_mases)
        
        # Calculate Overall Weighted Average (OWA) for each group
        for k in grouped_model_mases.keys():
            # OWA = ( (Model_MASE / Naive2_MASE) + (Model_sMAPE / Naive2_sMAPE) ) / 2
            grouped_owa[k] = (grouped_model_mases[k] / grouped_naive2_mases[k] +
                              grouped_smapes[k] / grouped_naive2_smapes[k]) / 2

        # Helper function to round all dictionary values for clean output
        def round_all(d):
            return {k: np.round(v, 3) for k, v in d.items()}

        return round_all(grouped_smapes), round_all(grouped_owa), round_all(grouped_mapes), round_all(grouped_model_mases)

    def summarize_groups(self, scores: dict) -> OrderedDict:
        """
        Aggregates scores per seasonal pattern according to official M4 rules.

        Specifically, it groups 'Weekly', 'Daily', and 'Hourly' into an 'Others' category
        and calculates a final weighted 'Average'.

        Args:
            scores (dict): A dictionary of scores with keys as seasonal pattern names.

        Returns:
            OrderedDict: An ordered dictionary with aggregated scores.
        """
        scores_summary = OrderedDict()

        def group_count(group_name):
            return len(np.where(self.test_set.groups == group_name)[0])

        # Calculate weighted scores for Yearly, Quarterly, Monthly
        weighted_score = {}
        for g in ['Yearly', 'Quarterly', 'Monthly']:
            weighted_score[g] = scores[g] * group_count(g)
            scores_summary[g] = scores[g]

        # Calculate combined score for the 'Others' group
        others_score = 0
        others_count = 0
        for g in ['Weekly', 'Daily', 'Hourly']:
            others_score += scores[g] * group_count(g)
            others_count += group_count(g)
        weighted_score['Others'] = others_score
        scores_summary['Others'] = others_score / others_count if others_count > 0 else 0

        # Calculate final weighted average across all groups
        average = np.sum(list(weighted_score.values())) / len(self.test_set.groups)
        scores_summary['Average'] = average

        return scores_summary