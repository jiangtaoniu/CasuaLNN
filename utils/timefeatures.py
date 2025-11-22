# This file is adapted from gluonts/src/gluonts/time_feature/_base.py.
# The original license information is reproduced below.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """
    Abstract base class for time feature extractors.
    Subclasses should implement the __call__ method to extract a specific
    time-based numerical feature from a pandas DatetimeIndex.
    """
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Extracts a numerical time feature from the given DatetimeIndex.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def __repr__(self):
        """
        Returns a string representation of the time feature class.
        """
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """
    Extracts the second of the minute, normalized to a value between -0.5 and 0.5.
    (e.g., 0-59 seconds -> [0, 1] -> [-0.5, 0.5])
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """
    Extracts the minute of the hour, normalized to a value between -0.5 and 0.5.
    (e.g., 0-59 minutes -> [0, 1] -> [-0.5, 0.5])
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """
    Extracts the hour of the day, normalized to a value between -0.5 and 0.5.
    (e.g., 0-23 hours -> [0, 1] -> [-0.5, 0.5])
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """
    Extracts the day of the week, normalized to a value between -0.5 and 0.5.
    (e.g., Mon=0, Sun=6 -> [0, 1] -> [-0.5, 0.5])
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """
    Extracts the day of the month, normalized to a value between -0.5 and 0.5.
    (e.g., 1-31 days -> [0, 1] -> [-0.5, 0.5])
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """
    Extracts the day of the year, normalized to a value between -0.5 and 0.5.
    (e.g., 1-366 days -> [0, 1] -> [-0.5, 0.5])
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """
    Extracts the month of the year, normalized to a value between -0.5 and 0.5.
    (e.g., Jan=1, Dec=12 -> [0, 1] -> [-0.5, 0.5])
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """
    Extracts the week of the year (ISO week), normalized to a value between -0.5 and 0.5.
    (e.g., 1-52/53 weeks -> [0, 1] -> [-0.5, 0.5])
    """
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # For week of year, index.isocalendar().week returns 1 to 52/53.
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time feature extractors appropriate for the given frequency string.

    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
        (e.g., 's' for second, 't'/'min' for minute, 'h' for hour, 'd' for day, 'w' for week, 'm' for month, 'y' for year).

    Returns
    -------
    List[TimeFeature]
        A list of instantiated TimeFeature objects.
    
    Raises
    ------
    RuntimeError
        If the provided frequency string is not supported.
    """
    # Mapping of pandas offset types to relevant TimeFeature classes.
    # Higher frequency granularities include features from lower granularities.
    features_by_offsets = {
        offsets.YearEnd: [MonthOfYear, DayOfMonth, DayOfYear, HourOfDay, MinuteOfHour, SecondOfMinute],
        offsets.QuarterEnd: [MonthOfYear, DayOfMonth, DayOfYear, HourOfDay, MinuteOfHour, SecondOfMinute],
        offsets.MonthEnd: [MonthOfYear, DayOfMonth, DayOfYear, HourOfDay, MinuteOfHour, SecondOfMinute],
        offsets.Week: [DayOfWeek, DayOfMonth, WeekOfYear, DayOfYear, HourOfDay, MinuteOfHour, SecondOfMinute],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear, HourOfDay, MinuteOfHour, SecondOfMinute],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear, HourOfDay, MinuteOfHour, SecondOfMinute], # Similar to Day but for business days
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear, MinuteOfHour, SecondOfMinute],
        offsets.Minute: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear, SecondOfMinute],
        offsets.Second: [SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
    }

    offset = to_offset(freq_str)

    # Find the most granular offset that matches the frequency string
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    # If no matching offset type is found, raise an error
    supported_freq_msg = f"""
    Unsupported frequency: {freq_str}.
    Supported granularities include:
        Y   - Yearly (alias: A)
        M   - Monthly
        W   - Weekly
        D   - Daily
        B   - Business days
        H   - Hourly
        T   - Minutely (alias: min)
        S   - Secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates: pd.DatetimeIndex, freq: str = 'h') -> np.ndarray:
    """
    Generates a NumPy array of numerical time features for a given DatetimeIndex.

    Parameters
    ----------
    dates (pd.DatetimeIndex):
        The DatetimeIndex for which to generate time features.
    freq (str, optional):
        The frequency string of the time series (e.g., 'h', 't', 'min').
        This is used to select the appropriate set of time features. Defaults to 'h'.

    Returns
    -------
    np.ndarray
        A 2D NumPy array where each row is a time feature and each column
        corresponds to a timestamp in the input `dates`.
    """
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])