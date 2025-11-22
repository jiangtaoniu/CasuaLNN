import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.utils.load_data import load_from_tsfile_to_dataframe

import warnings
from utils.augmentation import run_augmentation_single

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    '''
    Dataset class for the ETT (Electricity Transformer Temperature) hourly dataset.

    This dataset handles loading, preprocessing (scaling, time features),
    and splitting of ETT hourly data for time series forecasting tasks.
    '''

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        '''
        Initializes the ETT hourly dataset.

        Args:
            args: Configuration arguments (e.g., for data augmentation).
            root_path (str): Root directory of the dataset.
            flag (str): Data split type ('train', 'val', 'test').
            size (list, optional): [seq_len, label_len, pred_len] defining input/output lengths.
            features (str): Feature type ('M': multivariate, 'S': univariate, 'MS': multivariate to univariate).
            data_path (str): Name of the data CSV file.
            target (str): Name of the target column for univariate forecasting.
            scale (bool): Whether to apply StandardScaler.
            timeenc (int): Time feature encoding type (0: categorical, 1: numerical).
            freq (str): Frequency of data ('h' for hourly).
            seasonal_patterns: Not used for ETT.
        '''
        self.args = args # Store args for augmentation
        # Define sequence lengths
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # Initialize
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        '''
        Reads the data from the CSV file, performs scaling, and extracts time features.
        '''
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Define data split borders for ETT datasets (fixed sizes)
        # Train: first 12 months, Validation: next 4 months, Test: last 4 months
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select data columns based on feature type
        if self.features == 'M' or self.features == 'MS': # Multivariate or Multivariate-to-Univariate
            cols_data = df_raw.columns[1:] # Exclude 'date' column
            df_data = df_raw[cols_data]
        elif self.features == 'S': # Univariate
            df_data = df_raw[[self.target]]

        # Apply StandardScaler to the data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # Fit scaler only on training data
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0: # Categorical time features
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1: # Numerical (Fourier) time features
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) # (num_features, num_timestamps) -> (num_timestamps, num_features)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2] # For forecasting, target is often the same as input
        self.data_stamp = data_stamp

        # Apply data augmentation if specified (only for training set)
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _ = run_augmentation_single(self.data_x, self.data_y, self.args)

    def __getitem__(self, index):
        '''
        Returns one sample of data.
        '''
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        '''
        Returns the total number of samples available for a sliding window approach.
        '''
        # (Total length of data) - (input sequence length) - (prediction length) + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        '''
        Inverse transforms scaled data back to its original scale.
        '''
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    '''
    Dataset class for the ETT (Electricity Transformer Temperature) minute dataset.

    Similar to Dataset_ETT_hour, but specifically configured for minute-level data.
    '''

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        '''
        Initializes the ETT minute dataset.

        Args:
            args: Configuration arguments (e.g., for data augmentation).
            root_path (str): Root directory of the dataset.
            flag (str): Data split type ('train', 'val', 'test').
            size (list, optional): [seq_len, label_len, pred_len] defining input/output lengths.
            features (str): Feature type ('M': multivariate, 'S': univariate, 'MS': multivariate to univariate).
            data_path (str): Name of the data CSV file.
            target (str): Name of the target column for univariate forecasting.
            scale (bool): Whether to apply StandardScaler.
            timeenc (int): Time feature encoding type (0: categorical, 1: numerical).
            freq (str): Frequency of data ('t' for minute).
            seasonal_patterns: Not used for ETT.
        '''
        self.args = args # Store args for augmentation
        # Define sequence lengths
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Initialize
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        '''
        Reads the data from the CSV file, performs scaling, and extracts time features.
        '''
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Define data split borders for ETT datasets (fixed sizes)
        # Note: Minute data is much denser, so the borders are multiplied by 4 (for 15-minute intervals)
        # Train: first 12 months, Validation: next 4 months, Test: last 4 months
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select data columns based on feature type
        if self.features == 'M' or self.features == 'MS': # Multivariate or Multivariate-to-Univariate
            cols_data = df_raw.columns[1:] # Exclude 'date' column
            df_data = df_raw[cols_data]
        elif self.features == 'S': # Univariate
            df_data = df_raw[[self.target]]

        # Apply StandardScaler to the data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # Fit scaler only on training data
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0: # Categorical time features
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            # Aggregate minute data into 15-minute intervals
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1: # Numerical (Fourier) time features
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) # (num_features, num_timestamps) -> (num_timestamps, num_features)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2] # For forecasting, target is often the same as input
        self.data_stamp = data_stamp

        # Apply data augmentation if specified (only for training set)
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _ = run_augmentation_single(self.data_x, self.data_y, self.args)

    def __getitem__(self, index):
        '''
        Returns one sample of data.
        '''
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        '''
        Returns the total number of samples available for a sliding window approach.
        '''
        # (Total length of data) - (input sequence length) - (prediction length) + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        '''
        Inverse transforms scaled data back to its original scale.
        '''
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    '''
    Generic Dataset class for custom time series data.

    This class handles loading, preprocessing, and splitting of custom CSV data
    for time series forecasting tasks. Data split ratios are dynamic (70/10/20).
    '''

    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='custom.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        '''
        Initializes the custom dataset.

        Args:
            args: Configuration arguments (e.g., for data augmentation).
            root_path (str): Root directory of the dataset.
            flag (str): Data split type ('train', 'val', 'test').
            size (list, optional): [seq_len, label_len, pred_len] defining input/output lengths.
            features (str): Feature type ('M': multivariate, 'S': univariate, 'MS': multivariate to univariate).
            data_path (str): Name of the data CSV file.
            target (str): Name of the target column for univariate forecasting.
            scale (bool): Whether to apply StandardScaler.
            timeenc (int): Time feature encoding type (0: categorical, 1: numerical).
            freq (str): Frequency of data ('h' for hourly).
            seasonal_patterns: Not used for custom datasets.
        '''
        self.args = args # Store args for augmentation
        # Define sequence lengths
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Initialize
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        '''
        Reads the data from the CSV file, performs scaling, extracts time features,
        and dynamically splits data into train/validation/test sets.
        '''
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Reorder columns: 'date', other features, then target feature
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw(['date'] + cols + [self.target])

        # Dynamic data splitting (70% train, 20% test, 10% validation)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select data columns based on feature type
        if self.features == 'M' or self.features == 'MS': # Multivariate or Multivariate-to-Univariate
            cols_data = df_raw.columns[1:] # Exclude 'date' column
            df_data = df_raw[cols_data]
        elif self.features == 'S': # Univariate
            df_data = df_raw[[self.target]]

        # Apply StandardScaler to the data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # Fit scaler only on training data
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Extract time features
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0: # Categorical time features
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1: # Numerical (Fourier) time features
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) # (num_features, num_timestamps) -> (num_timestamps, num_features)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2] # For forecasting, target is often the same as input
        self.data_stamp = data_stamp

        # Apply data augmentation if specified (only for training set)
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, _ = run_augmentation_single(self.data_x, self.data_y, self.args)

    def __getitem__(self, index):
        '''
        Returns one sample of data.
        '''
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        '''
        Returns the total number of samples available for a sliding window approach.
        '''
        # (Total length of data) - (input sequence length) - (prediction length) + 1
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        '''
        Inverse transforms scaled data back to its original scale.
        '''
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    '''
    Dataset class for the M4 competition dataset.

    This class handles loading and preparing M4 time series, which have varying
    lengths and specific train/test splits. It's tailored for short-term forecasting.
    '''

    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv', # data_path not directly used by M4
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        '''
        Initializes the M4 dataset.

        Args:
            args: Configuration arguments.
            root_path (str): Directory containing M4 dataset files.
            flag (str): Data split type ('train', 'pred'). 'pred' is used for test/evaluation here.
            size (list, optional): [seq_len, label_len, pred_len] defining input/output lengths.
            features (str): Feature type ('S' for univariate M4).
            data_path (str): Not directly used by M4 dataset class.
            target (str): Not directly used by M4 dataset class.
            scale (bool): Whether to apply StandardScaler (False by default for M4).
            inverse (bool): Whether to perform inverse transform (False by default).
            timeenc (int): Time feature encoding type (0: categorical, 1: numerical).
            freq (str): Frequency of data (e.g., '15min').
            seasonal_patterns (str): M4 specific seasonal pattern ('Yearly', 'Monthly', etc.).
        '''
        self.features = features
        self.target = target
        self.scale = scale # M4 often handled without external scaling
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        # Define sequence lengths based on model config
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        # M4 specific: history size and window sampling limit based on seasonal pattern
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        '''
        Loads M4 time series data, filtering by seasonal pattern.
        '''
        # M4Dataset.initialize() # If external initialization is needed
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else: # 'pred' flag usually corresponds to test/inference
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        
        # Filter time series based on the specified seasonal pattern
        training_values = [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]]
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        '''
        Returns one sample of data from the M4 dataset, typically involving random window sampling.
        '''
        # Initialize arrays for input, output, and masks
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        
        # Randomly select a cut point within the time series for window sampling
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        # Extract insample window
        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0 # Mask indicates valid data points

        # Extract outsample window (ground truth for prediction and label)
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0 # Mask indicates valid data points

        # Return insample (seq_x), outsample (seq_y), and their masks (used as x_mark, y_mark)
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        '''
        Returns the total number of individual time series in the dataset.
        '''
        return len(self.timeseries)

    def inverse_transform(self, data):
        '''
        Inverse transforms scaled data back to its original scale.
        (Note: M4 is typically unscaled, so this might not be used or implemented here.)
        '''
        # M4 datasets are often not scaled using StandardScaler; if used, scaler would need to be defined
        # For current M4 implementation, `scale=False` by default, so this might lead to error if called.
        raise NotImplementedError("Inverse transform is not typically implemented for M4 data in this setup.")

    def last_insample_window(self):
        '''
        Retrieves the last in-sample window for all time series.
        This function is used during evaluation to provide the model with the most
        recent historical data for forecasting.
        '''
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts_last_window):] = ts_last_window # Fill from the end
            insample_mask[i, -len(ts_last_window):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    '''
    Dataset loader for the PSM (Process Search Monitor) dataset, typically used for anomaly detection.

    This class prepares segmented windows of data and their corresponding labels.
    '''

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        '''
        Initializes the PSM segment loader.

        Args:
            args: Configuration arguments.
            root_path (str): Root directory containing 'train.csv', 'test.csv', 'test_label.csv'.
            win_size (int): Size of the sliding window for segments.
            step (int): Stride for the sliding window.
            flag (str): Data split type ("train", "val", "test").
        '''
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load and preprocess training data
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:] # Exclude timestamp column
        data = np.nan_to_num(data) # Handle potential NaN values
        self.scaler.fit(data) # Fit scaler on training data
        self.train = self.scaler.transform(data)

        # Load and preprocess test data
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:] # Exclude timestamp column
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)

        # Split training data to create a validation set
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        
        # Load test labels
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        
        print(f"PSMSegLoader - Train shape: {self.train.shape}, Test shape: {self.test.shape}")

    def __len__(self):
        '''
        Returns the total number of segments available for a sliding window approach.
        '''
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else: # For custom flags, e.g., 'inference' if different step logic is needed
            return (self.test.shape[0] - self.win_size) // self.win_size + 1 # Use win_size as step for full coverage

    def __getitem__(self, index):
        '''
        Returns one data segment and its corresponding label.
        '''
        index = index * self.step # Calculate start index of the segment
        
        # Extract data segment based on flag
        if self.flag == "train":
            # Note: test_labels are used for training data labels, likely for a specific AD setup.
            # This means the training labels are fixed to the first `win_size` of test_labels.
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'test':
            # For testing, labels correspond to the current segment
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            # Special case: segment extraction for other flags, e.g., using win_size as step
            start_idx = (index // self.step) * self.win_size
            return np.float32(self.test[
                              start_idx:start_idx + self.win_size]), np.float32(
                self.test_labels[start_idx:start_idx + self.win_size])


class MSLSegLoader(Dataset):
    '''
    Dataset loader for the MSL (Mars Science Laboratory) dataset, typically used for anomaly detection.

    Similar structure to PSMSegLoader, but loads data from .npy files.
    '''

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        '''
        Initializes the MSL segment loader.

        Args:
            args: Configuration arguments.
            root_path (str): Root directory containing 'MSL_train.npy', 'MSL_test.npy', 'MSL_test_label.npy'.
            win_size (int): Size of the sliding window for segments.
            step (int): Stride for the sliding window.
            flag (str): Data split type ("train", "val", "test").
        '''
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load and preprocess training data
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data) # Fit scaler on training data
        self.train = self.scaler.transform(data)

        # Load and preprocess test data
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)

        # Split training data to create a validation set
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        
        # Load test labels
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        
        print(f"MSLSegLoader - Train shape: {self.train.shape}, Test shape: {self.test.shape}")

    def __len__(self):
        '''
        Returns the total number of segments available for a sliding window approach.
        '''
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.flag == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        '''
        Returns one data segment and its corresponding label.
        '''
        index = index * self.step # Calculate start index of the segment
        
        # Extract data segment based on flag
        if self.flag == "train":
            # Note: test_labels are used for training data labels, likely for a specific AD setup.
            # This means the training labels are fixed to the first `win_size` of test_labels.
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.flag == 'test':
            # For testing, labels correspond to the current segment
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            # Special case: segment extraction for other flags, e.g., using win_size as step
            start_idx = (index // self.step) * self.win_size
            return np.float32(self.test[
                              start_idx:start_idx + self.win_size]), np.float32(
                self.test_labels[start_idx:start_idx + self.win_size])


class SMAPSegLoader(Dataset):
    '''
    Dataset loader for the SMAP (Soil Moisture Active Passive) dataset, typically used for anomaly detection.

    Similar structure to PSMSegLoader, but loads data from .npy files.
    '''

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        '''
        Initializes the SMAP segment loader.

        Args:
            args: Configuration arguments.
            root_path (str): Root directory containing 'SMAP_train.npy', 'SMAP_test.npy', 'SMAP_test_label.npy'.
            win_size (int): Size of the sliding window for segments.
            step (int): Stride for the sliding window.
            flag (str): Data split type ("train", "val", "test").
        '''
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load and preprocess training data
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data) # Fit scaler on training data
        self.train = self.scaler.transform(data)

        # Load and preprocess test data
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)

        # Split training data to create a validation set
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        
        # Load test labels
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        
        print(f"SMAPSegLoader - Train shape: {self.train.shape}, Test shape: {self.test.shape}")

    def __len__(self):
        '''
        Returns the total number of segments available for a sliding window approach.
        '''

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        '''
        Returns one data segment and its corresponding label.
        '''
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    '''
    Dataset loader for the SMD (Server Machine Dataset) dataset, typically used for anomaly detection.

    Similar structure to PSMSegLoader, but loads data from .npy files and has a different default step.
    '''

    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        '''
        Initializes the SMD segment loader.

        Args:
            args: Configuration arguments.
            root_path (str): Root directory containing 'SMD_train.npy', 'SMD_test.npy', 'SMD_test_label.npy'.
            win_size (int): Size of the sliding window for segments.
            step (int): Stride for the sliding window (default 100 for SMD).
            flag (str): Data split type ("train", "val", "test").
        '''
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load and preprocess training data
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data) # Fit scaler on training data
        self.train = self.scaler.transform(data)

        # Load and preprocess test data
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)

        # Split training data to create a validation set
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        
        # Load test labels
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

    def __len__(self):
        '''
        Returns the total number of segments available for a sliding window approach.
        '''
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        '''
        Returns one data segment and its corresponding label.
        '''
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    '''
    Dataset loader for the SWAT (Secure Water Treatment) dataset, typically used for anomaly detection.

    Similar structure to PSMSegLoader, but loads data from CSV files and preprocesses labels.
    '''

    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        '''
        Initializes the SWAT segment loader.

        Args:
            args: Configuration arguments.
            root_path (str): Root directory containing 'swat_train2.csv', 'swat2.csv'.
            win_size (int): Size of the sliding window for segments.
            step (int): Stride for the sliding window.
            flag (str): Data split type ("train", "val", "test").
        '''
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load and preprocess training data
        train_data_df = pd.read_csv(os.path.join(root_path, 'swat_train2.csv'))
        train_data = train_data_df.values[:, :-1] # Exclude potential last column if it's not a feature
        
        # Load test data and labels
        test_data_df = pd.read_csv(os.path.join(root_path, 'swat2.csv'))
        labels = test_data_df.values[:, -1:] # Last column assumed to be labels
        test_data = test_data_df.values[:, :-1] # Exclude labels from features

        self.scaler.fit(train_data)
        self.train = self.scaler.transform(train_data)
        self.test = self.scaler.transform(test_data)

        # Split training data to create a validation set
        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]
        
        # Store test labels
        self.test_labels = labels
        
        print(f"SWATSegLoader - Train shape: {self.train.shape}, Test shape: {self.test.shape}")

    def __len__(self):
        '''
        Number of images in the object dataset.
        '''
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        '''
        Returns one data segment and its corresponding label.
        '''
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    '''
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    '''

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        '''
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        '''
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)

# --- ADDED PEMS DATASET CLASS ---
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class Dataset_PEMS(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='pems-bay.npz',  # data_path现在应指向包含3列特征的文件
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        # --- 1. 加载已包含 [value, tod, dow] 的数据 ---
        data_file = os.path.join(self.root_path, self.data_path)
        print(f'加载PEMS数据 (包含value, tod, dow): {data_file}')
        # 假设数据形状为 (总时间步数, 节点数, 3)
        full_data = np.load(data_file, allow_pickle=True)['data']

        # --- 2. 数据集划分 ---
        train_ratio = 0.6
        valid_ratio = 0.2
        num_samples = len(full_data)
        num_train = int(num_samples * train_ratio)
        num_valid = int(num_samples * valid_ratio)

        border1s = [0, num_train, num_train + num_valid]
        border2s = [num_train, num_train + num_valid, num_samples]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 获取当前数据集划分
        current_data_split = full_data[border1:border2]

        # --- 3. 数据集划分 ---
        self.scaler = StandardScaler()
        if self.scale:
            # a. 从训练集中提取 'value' 特征来拟合 scaler
            train_values = full_data[:num_train, :, 0]
            self.scaler.fit(train_values)

            # b. 从当前数据划分中提取 'value' 特征进行变换
            values_to_scale = current_data_split[:, :, 0]
            scaled_values = self.scaler.transform(values_to_scale)

            # c. 将缩放后的 'value' 与原始的 'tod' 和 'dow' 重组
            time_features = current_data_split[:, :, 1:]
            scaled_values_3d = np.expand_dims(scaled_values, axis=-1)

            # d. 最终的数据
            data = np.concatenate([scaled_values_3d, time_features], axis=-1)
            print(f"数据已选择性标准化 (fit on train split's value feature)。")
        else:
            data = current_data_split

        print(f"最终 '{self.set_type}' 数据形状: {data.shape}")

        self.data_x = data
        self.data_y = data

    def __getitem__(self, index):
        # 与您的实验代码逻辑保持一致
        if self.set_type == 2:  # test
            s_begin = index * 1        #以前是*12
        else:
            s_begin = index

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        # _mark 变量是占位符, 因为您的模型框架会忽略它们
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        total_len = len(self.data_x)
        # 确保窗口不会超出数据末端
        # 注意: 您的原始 __len__ 对于预测任务是正确的
        # len(self.data_x) - self.seq_len - self.pred_len + 1
        window_size = self.seq_len + self.pred_len

        if self.set_type == 2:  # test: sliding window with stride 12
            return (total_len - window_size + 1) // 1 if total_len >= window_size else 0       #以前是// 12
        else:
            return total_len - window_size + 1 if total_len >= window_size else 0

    def inverse_transform(self, data):
        # data是模型预测出的'value'部分, 形状为(B*T, N), 正好匹配scaler
        return self.scaler.inverse_transform(data)
