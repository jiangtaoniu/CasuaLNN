from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader, Dataset_PEMS
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader


# Dictionary mapping dataset names to their respective Dataset classes.
# These classes are imported from data_provider.data_loader.
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom, # Generic custom dataset
    'm4': Dataset_M4,         # M4 competition dataset
    'PSM': PSMSegLoader,      # Process Search Monitor (Anomaly Detection)
    'MSL': MSLSegLoader,      # Mars Science Laboratory (Anomaly Detection)
    'SMAP': SMAPSegLoader,    # Soil Moisture Active Passive (Anomaly Detection)
    'SMD': SMDSegLoader,      # Server Machine Dataset (Anomaly Detection)
    'SWAT': SWATSegLoader,    # Secure Water Treatment (Anomaly Detection)
    'UEA': UEAloader,         # UEA (UCR) time series classification datasets
    'PEMS': Dataset_PEMS,     # PEMS traffic datasets
}


def data_provider(args, flag):
    """
    Provides dataset and data loader instances based on specified arguments and flag.

    This function acts as a factory, dynamically selecting the appropriate Dataset class
    and configuring the DataLoader for different time series tasks (forecasting,
    anomaly detection, classification).

    Args:
        args (argparse.Namespace): Configuration arguments for the experiment.
        flag (str): Indicates the data split ('train', 'val', 'test', 'TRAIN', 'TEST').

    Returns:
        tuple: A tuple containing the dataset instance and the data loader instance.
    """
    Data = data_dict[args.data] # Select the Dataset class based on args.data
    
    # Determine time encoding type: 0 for non-timeF embed, 1 for timeF embed.
    # This affects how time features are generated and embedded within the dataset.
    timeenc = 0 if args.embed != 'timeF' else 1

    # Shuffle training data, but not validation/test data for consistent evaluation.
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    
    # drop_last is typically True for training to avoid batch size inconsistencies,
    # but often False for evaluation to include all samples.
    drop_last = False
    
    batch_size = args.batch_size
    freq = args.freq

    # --- Anomaly Detection Task Configuration ---
    if args.task_name == 'anomaly_detection':
        drop_last = False # Ensure all samples are processed in evaluation
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len, # 'win_size' often refers to sequence length in AD tasks
            flag=flag,
        )
        print(f"{flag} data size: {len(data_set)} samples")
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader
    
    # --- Classification Task Configuration ---
    elif args.task_name == 'classification':
        drop_last = False # Ensure all samples are processed in evaluation
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )
        # For classification, a custom collate_fn might be needed to handle
        # variable-length sequences or specific padding requirements.
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len) # UEA specific collate_fn
        )
        return data_set, data_loader
    
    # --- General Forecasting/Imputation Task Configuration ---
    else:
        # For M4, drop_last is typically False during evaluation.
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len], # Define input/output sequence lengths
            features=args.features, # Type of features (M, S, MS)
            target=args.target,     # Target column name
            timeenc=timeenc,        # Time encoding type
            freq=freq,              # Frequency for time features
            seasonal_patterns=args.seasonal_patterns # M4 specific
        )
        print(f"{flag} data size: {len(data_set)} samples")
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
        return data_set, data_loader