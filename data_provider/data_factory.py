from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_PEMS
from torch.utils.data import DataLoader


# Dictionary mapping dataset names to their respective Dataset classes.
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'PEMS': Dataset_PEMS,
}


def data_provider(args, flag):
    """
    Provides dataset and data loader instances based on specified arguments and flag.

    This function acts as a factory, dynamically selecting the appropriate Dataset class
    and configuring the DataLoader for long-term forecasting tasks.

    Args:
        args (argparse.Namespace): Configuration arguments for the experiment.
        flag (str): Indicates the data split ('train', 'val', 'test').

    Returns:
        tuple: A tuple containing the dataset instance and the data loader instance.
    """
    Data = data_dict[args.data]

    # Determine time encoding type: 0 for non-timeF embed, 1 for timeF embed.
    timeenc = 0 if args.embed != 'timeF' else 1

    # Shuffle training data, but not validation/test data for consistent evaluation.
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    data_set = Data(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        seasonal_patterns=args.seasonal_patterns
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