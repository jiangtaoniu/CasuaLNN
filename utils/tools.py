import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

# Set matplotlib backend to 'agg' for non-interactive plotting,
# which is essential for running scripts on servers without a graphical display.
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjusts the learning rate of the optimizer based on a predefined schedule.

    Args:
        optimizer: The optimizer for which to adjust the learning rate.
        epoch (int): The current epoch number.
        args: An object containing experiment configurations, including `lradj`
              and `learning_rate`.
    """
    # Default to no adjustment
    lr_adjust = {}
    
    if args.lradj == 'type1':
        # Halve the learning rate at each epoch
        lr_adjust = {epoch: args.learning_rate * (0.5 ** (epoch - 1))}
    elif args.lradj == 'type2':
        # A specific, predefined learning rate schedule
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        # Decay learning rate by a factor of 0.9 after the 3rd epoch
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** (epoch - 3))}
    elif args.lradj == "cosine":
        # Cosine annealing schedule
        lr_adjust = {epoch: args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    
    # If the current epoch is in the adjustment dictionary, update the learning rate
    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'Updating learning rate to {lr}')


class EarlyStopping:
    """
    Implements early stopping to prevent overfitting.

    Monitors a validation metric and stops training if the metric does not improve
    after a given 'patience' number of epochs.
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How many epochs to wait for improvement before stopping.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        Call method to update early stopping state.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
            path (str): The path to save the best model checkpoint.
        """
        # For loss, lower is better, so score is negative loss.
        # For accuracy, higher is better, so a positive score is used.
        score = -val_loss

        if self.best_score is None:
            # First epoch, save the score and checkpoint
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # Metric did not improve
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Metric improved, save checkpoint and reset counter
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """
        Saves the model checkpoint when validation loss decreases.

        Args:
            val_loss (float): The validation loss.
            model (torch.nn.Module): The model to save.
            path (str): Directory path to save the checkpoint file.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, 'checkpoint.pth'))
        self.val_loss_min = val_loss


class dotdict(dict):
    """
    A dictionary subclass that allows attribute-style access (dot notation).
    e.g., d.key instead of d['key']
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    """
    A custom, simplified implementation of a standard scaler.
    """
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        """
        Args:
            mean (np.ndarray): The mean to use for scaling.
            std (np.ndarray): The standard deviation to use for scaling.
        """
        self.mean = mean
        self.std = std

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Scales the data using the stored mean and standard deviation.

        Formula: (data - mean) / std
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transforms the scaled data back to its original scale.

        Formula: (data * std) + mean
        """
        return (data * self.std) + self.mean


def visual(true: np.ndarray, preds: Optional[np.ndarray] = None, name: str = './pic/test.pdf'):
    """
    Visualizes the ground truth and predicted values and saves the plot.

    Args:
        true (np.ndarray): The ground truth time series values.
        preds (Optional[np.ndarray]): The predicted time series values. Defaults to None.
        name (str): The file path to save the plot. Defaults to './pic/test.pdf'.
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')
    plt.close() # Close the figure to free up memory


def adjustment(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Adjusts anomaly predictions based on ground truth labels.

    This function is used for more lenient evaluation of anomaly detection.
    If a point is correctly identified as part of a true anomaly segment,
    this function "fills in" the prediction for the entire segment, correcting
    for any missed points within that single true anomalous event.

    Args:
        gt (np.ndarray): The ground truth labels (1 for anomaly, 0 for normal).
        pred (np.ndarray): The binary prediction labels.

    Returns:
        tuple[np.ndarray, np.ndarray]: The original ground truth and the adjusted predictions.
    """
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            # Detected the start of a true anomaly segment
            anomaly_state = True
            # Adjust predictions for the entire true anomaly segment (backwards)
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1 # Correct missed prediction within the segment
            # Adjust predictions for the entire true anomaly segment (forwards)
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1 # Correct missed prediction within the segment
        elif gt[i] == 0:
            anomaly_state = False
        
        # If already inside a true anomaly segment, ensure prediction remains 1
        if anomaly_state:
            pred[i] = 1
            
    return gt, pred


def cal_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculates the classification accuracy.

    Args:
        y_pred (np.ndarray): Predicted class labels.
        y_true (np.ndarray): True class labels.

    Returns:
        float: The classification accuracy.
    """
    return np.mean(y_pred == y_true)