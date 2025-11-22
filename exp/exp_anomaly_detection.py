import torch.multiprocessing
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

# This is often necessary for multi-process data loading (e.g., using num_workers > 0 in DataLoader)
# on systems like Linux to prevent issues with shared memory.
torch.multiprocessing.set_sharing_strategy('file_system')

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    """
    Experiment class for Time Series Anomaly Detection tasks.

    This class handles the training and evaluation pipeline for models
    designed to detect anomalies in time series data. It typically involves
    training a reconstruction model (often unsupervised), calculating
    reconstruction errors, setting a threshold, and then evaluating
    detection performance.
    """

    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        """
        Builds and returns the model instance.
        The model is selected from `self.model_dict` (defined in Exp_Basic)
        based on `self.args.model`. It also handles DataParallel wrapping
        if multi-GPU training is enabled.
        """
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        Retrieves the dataset and data loader for a given flag (e.g., 'train', 'val', 'test').
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        Selects and returns the Adam optimizer for model training.
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        Selects and returns the Mean Squared Error (MSE) loss criterion.
        For anomaly detection, this is typically used to measure reconstruction error.
        """
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        Performs validation for anomaly detection by calculating the average reconstruction loss.
        Labels (batch_y) are not used in this process as training is often unsupervised.
        """
        total_loss = []
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            for i, (batch_x, batch_y) in enumerate(vali_loader): # batch_y is ignored here
                batch_x = batch_x.float().to(self.device)

                # Model forward pass to reconstruct input
                outputs = self.model(batch_x, None, None, None)

                # Extract relevant feature dimensions
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                
                pred = outputs.detach()
                true = batch_x.detach()

                # Calculate reconstruction loss
                loss = criterion(pred, true)
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train() # Set model back to training mode
        return total_loss

    def train(self, setting):
        """
        Conducts the training process for the model on anomaly detection tasks.
        Models are typically trained to reconstruct normal behavior, often in an unsupervised manner.
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # Create directory for saving checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # --- Training Loop ---
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train() # Set model to training mode
            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader): # batch_y (labels) is ignored for unsupervised training
                iter_count += 1
                model_optim.zero_grad() # Zero gradients for each batch

                batch_x = batch_x.float().to(self.device)

                # Model forward pass to reconstruct input
                outputs = self.model(batch_x, None, None, None)

                # Extract relevant feature dimensions
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                
                # Calculate reconstruction loss
                loss = criterion(outputs, batch_x)
                train_loss.append(loss.item())

                # Log training progress
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; estimated time remaining: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward() # Backpropagation
                model_optim.step() # Update model parameters

            # End of epoch logging and validation
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion) # Evaluate on test set (reconstruction loss)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} "
                  f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            # Early stopping check
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
            
            # Adjust learning rate using a custom function `adjust_learning_rate`
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Load the best model found by early stopping
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """
        Evaluates the trained model for anomaly detection on the test set.

        This involves:
        1.  Calculating anomaly scores (reconstruction errors) on the training set
            to establish a baseline for normal behavior.
        2.  Calculating anomaly scores on the test set.
        3.  Determining a threshold using the combined scores and `args.anomaly_ratio`.
        4.  Making binary anomaly predictions (`pred`).
        5.  Applying an adjustment to predictions (`adjustment` utility) to improve detection.
        6.  Evaluating performance using standard anomaly detection metrics (Precision, Recall, F1-score).
        """
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train') # Also load train data for thresholding

        if test: # If running in test-only mode, load the pre-trained model
            print('Loading model for testing...')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Error: Checkpoint file not found at {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        attens_energy = [] # Stores anomaly scores (reconstruction errors)
        # Create folder for test results (e.g., plots)
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval() # Set model to evaluation mode
        # Using MSELoss without reduction to get individual sample-wise errors
        self.anomaly_criterion = nn.MSELoss(reduction='none')

        # (1) Calculate anomaly scores on the training set (normal data)
        # This establishes the distribution of "normal" reconstruction errors.
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None) # Reconstruct
                
                # Compute per-sample reconstruction error (mean over features)
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                attens_energy.append(score.detach().cpu().numpy())

        train_energy = np.concatenate(attens_energy, axis=0).reshape(-1) # Flatten all train scores

        # (2) Calculate anomaly scores on the test set and collect ground truth labels
        attens_energy = [] # Reset for test data
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, None, None, None) # Reconstruct
                
                # Compute per-sample reconstruction error
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                attens_energy.append(score.detach().cpu().numpy())
                test_labels.append(batch_y.detach().cpu().numpy()) # Collect true labels

        test_energy = np.concatenate(attens_energy, axis=0).reshape(-1) # Flatten all test scores
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        gt = test_labels.astype(int) # Ensure ground truth labels are integers

        # Combine train and test scores to determine an appropriate threshold
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        # Threshold is set as the (100 - anomaly_ratio)-th percentile of combined scores.
        # This means `anomaly_ratio` % of the highest scores are considered anomalous.
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print(f"Anomaly Detection Threshold: {threshold:.4f}")

        # (3) Make binary anomaly predictions based on the threshold
        pred = (test_energy > threshold).astype(int) # 1 if anomalous, 0 if normal

        print("Raw prediction shape:", pred.shape)
        print("Ground truth shape:", gt.shape)

        # (4) Apply detection adjustment to refine predictions
        # This utility often merges adjacent anomalous points or aligns detected anomalies
        # with ground truth boundaries in a post-processing step.
        gt_adjusted, pred_adjusted = adjustment(gt, pred)

        print("Adjusted prediction shape:", pred_adjusted.shape)
        print("Adjusted ground truth shape:", gt_adjusted.shape)

        # (5) Evaluate performance using standard anomaly detection metrics
        accuracy = accuracy_score(gt_adjusted, pred_adjusted)
        precision, recall, f_score, support = precision_recall_fscore_support(gt_adjusted, pred_adjusted, average='binary')
        
        print("\n--- Final Anomaly Detection Metrics (Adjusted) ---")
        print(f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, "
              f"Recall : {recall:.4f}, F-score : {f_score:.4f}")
        print("--------------------------------------------------")

        # Log results to a summary file
        with open("result_anomaly_detection.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write(f"Accuracy : {accuracy:.4f}, Precision : {precision:.4f}, "
                    f"Recall : {recall:.4f}, F-score : {f_score:.4f}")
            f.write('\n\n')
        
        # Note: No numpy files are saved by default for raw predictions/scores in this setup.
        # If needed, `np.save` could be added here for `pred`, `gt`, `test_energy` etc.
        return