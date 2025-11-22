from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    """
    Experiment class for Time Series Imputation tasks.

    This class handles the training and evaluation pipeline for models
    designed to fill in missing values in time series data. It includes
    logic for generating masks, computing loss only on masked values,
    and evaluating imputation performance.
    """

    def __init__(self, args):
        super(Exp_Imputation, self).__init__(args)

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
        """
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        Performs validation on the given data loader for imputation tasks.
        A random mask is applied to the input, and loss is computed only on the masked values.
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Generate a random mask for imputation.
                # 'mask_rate' determines the proportion of values to be masked (set to 0).
                B, T, N = batch_x.shape
                # B = batch size, T = sequence length, N = number of features
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # Values to be imputed (masked)
                mask[mask > self.args.mask_rate] = 1   # Values to remain observed
                
                # Create input for the model: original data with masked values set to 0.
                inp = batch_x.masked_fill(mask == 0, 0)

                # Model forward pass for imputation.
                # The model receives the corrupted input, time features, and the mask.
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # Extract relevant feature dimensions.
                # f_dim = -1 for multivariate-to-univariate (MS) or 0 for other feature types
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # Adjust ground truth and mask for relevant feature dimensions.
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                pred = outputs.detach()
                true = batch_x.detach()
                mask = mask.detach()

                # Compute loss only on the masked (imputed) values.
                loss = criterion(pred[mask == 0], true[mask == 0])
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train() # Set model back to training mode
        return total_loss

    def train(self, setting):
        """
        Conducts the training process for the model on imputation tasks.
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

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad() # Zero gradients for each batch

                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Generate a random mask for imputation.
                # 'mask_rate' determines the proportion of values to be masked (set to 0).
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # Values to be imputed (masked)
                mask[mask > self.args.mask_rate] = 1   # Values to remain observed
                
                # Create input for the model: original data with masked values set to 0.
                inp = batch_x.masked_fill(mask == 0, 0)

                # Model forward pass for imputation
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # Extract relevant feature dimensions
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]

                # Adjust ground truth and mask for relevant feature dimensions
                batch_x = batch_x[:, :, f_dim:]
                mask = mask[:, :, f_dim:]

                # Compute loss only on the masked (imputed) values.
                loss = criterion(outputs[mask == 0], batch_x[mask == 0])
                train_loss.append(loss.item())

                # Log training progress
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; estimated time remaining: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            # End of epoch logging and validation
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

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
        Evaluates the trained model on the test set for imputation tasks.
        """
        test_data, test_loader = self._get_data(flag='test')
        if test: # If running in test-only mode, load the pre-trained model
            print('Loading model for testing...')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Error: Checkpoint file not found at {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        preds, trues, masks = [], [], []
        # Create folder for test results and visualizations
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                # Generate a random mask for imputation, identical to training/validation.
                B, T, N = batch_x.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # Values to be imputed (masked)
                mask[mask > self.args.mask_rate] = 1   # Values to remain observed
                
                # Create input for the model: original data with masked values set to 0.
                inp = batch_x.masked_fill(mask == 0, 0)

                # Model forward pass for imputation (imputes the masked values)
                outputs = self.model(inp, batch_x_mark, None, None, mask)

                # Extract relevant feature dimensions for evaluation
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                batch_x = batch_x[:, :, f_dim:] # Ground truth for relevant features
                mask = mask[:, :, f_dim:]       # Mask for relevant features

                # Store predictions, true values, and masks
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_x.detach().cpu().numpy())
                masks.append(mask.detach().cpu().numpy()) # Store as numpy for concatenation

                # Visualize imputation results for a sample every 20 batches
                if i % 20 == 0:
                    # 'filled' represents the original series where masked parts are replaced by predictions
                    filled_sample = trues[0][0, :, -1].copy() # Original series, last feature
                    masked_idx = (masks[0][0, :, -1] == 0) # Indices where values were masked
                    filled_sample[masked_idx] = preds[0][0, masked_idx, -1] # Fill masked parts with predictions
                    
                    # Original series (ground truth) vs. the filled series
                    visual(trues[0][0, :, -1], filled_sample, os.path.join(folder_path, f'{i}.pdf'))

        # Concatenate all stored arrays
        preds = np.concatenate(preds, 0)
        trues = np.concatenate(trues, 0)
        masks = np.concatenate(masks, 0)
        print('Test data shape:', preds.shape, trues.shape, masks.shape)

        # Save results and calculate metrics
        # Use a distinct folder for imputation results
        imputation_results_folder = './results/' + setting + '/'
        if not os.path.exists(imputation_results_folder):
            os.makedirs(imputation_results_folder)

        # Calculate metrics only on the imputed (masked) values
        mae, mse, rmse, mape, mspe = metric(preds[masks == 0], trues[masks == 0])
        print(f'Imputation Metrics: MSE: {mse:.7f}, MAE: {mae:.7f}')
        
        # Log results to a summary file
        with open("result_imputation.txt", 'a') as f:
            f.write(setting + "  \n")
            f.write(f'mse:{mse:.7f}, mae:{mae:.7f}')
            f.write('\n\n')

        np.save(os.path.join(imputation_results_folder, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(imputation_results_folder, 'pred.npy'), preds)
        np.save(os.path.join(imputation_results_folder, 'true.npy'), trues)
        np.save(os.path.join(imputation_results_folder, 'mask.npy'), masks)
        print("Test results saved.")
        return