from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.m4_summary import M4Summary
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd # Changed alias from 'pandas' to 'pd' for common convention

warnings.filterwarnings('ignore')


class Exp_Short_Term_Forecast(Exp_Basic):
    """
    Experiment class for Short-Term Forecasting, specifically tailored for the M4 dataset.

    This class extends Exp_Basic to handle the unique data loading, model configuration,
    training, and evaluation procedures required for short-term forecasting on M4.
    It incorporates M4-specific loss functions and evaluation metrics.
    """

    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)
        # Note: Short-term forecasting for M4 does not typically involve causal loss,
        # so related args (lambda_causal, lambda_l1) are not initialized here.

    def _build_model(self):
        """
        Builds and returns the model instance, applying M4-specific configurations.

        For the M4 dataset, pred_len, seq_len, label_len, and frequency_map are
        dynamically determined based on the seasonal pattern defined in args.
        """
        if self.args.data == 'm4':
            self.args.pred_len = M4Meta.horizons_map[self.args.seasonal_patterns]  # Set prediction length based on M4 config
            self.args.seq_len = 2 * self.args.pred_len  # Input length is typically twice the prediction length for M4
            self.args.label_len = self.args.pred_len    # Label length for decoder input
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns] # Frequency for time features
        
        # Instantiate the model from the model dictionary defined in Exp_Basic
        model = self.model_dict[self.args.model].Model(self.args).float()

        # Wrap model with DataParallel if multi-GPU is enabled
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

    def _select_criterion(self, loss_name='MSE'):
        """
        Selects and returns the appropriate loss criterion based on the specified loss name.

        Args:
            loss_name (str): The name of the loss function ('MSE', 'MAPE', 'MASE', 'SMAPE').
        Returns:
            torch.nn.Module: The selected loss function.
        """
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()  # Mean Absolute Percentage Error (custom implementation)
        elif loss_name == 'MASE':
            return mase_loss()  # Mean Absolute Scaled Error (custom implementation)
        elif loss_name == 'SMAPE':
            return smape_loss() # Symmetric Mean Absolute Percentage Error (custom implementation)
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def train(self, setting):
        """
        Conducts the training process for the model on short-term forecasting tasks.
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # Create directory for saving checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        # mse = nn.MSELoss() # Declared but not directly used in the current loss calculation

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
                batch_y = batch_y.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Decoder input: concatenate ground truth from historical data with zeros for prediction horizon
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Model forward pass (Note: x_mark and y_mark are passed as None here)
                outputs = self.model(batch_x, None, dec_inp, None)

                # Extract relevant output dimensions and align with ground truth
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y_mark = batch_y_mark[:, -self.args.pred_len:, f_dim:].to(self.device)

                # Calculate loss
                # Custom M4 losses (mape_loss, mase_loss, smape_loss) often require additional inputs like frequency map
                loss_value = criterion(batch_x, self.args.frequency_map, outputs, batch_y, batch_y_mark)
                
                # Optionally add a sharpness loss (currently commented out)
                # loss_sharpness = mse((outputs[:, 1:, :] - outputs[:, :-1, :]), (batch_y[:, 1:, :] - batch_y[:, :-1, :]))
                loss = loss_value # + loss_sharpness * 1e-5
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
            vali_loss = self.vali(train_loader, vali_loader, criterion) # Validation uses training data for scaling in M4 losses
            test_loss = vali_loss # Test loss is set to validation loss in this setup
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            
            # Early stopping check
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

            # Adjust learning rate (using a custom function `adjust_learning_rate`)
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # Load the best model found by early stopping
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def vali(self, train_loader, vali_loader, criterion):
        """
        Performs validation for short-term forecasting, specifically for the M4 dataset.

        Args:
            train_loader: Used to get the last in-sample window for scaling (M4 specific).
            vali_loader: The validation data loader.
            criterion: The loss criterion.
        Returns:
            float: The average validation loss.
        """
        # M4 specific: Get the last in-sample window from training data for scaling or reference
        x, _ = train_loader.dataset.last_insample_window()
        y = vali_loader.dataset.timeseries # Ground truth time series for validation
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1) # Add feature dimension if univariate

        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            B, _, C = x.shape
            # Decoder input: concatenate ground truth from historical data with zeros for prediction horizon
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            
            # Perform inference in batches for potentially large validation sets
            outputs = torch.zeros((B, self.args.pred_len, C)).float()
            id_list = np.arange(0, B, 500) # Process in chunks of 500
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                batch_start, batch_end = id_list[i], id_list[i + 1]
                outputs[batch_start:batch_end, :, :] = self.model(
                    x[batch_start:batch_end], None,
                    dec_inp[batch_start:batch_end], None
                ).detach().cpu()

            # Extract relevant output dimensions
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            
            pred = outputs
            true = torch.from_numpy(np.array(y))
            batch_y_mark = torch.ones(true.shape) # Mark tensor for custom loss

            # Calculate validation loss using the specified criterion
            loss = criterion(x.detach().cpu()[:, :, 0], self.args.frequency_map, pred[:, :, 0], true, batch_y_mark)

        self.model.train() # Set model back to training mode
        return loss

    def test(self, setting, test=0):
        """
        Evaluates the trained model on the test set for short-term forecasting (M4).
        """
        _, train_loader = self._get_data(flag='train')
        _, test_loader = self._get_data(flag='test')
        
        # M4 specific: Get the last in-sample window from training data
        x, _ = train_loader.dataset.last_insample_window()
        y = test_loader.dataset.timeseries # Ground truth time series for test
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x = x.unsqueeze(-1) # Add feature dimension if univariate

        if test: # If running in test-only mode, load the pre-trained model
            print('Loading model for testing...')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Error: Checkpoint file not found at {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        # Create folder for test results and visualizations
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            B, _, C = x.shape
            # Decoder input
            dec_inp = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            dec_inp = torch.cat([x[:, -self.args.label_len:, :], dec_inp], dim=1).float()
            
            # Perform inference in batches for potentially large test sets
            outputs = torch.zeros((B, self.args.pred_len, C)).float().to(self.device)
            id_list = np.arange(0, B, 1) # Process one by one or in small batches
            id_list = np.append(id_list, B)
            for i in range(len(id_list) - 1):
                batch_start, batch_end = id_list[i], id_list[i + 1]
                outputs[batch_start:batch_end, :, :] = self.model(
                    x[batch_start:batch_end], None,
                    dec_inp[batch_start:batch_end], None
                )

                if id_list[i] % 1000 == 0:
                    print(f"Test inference progress: {id_list[i]} of {B} samples.")

            # Extract relevant output dimensions and convert to numpy
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            preds = outputs.detach().cpu().numpy()
            trues = y # Ground truth for M4 is typically already in numpy
            x = x.detach().cpu().numpy() # Input for visualization

            # Visualize predictions for a subset of samples
            for i in range(0, preds.shape[0], preds.shape[0] // 10): # Visualize 10 evenly spaced samples
                # Concatenate historical input with predictions/ground truth for visualization
                gt = np.concatenate((x[i, :, 0], trues[i]), axis=0)
                pd = np.concatenate((x[i, :, 0], preds[i, :, 0]), axis=0)
                visual(gt, pd, os.path.join(folder_path, f'{i}.pdf'))

        print('Test prediction shape:', preds.shape)

        # --- M4 Specific Result Saving and Evaluation ---
        # Save forecasts to CSV files
        m4_results_folder = './m4_results/' + self.args.model + '/'
        if not os.path.exists(m4_results_folder):
            os.makedirs(m4_results_folder)

        forecasts_df = pd.DataFrame(preds[:, :, 0], columns=[f'V{i + 1}' for i in range(self.args.pred_len)])
        forecasts_df.index = test_loader.dataset.ids[:preds.shape[0]]
        forecasts_df.index.name = 'id'
        # Set the 'id' column as the index for M4 submission format
        # forecasts_df.set_index(forecasts_df.columns[0], inplace=True) # This line seems incorrect; should be id
        forecasts_df.to_csv(os.path.join(m4_results_folder, f'{self.args.seasonal_patterns}_forecast.csv'))

        print(f"Forecasts saved for model: {self.args.model}, seasonal pattern: {self.args.seasonal_patterns}")
        
        # Check if all M4 seasonal patterns have been processed for aggregated evaluation
        expected_m4_files = [
            'Weekly_forecast.csv', 'Monthly_forecast.csv', 'Yearly_forecast.csv',
            'Daily_forecast.csv', 'Hourly_forecast.csv', 'Quarterly_forecast.csv'
        ]
        all_files_present = True
        for filename in expected_m4_files:
            if filename not in os.listdir(m4_results_folder):
                all_files_present = False
                break

        if all_files_present:
            print(f'All 6 M4 tasks are finished. Calculating averaged evaluation metrics...')
            m4_summary = M4Summary(m4_results_folder, self.args.root_path)
            smape_results, owa_results, mape, mase = m4_summary.evaluate()
            print(f'Final M4 Averaged Metrics:')
            print(f'  SMAPE: {smape_results:.4f}')
            print(f'  MAPE:  {mape:.4f}')
            print(f'  MASE:  {mase:.4f}')
            print(f'  OWA:   {owa_results:.4f}')
            
            # Save final M4 results to a summary file
            with open("result_short_term_forecast_m4_summary.txt", 'a') as f:
                f.write(f'{self.args.model} - M4 Summary\n')
                f.write(f'  SMAPE: {smape_results:.4f}\n')
                f.write(f'  MAPE:  {mape:.4f}\n')
                f.write(f'  MASE:  {mase:.4f}\n')
                f.write(f'  OWA:   {owa_results:.4f}\n\n')

        else:
            print('After all 6 M4 seasonal patterns are finished, the averaged index can be calculated.')
        return