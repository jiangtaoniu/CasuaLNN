import os
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.dtw_metric import dtw, accelerated_dtw

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    """
    Experiment class for Long-Term Forecasting.
    This class handles the training and evaluation pipeline for models applied
    to long-term forecasting tasks, including specific logic for causal models.
    """

    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.setting_with_time = None

        # Initialize causal loss parameters with default values if not explicitly provided.
        # These parameters are model-specific but managed at the experiment level for flexibility.
        if not hasattr(args, 'lambda_causal'):
            self.args.lambda_causal = 0.0  # Weight for the DAG constraint loss
        if not hasattr(args, 'lambda_l1'):
            self.args.lambda_l1 = 0.0      # Weight for L1 sparsity on the causal graph
        if not hasattr(args, 'causal_loss_type'):
            self.args.causal_loss_type = 'polynomial_notears' # Type of DAG constraint

        print(f">>> [Experiment] Causal loss type: {self.args.causal_loss_type}")
        print(f">>> [Experiment] Causal loss lambda: {self.args.lambda_causal}, L1 lambda: {self.args.lambda_l1}")


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
        Selects and returns the optimizer for model training.
        """
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        Selects and returns the loss criterion for model training.
        Uses L1Loss for PEMS data and MSELoss for other datasets.
        """
        if self.args.data == 'PEMS':
            return nn.L1Loss()
        else:
            return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        """
        Performs validation on the given data loader.
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Specific data handling for PEMS and Solar datasets
                if self.args.data == 'PEMS':
                    # For PEMS, target might have an extra dimension or need specific slicing
                    batch_y = batch_y[:, :, :, 0].float().to(self.device)
                if self.args.data == 'PEMS' or self.args.data == 'Solar':
                    # Mark data might not be relevant or needs to be set to None
                    batch_x_mark, batch_y_mark = None, None

                # Initialize decoder input for Autoformer-like models
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Model forward pass
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Extract relevant output dimensions
                # f_dim = -1 for multivariate-to-univariate (MS) or 0 for other feature types
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred, true = outputs.detach(), batch_y.detach()

                # Inverse transform predictions for PEMS data to calculate metrics on original scale
                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred_np = vali_data.inverse_transform(pred.cpu().numpy().reshape(-1, C)).reshape(B, T, C)
                    true_np = vali_data.inverse_transform(true.cpu().numpy().reshape(-1, C)).reshape(B, T, C)
                    mae, _, _, _, _ = metric(pred_np, true_np)
                    total_loss.append(mae)
                else:
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train() # Set model back to training mode
        return total_loss

    def train(self, setting):
        """
        Conducts the training process for the model.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.setting_with_time = f"{timestamp}_{setting}"
        print(f"Starting training, experiment ID: {self.setting_with_time}")

        # Create directory for saving checkpoints and logs
        path = os.path.join(self.args.checkpoints, self.setting_with_time)
        if not os.path.exists(path):
            os.makedirs(path)

        # Initialize log file
        log_file_path = os.path.join(path, 'run_log.txt')
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Experiment Setting: {self.setting_with_time}\n")
            log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            log_file.write("------------------- Hyperparameters -------------------")
            for arg, value in sorted(vars(self.args).items()):
                log_file.write(f"{arg}: {value}\n")
            log_file.write("-----------------------------------------------------")
            log_file.write("\n-------------------- Training Log -------------------")
            if self.args.lambda_causal > 0 or self.args.lambda_l1 > 0:
                log_file.write(f"[Causal Exp] Using DAG loss type: {self.args.causal_loss_type}\n\n")

        # Load data
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # Learning rate scheduler for OneCycleLR strategy
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim, steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start, epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        # GradScaler for Automatic Mixed Precision (AMP) training
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Check if the model is a causal model and if causal loss should be added
        is_causal_model = self.args.model == 'CasuaLNN'
        add_causal_loss = is_causal_model and (self.args.lambda_causal > 0 or self.args.lambda_l1 > 0)

        # --- Training Loop ---
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train() # Set model to training mode
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad() # Zero gradients for each batch

                # Move data to device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Specific data handling for PEMS and Solar datasets
                if self.args.data == 'PEMS': batch_y = batch_y[:, :, :, 0].float().to(self.device)
                if self.args.data == 'PEMS' or self.args.data == 'Solar':
                    batch_x_mark, batch_y_mark = None, None

                # Initialize decoder input for Autoformer-like models
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                total_causal_loss_item, pred_loss_item = 0.0, 0.0 # Track individual loss components

                def forward_and_loss_computation():
                    """
                    Performs a single forward pass and computes the total loss,
                    including prediction loss and, if applicable, causal loss.
                    """
                    nonlocal pred_loss_item, total_causal_loss_item

                    # Model forward pass. Causal models might return additional outputs like a causal graph.
                    if is_causal_model and hasattr(self.model, 'causal_graph_logits'):
                         # Assuming the model returns outputs and causal_graph_logits
                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                         G = self.model.causal_graph_logits # Extract causal graph logits for loss calculation
                    else:
                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                         G = None

                    # Extract prediction part and ground truth
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_pred = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    # Compute prediction loss
                    loss = criterion(outputs, batch_y_pred)
                    pred_loss_item = loss.item()

                    # Add causal loss components if enabled and applicable
                    if add_causal_loss and G is not None:
                        try:
                            loss_causal = torch.tensor(0.0, device=self.device)
                            N = self.args.enc_in # Number of variables/nodes in the causal graph
                            
                            # L1 regularization on the causal graph G
                            if self.args.lambda_l1 > 0:
                                l1_loss = self.args.lambda_l1 * torch.norm(G, p=1)
                                if torch.isfinite(l1_loss): # Only add if finite to prevent NaN propagation
                                    loss_causal += l1_loss
                            
                            # DAG constraint loss (ensures acyclicity)
                            if self.args.lambda_causal > 0:
                                dag_loss_val = torch.tensor(0.0, device=G.device)
                                if self.args.causal_loss_type == 'polynomial_notears':
                                    # Polynomial approximation of the acyclicity constraint (h(G) = trace((I+G^2/N)^N) - N)
                                    G_squared = G * G
                                    I = torch.eye(N, device=G.device)
                                    A = I + (G_squared / N)
                                    M = torch.matrix_power(A, N)
                                    dag_loss_val = self.args.lambda_causal * (torch.trace(M) - N)
                                elif self.args.causal_loss_type == 'dagma_logdet':
                                    # Log-determinant formulation for DAG constraint, often used in DAGMA
                                    G_squared = G * G
                                    I = torch.eye(N, device=G.device)
                                    # Numerically stable log-determinant (using slogdet for sign and logabsdet)
                                    _, logdet = torch.slogdet(I - G_squared)
                                    dag_loss_val = self.args.lambda_causal * (-logdet)

                                if torch.isfinite(dag_loss_val): # Only add if finite
                                    loss_causal += dag_loss_val
                            
                            # Add total causal loss to the main prediction loss
                            if torch.isfinite(loss_causal):
                                loss += loss_causal
                                total_causal_loss_item = loss_causal.item()

                        except Exception as e:
                            # Log warning and skip causal loss for current batch if calculation fails
                            print(f"Warning (Epoch {epoch + 1}, Iter {i + 1}): Causal loss calculation failed: {e}. Skipping for this batch.")
                    
                    return loss

                # Backpropagation step
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = forward_and_loss_computation()
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss = forward_and_loss_computation()
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

                # Log training progress
                if (i + 1) % 100 == 0:
                    log_msg = f"\titers: {i + 1}, epoch: {epoch + 1} | total_loss: {loss.item():.7f} (pred: {pred_loss_item:.7f})"
                    if add_causal_loss and total_causal_loss_item > 0:
                        log_msg += f" | (causal_loss_sum: {total_causal_loss_item:.7f})"
                    print(log_msg)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; estimated time remaining: {:.4f}s'.format(speed, left_time))
                    iter_count, time_now = 0, time.time() # Reset counters for speed estimation

                # Learning rate adjustment for TST strategy
                if self.args.lradj == 'TST':
                    scheduler.step()

            # End of epoch logging and validation
            current_epoch_time = time.time() - epoch_time
            print(f"Epoch: {epoch + 1} cost time: {current_epoch_time:.2f}s")
            train_loss_avg = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss_avg:.7f} "
                  f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            # Append epoch results to log file
            with open(log_file_path, 'a') as log_file:
                log_file.write(
                    f"Epoch: {epoch + 1:03d} | Train Loss: {train_loss_avg:.7f} | Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f} | Time: {current_epoch_time:.2f}s\n"
                )

            # Early stopping check
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"\nEarly stopping triggered at epoch {epoch + 1}.\n")
                break

            # Adjust learning rate for non-TST strategies
            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            else:
                print(f'Learning rate updated to {scheduler.get_last_lr()[0]:.6f}')

        # Load the best model found by early stopping
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        """
        Evaluates the trained model on the test set.
        """
        # Ensure setting_with_time is available, especially if running test-only mode
        if self.setting_with_time is None:
            if test == 1: # If explicitly running test mode
                self.setting_with_time = setting
            else:
                print("Warning: setting_with_time is None in test mode, using provided setting.")
                self.setting_with_time = setting

        print(f"Starting testing, experiment ID: {self.setting_with_time}")

        test_data, test_loader = self._get_data(flag='test')
        if test: # If test-only, load pre-trained model
            print('Loading model from checkpoint...')
            model_path = os.path.join(self.args.checkpoints, self.setting_with_time, 'checkpoint.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Error: Checkpoint file not found at {model_path}")

            state_dict = torch.load(model_path, map_location=self.device)
            # Handle DataParallel state dict loading if necessary
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError:
                print("Detected DataParallel model, attempting to fix state dict keys for single GPU load...")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '') # Remove 'module.' prefix
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

        preds, trues = [], []
        # Create folder for test results and visualizations
        folder_path = './test_results/' + self.setting_with_time + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Specific data handling for PEMS and Solar datasets
                if self.args.data == 'PEMS': batch_y = batch_y[:, :, :, 0].float().to(self.device)
                if self.args.data == 'PEMS' or self.args.data == 'Solar':
                    batch_x_mark, batch_y_mark = None, None

                # Initialize decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Model forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Extract prediction and ground truth
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

                # Visualize predictions every 20 batches
                if i % 20 == 0:
                    try:
                        input_data = batch_x.detach().cpu().numpy()
                        # Ensure data is suitable for visualization (e.g., single series)
                        if input_data.shape[0] > 0 and batch_y.shape[0] > 0:
                            # Visualize the last feature for a single sample
                            gt = np.concatenate((input_data[0, :, -1], trues[-1][0, :, -1]), axis=0)
                            pd = np.concatenate((input_data[0, :, -1], preds[-1][0, :, -1]), axis=0)
                            visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    except Exception as e:
                        print(f"Warning: Visualization failed for batch {i}: {e}")

        # Concatenate all predictions and ground truths
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('Test data shape before inverse transform:', preds.shape, trues.shape)

        # Inverse transform if applicable
        if self.args.data == 'PEMS':
            # PEMS specific inverse transform
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)
        elif hasattr(test_data, 'inverse_transform') and self.args.inverse:
            # General inverse transform if `inverse` flag is set
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)
        
        print('Test data shape after inverse transform:', preds.shape, trues.shape)

        # Calculate DTW metric if enabled
        dtw_val = 'N/A'
        if self.args.use_dtw:
            print("Calculating DTW...")
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                # Reshape for DTW (expecting 2D arrays: (sequence_length, num_features))
                x, y = preds[i].reshape(-1, 1), trues[i].reshape(-1, 1)
                if i % 100 == 0: print(f"DTW calculation progress: {i} of {preds.shape[0]}")
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_val = np.array(dtw_list).mean()
            print("DTW calculation finished.")

        # Calculate standard metrics (MAE, MSE, RMSE, MAPE, MSPE)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('--- Final Test Metrics ---')
        print(f'MSE:  {mse:.7f}\nMAE:  {mae:.7f}\nRMSE: {rmse:.7f}\nMAPE: {mape:.7f}\nMSPE: {mspe:.7f}\nDTW:  {dtw_val}')
        print('--------------------------')

        # Log final test results to file
        log_file_path = os.path.join(self.args.checkpoints, self.setting_with_time, 'run_log.txt')
        try:
            with open(log_file_path, 'a') as log_file:
                log_file.write("\n--------------- Final Test Results ---------------")
                log_file.write(f"Tested at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f'MSE:  {mse:.7f}\nMAE:  {mae:.7f}\nRMSE: {rmse:.7f}\nMAPE: {mape:.7f}\nMSPE: {mspe:.7f}\nDTW:  {dtw_val}\n')
                log_file.write("--------------------------------------------------")
        except FileNotFoundError:
            print(f"Warning: Log file not found at {log_file_path}. Skipping test metric logging.")

        # Save results to a summary file and individual numpy files
        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(self.setting_with_time + "  \n")
            f.write(f"mse:{mse:.7f}, mae:{mae:.7f}, rmse:{rmse:.7f}, mape:{mape:.7f}, mspe:{mspe:.7f}, dtw:{dtw_val}\n\n")

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)
        print("Test results saved.")

        # This return statement was empty, ensuring it returns something if intended, or removed if not.
        # Since it's at the end of a method that is primarily for side-effects (testing, logging),
        # explicitly returning None or removing it is fine. For now, matching the original implicit return.
