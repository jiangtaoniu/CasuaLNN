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
    This class is now generalized for the CasuaLNN model.
    """

    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        self.setting_with_time = None

        # --- Set default causal loss parameters if not provided ---
        if not hasattr(args, 'lambda_causal'): self.args.lambda_causal = 0.0
        if not hasattr(args, 'lambda_l1'): self.args.lambda_l1 = 0.0
        if not hasattr(args, 'causal_loss_type'):
            self.args.causal_loss_type = 'polynomial_notears'
        
        print(f">>> [Experiment] Causal loss type: {self.args.causal_loss_type}")
        print(f">>> [Experiment] Causal loss lambda: {self.args.lambda_causal}, L1 lambda: {self.args.lambda_l1}")


    def _build_model(self):
        # The model is selected from model_dict in the base class
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.data == 'PEMS':
            return nn.L1Loss()
        else:
            return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.data == 'PEMS': batch_y = batch_y[:, :, :, 0].float().to(self.device)
                if self.args.data == 'PEMS' or self.args.data == 'Solar':
                    batch_x_mark, batch_y_mark = None, None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                pred, true = outputs.detach(), batch_y.detach()

                if self.args.data == 'PEMS':
                    B, T, C = pred.shape
                    pred_np, true_np = pred.cpu().numpy(), true.cpu().numpy()
                    pred_np = vali_data.inverse_transform(pred_np.reshape(-1, C)).reshape(B, T, C)
                    true_np = vali_data.inverse_transform(true_np.reshape(-1, C)).reshape(B, T, C)
                    mae, _, _, _, _ = metric(pred_np, true_np)
                    total_loss.append(mae)
                else:
                    loss = criterion(pred, true)
                    total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.setting_with_time = f"{timestamp}_{setting}"
        print(f"Starting training, experiment ID: {self.setting_with_time}")
        path = os.path.join(self.args.checkpoints, self.setting_with_time)
        if not os.path.exists(path): os.makedirs(path)

        log_file_path = os.path.join(path, 'run_log.txt')
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Experiment Setting: {self.setting_with_time}\n")
            log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            log_file.write("------------------- Hyperparameters -------------------\n")
            for arg, value in sorted(vars(self.args).items()):
                log_file.write(f"{arg}: {value}\n")
            log_file.write("-----------------------------------------------------\n\n")
            log_file.write("-------------------- Training Log -------------------\n")
            if hasattr(self.args, 'causal_loss_type'):
                log_file.write(f"[Causal Exp] Using DAG loss type: {self.args.causal_loss_type}\n\n")

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim, steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start, epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.data == 'PEMS': batch_y = batch_y[:, :, :, 0].float().to(self.device)
                if self.args.data == 'PEMS' or self.args.data == 'Solar':
                    batch_x_mark, batch_y_mark = None, None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # --- Simplified logic for CasuaLNN ---
                is_causal_model = self.args.model == 'CasuaLNN'
                add_causal_loss = is_causal_model and (
                        (hasattr(self.args, 'lambda_causal') and self.args.lambda_causal > 0) or
                        (hasattr(self.args, 'lambda_l1') and self.args.lambda_l1 > 0)
                )

                total_causal_loss_item, pred_loss_item = 0.0, 0.0
                
                # --- Combined forward pass and loss calculation ---
                def forward_and_loss():
                    nonlocal pred_loss_item, total_causal_loss_item

                    # The model returns the causal graph logits if it's a causal model
                    if is_causal_model and hasattr(self.model, 'causal_graph_logits'):
                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                         G = self.model.causal_graph_logits
                    else:
                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                         G = None

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_pred = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                    loss = criterion(outputs, batch_y_pred)
                    pred_loss_item = loss.item()

                    if add_causal_loss and G is not None:
                        try:
                            loss_causal = torch.tensor(0.0, device=self.device)
                            N = self.args.enc_in
                            
                            # L1 regularization
                            if hasattr(self.args, 'lambda_l1') and self.args.lambda_l1 > 0:
                                l1_loss = self.args.lambda_l1 * torch.norm(G, p=1)
                                if torch.isfinite(l1_loss): loss_causal += l1_loss
                            
                            # DAG constraint (causal loss)
                            if hasattr(self.args, 'lambda_causal') and self.args.lambda_causal > 0:
                                dag_loss_val = torch.tensor(0.0, device=G.device)
                                if self.args.causal_loss_type == 'polynomial_notears':
                                    G_squared = G * G
                                    A = torch.eye(N, device=G.device) + (G_squared / N)
                                    M = torch.matrix_power(A, N)
                                    dag_loss_val = self.args.lambda_causal * (torch.trace(M) - N)
                                elif self.args.causal_loss_type == 'dagma_logdet':
                                    G_squared = G * G
                                    I = torch.eye(N, device=G.device)
                                    logdet = torch.slogdet(I - G_squared).logabsdet
                                    dag_loss_val = self.args.lambda_causal * (-logdet)

                                if torch.isfinite(dag_loss_val): loss_causal += dag_loss_val
                            
                            # Add causal loss to the main loss (mixed loss)
                            if torch.isfinite(loss_causal):
                                loss += loss_causal
                                total_causal_loss_item = loss_causal.item()

                        except Exception as e:
                            print(f"Warning (Epoch {epoch + 1}, Iter {i + 1}): Causal loss calculation failed: {e}. Skipping.")
                    
                    return loss

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = forward_and_loss()
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss = forward_and_loss()
                    loss.backward()
                    model_optim.step()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    log_msg = f"\titers: {i + 1}, epoch: {epoch + 1} | total_loss: {loss.item():.7f} (pred: {pred_loss_item:.7f})"
                    if add_causal_loss and total_causal_loss_item > 0:
                        log_msg += f" | (causal_loss_sum: {total_causal_loss_item:.7f})"
                    print(log_msg)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count, time_now = 0, time.time()

                if self.args.lradj == 'TST':
                    scheduler.step()

            current_epoch_time = time.time() - epoch_time
            print("Epoch: {} cost time: {:.2f}s".format(epoch + 1, current_epoch_time))
            train_loss_avg = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_avg, vali_loss, test_loss))

            with open(log_file_path, 'a') as log_file:
                log_file.write(
                    f"Epoch: {epoch + 1:03d} | Train Loss: {train_loss_avg:.7f} | Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f} | Time: {current_epoch_time:.2f}s\n"
                )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"\nEarly stopping at epoch {epoch + 1}.\n")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            else:
                print('Learning rate updated to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return self.model

    def test(self, setting, test=0):
        # This method is largely generic and does not need major changes.
        if self.setting_with_time is None:
            if test == 1:
                self.setting_with_time = setting
            else:
                print("Warning: setting_with_time is None in test mode.")
                self.setting_with_time = setting

        print(f"Starting testing, experiment ID: {self.setting_with_time}")

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('Loading model...')
            model_path = os.path.join(self.args.checkpoints, self.setting_with_time, 'checkpoint.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Error: Checkpoint file not found at {model_path}")

            state_dict = torch.load(model_path, map_location=self.device)
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError:
                print("Detected DataParallel model, attempting to fix state dict keys...")
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)

        preds, trues, folder_path = [], [], './test_results/' + self.setting_with_time + '/'
        if not os.path.exists(folder_path): os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                if self.args.data == 'PEMS': batch_y = batch_y[:, :, :, 0].float().to(self.device)
                if self.args.data == 'PEMS' or self.args.data == 'Solar': batch_x_mark, batch_y_mark = None, None

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

                if i % 20 == 0:
                    try:
                        input_data = batch_x.detach().cpu().numpy()
                        if input_data.shape[0] > 0 and batch_y.shape[0] > 0:
                            gt = np.concatenate((input_data[0, :, -1], trues[-1][0, :, -1]), axis=0)
                            pd = np.concatenate((input_data[0, :, -1], preds[-1][0, :, -1]), axis=0)
                            visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    except Exception as e:
                        print(f"Warning: Visualization failed (iter {i}): {e}")

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('Test data shape before inverse transform:', preds.shape, trues.shape)

        if self.args.data == 'PEMS':
            B, T, C = preds.shape
            preds = test_data.inverse_transform(preds.reshape(-1, C)).reshape(B, T, C)
            trues = test_data.inverse_transform(trues.reshape(-1, C)).reshape(B, T, C)
        elif hasattr(test_data, 'inverse_transform') and self.args.inverse:
            preds = test_data.inverse_transform(preds)
            trues = test_data.inverse_transform(trues)
        
        print('Test data shape after inverse transform:', preds.shape, trues.shape)

        dtw_val = 'N/A'
        if self.args.use_dtw:
            print("Calculating DTW...")
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x, y = preds[i].reshape(-1, 1), trues[i].reshape(-1, 1)
                if i % 100 == 0: print("dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw_val = np.array(dtw_list).mean()
            print("DTW calculation finished.")

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('--- Final Test Metrics ---')
        print(f'MSE:  {mse:.7f}\nMAE:  {mae:.7f}\nRMSE: {rmse:.7f}\nMAPE: {mape:.7f}\nMSPE: {mspe:.7f}\nDTW:  {dtw_val}')
        print('--------------------------')

        log_file_path = os.path.join(self.args.checkpoints, self.setting_with_time, 'run_log.txt')
        try:
            with open(log_file_path, 'a') as log_file:
                log_file.write("\n--------------- Final Test Results ---------------\n")
                log_file.write(f"Tested at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f'MSE:  {mse:.7f}\nMAE:  {mae:.7f}\nRMSE: {rmse:.7f}\nMAPE: {mape:.7f}\nMSPE: {mspe:.7f}\nDTW:  {dtw_val}\n')
                log_file.write("--------------------------------------------------\n")
        except FileNotFoundError:
            print(f"Warning: Log file not found at {log_file_path}. Skipping test metric logging.")

        with open("result_long_term_forecast.txt", 'a') as f:
            f.write(self.setting_with_time + "  \n")
            f.write(f"mse:{mse:.7f}, mae:{mae:.7f}, rmse:{rmse:.7f}, mape:{mape:.7f}, mspe:{mspe:.7f}, dtw:{dtw_val}\n\n")

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        print("Test results saved.")

        return