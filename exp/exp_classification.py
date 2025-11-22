from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Classification(Exp_Basic):
    """
    Experiment class for Time Series Classification tasks.

    This class handles the training and evaluation pipeline for models
    designed to classify time series data. It includes dynamic adjustment
    of model input parameters based on dataset characteristics and
    evaluation using accuracy.
    """

    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        """
        Builds and returns the model instance, dynamically adjusting input
        parameters based on the training and test data characteristics.
        """
        # Load a small sample of data to infer necessary model input dimensions
        train_data, _ = self._get_data(flag='TRAIN')
        test_data, _ = self._get_data(flag='TEST')
        
        # Dynamically set sequence length, input feature dimension, and number of classes
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0 # Prediction length is not applicable for classification
        self.args.enc_in = train_data.feature_df.shape[1] # Number of features/variables
        self.args.num_class = len(train_data.class_names) # Number of unique classes

        # Instantiate the model from the model dictionary defined in Exp_Basic
        model = self.model_dict[self.args.model].Model(self.args).float()
        
        # Wrap model with DataParallel if multi-GPU is enabled
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """
        Retrieves the dataset and data loader for a given flag (e.g., 'TRAIN', 'TEST').
        """
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """
        Selects and returns the RAdam optimizer for model training.
        RAdam is chosen for its adaptive learning rate capabilities, often
        providing better convergence for deep learning models.
        """
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """
        Selects and returns the CrossEntropyLoss criterion, suitable for
        multi-class classification problems.
        """
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        Performs validation on the given data loader for classification tasks.
        Evaluates average loss and accuracy.
        """
        total_loss = []
        preds = []
        trues = []
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device) # Mask for handling variable sequence lengths
                label = label.to(self.device)

                # Model forward pass (x_mark and y_mark are not used in this task, hence None)
                outputs = self.model(batch_x, padding_mask, None, None)

                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze()) # Compute loss with ground truth labels
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        # Concatenate predictions and true labels from all batches
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        
        # Convert raw model outputs (logits) to probabilities and then to predicted class indices
        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes) estimated probabilities
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) integer class index
        trues = trues.flatten().cpu().numpy() # Flatten true labels for accuracy calculation
        
        accuracy = cal_accuracy(predictions, trues) # Calculate accuracy using utility function

        self.model.train() # Set model back to training mode
        return total_loss, accuracy

    def train(self, setting):
        """
        Conducts the training process for the model on classification tasks.
        """
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST') # Using TEST for validation as per common practice in some setups
        test_data, test_loader = self._get_data(flag='TEST')

        # Create directory for saving checkpoints
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        # Early stopping monitors a metric. For accuracy, we monitor -accuracy to detect increases.
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # --- Training Loop ---
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train() # Set model to training mode
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad() # Zero gradients for each batch

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device) # Mask for handling variable sequence lengths
                label = label.to(self.device)

                # Model forward pass
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, label.long().squeeze(-1)) # Compute loss

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
                # Clip gradients to prevent exploding gradients, improving training stability
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step() # Update model parameters

            # End of epoch logging and validation
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion) # Evaluate on test set for final reporting

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.3f} "
                  f"Vali Loss: {vali_loss:.3f} Vali Acc: {val_accuracy:.3f} "
                  f"Test Loss: {test_loss:.3f} Test Acc: {test_accuracy:.3f}")
            
            # Early stopping check: monitor validation accuracy (negative loss for max accuracy)
            early_stopping(-val_accuracy, self.model, path)
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
        Evaluates the trained model on the test set for classification tasks.
        """
        test_data, test_loader = self._get_data(flag='TEST')
        if test: # If running in test-only mode, load the pre-trained model
            print('Loading model for testing...')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Error: Checkpoint file not found at {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        preds = []
        trues = []
        # Create folder for test results (e.g., visualizations if any)
        folder_path = './test_results/' + setting + '/' 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                # Model forward pass
                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        # Concatenate predictions and true labels from all batches
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('Test data shape (predictions, trues):', preds.shape, trues.shape)

        # Convert raw model outputs (logits) to probabilities and then to predicted class indices
        probs = torch.nn.functional.softmax(preds, dim=1)  # (total_samples, num_classes)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,)
        trues = trues.flatten().cpu().numpy() # Flatten true labels
        
        accuracy = cal_accuracy(predictions, trues) # Calculate accuracy

        # --- Save results ---
        # Use a distinct folder for classification results
        classification_results_folder = './results/' + setting + '/' 
        if not os.path.exists(classification_results_folder):
            os.makedirs(classification_results_folder)

        print(f'Test Accuracy: {accuracy:.4f}')
        
        # Log results to a summary file
        file_name='result_classification.txt'
        with open(os.path.join(classification_results_folder, file_name), 'a') as f:
            f.write(setting + "  \n")
            f.write(f'accuracy:{accuracy:.4f}')
            f.write('\n\n')
        
        # Note: For classification, typically no numpy files are saved unless detailed
        # predictions/confidences are needed for further analysis.
        return