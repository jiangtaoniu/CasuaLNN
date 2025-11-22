import os
import torch
from models import CasuaLNN


class Exp_Basic(object):
    """
    Abstract base class for all experiment pipelines.

    This class provides common functionalities such as argument handling,
    device acquisition, and a registry for models. Subclasses are expected
    to implement specific training, validation, and testing logic.
    """

    def __init__(self, args):
        """
        Initializes the Exp_Basic instance.

        Args:
            args: An argparse.Namespace object containing all experiment configurations.
        """
        self.args = args
        # Dictionary to store available model classes, allowing dynamic model instantiation.
        self.model_dict = {
            'CasuaLNN': CasuaLNN, # Default model entry
        }
        # Dynamically load other models if specified and available
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm (e.g., pip install mamba_ssm)')
            from models import Mamba # Assuming Mamba model is defined in models/Mamba.py
            self.model_dict['Mamba'] = Mamba

        self.device = self._acquire_device() # Configure and acquire the appropriate computing device
        self.model = self._build_model().to(self.device) # Build and move model to device

    def _build_model(self):
        """
        Abstract method to build the specific model for the experiment.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _build_model() method.")

    def _acquire_device(self):
        """
        Configures and returns the appropriate torch.device (CPU, CUDA, or MPS).
        Sets CUDA_VISIBLE_DEVICES environment variable if using CUDA.
        """
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            # Set CUDA_VISIBLE_DEVICES for single or multi-GPU setup
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Using GPU: cuda:{self.args.gpu}')
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            # Use Apple's Metal Performance Shaders (MPS) for GPU acceleration
            device = torch.device('mps')
            print('Using GPU: mps')
        else:
            # Fallback to CPU if no GPU is requested or available
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _get_data(self, flag):
        """
        Abstract method to retrieve the dataset and data loader.
        This method must be implemented by subclasses to provide data for training, validation, or testing.
        """
        raise NotImplementedError("Subclasses must implement _get_data() method.")

    def vali(self, vali_data, vali_loader, criterion):
        """
        Abstract method to perform validation.
        This method must be implemented by subclasses to evaluate model performance on a validation set.
        """
        raise NotImplementedError("Subclasses must implement vali() method.")

    def train(self, setting):
        """
        Abstract method to perform model training.
        This method must be implemented by subclasses to define the training loop.
        """
        raise NotImplementedError("Subclasses must implement train() method.")

    def test(self, setting, test=0):
        """
        Abstract method to perform model testing.
        This method must be implemented by subclasses to evaluate model performance on a test set.
        """
        raise NotImplementedError("Subclasses must implement test() method.")