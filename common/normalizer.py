import numpy as np
import torch
from common import util


class StandardNormalizer(object):
    def __init__(self):
        self.tot_count = 0
    
    def reset(self):
        self.mean = None
        self.var = None
        self.tot_count = 0

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        if isinstance(data, torch.Tensor):
            self.mean = torch.mean(data, dim=0, keepdims=True).to(util.device)
            self.var = torch.var(data, dim=0, keepdims=True).to(util.device)
        elif isinstance(data, np.ndarray):
            self.mean = np.mean(data, axis=0, keepdims=True)
            self.var = np.var(data, axis=0, keepdims=True)
        self.var[self.var < 1e-12] = 1.0
        self.tot_count = len(data)

    def update(self, samples):
        sample_count = len(samples)

        # Initialize on first update
        if self.tot_count == 0:
            dim = samples.shape[1]
            if isinstance(samples, torch.Tensor):
                self.mean = torch.zeros((1, dim), dtype=torch.float32, device=samples.device)
                self.var = torch.ones((1, dim), dtype=torch.float32, device=samples.device)
            elif isinstance(samples, np.ndarray):
                self.mean = np.zeros((1, dim), dtype=np.float32)
                self.var = np.ones((1, dim), dtype=np.float32)

        # Compute sample statistics
        if isinstance(samples, torch.Tensor):
            sample_mean = torch.mean(samples, dim=0, keepdims=True)
            sample_var = torch.var(samples, dim=0, keepdims=True)

            # Convert existing mean/var to torch tensors on the same device if needed
            if not torch.is_tensor(self.mean):
                self.mean = torch.as_tensor(self.mean, device=samples.device, dtype=torch.float32)
                self.var = torch.as_tensor(self.var, device=samples.device, dtype=torch.float32)
            else:
                # Ensure same device
                if self.mean.device != samples.device:
                    self.mean = self.mean.to(samples.device)
                    self.var = self.var.to(samples.device)

        elif isinstance(samples, np.ndarray):
            sample_mean = np.mean(samples, axis=0, keepdims=True)
            sample_var = np.var(samples, axis=0, keepdims=True)

            # Convert existing mean/var to numpy if needed
            if torch.is_tensor(self.mean):
                self.mean = self.mean.detach().cpu().numpy()
                self.var = self.var.detach().cpu().numpy()

        # Incremental update formulas
        delta_mean = sample_mean - self.mean

        new_mean = self.mean + delta_mean * sample_count / (self.tot_count + sample_count)
        prev_var = self.var * self.tot_count
        sample_var_scaled = sample_var * sample_count
        new_var = prev_var + sample_var_scaled + delta_mean * delta_mean * self.tot_count * sample_count / (self.tot_count + sample_count)
        new_var = new_var / (self.tot_count + sample_count)

        # Convert to appropriate type and ensure float32
        if isinstance(samples, torch.Tensor):
            self.mean = new_mean.to(torch.float32)
            self.var = new_var.to(torch.float32)
        elif isinstance(samples, np.ndarray):
            self.mean = new_mean.astype(np.float32)
            self.var = new_var.astype(np.float32)

        # Prevent numerical instability
        self.var[self.var < 1e-12] = 1.0
        self.tot_count += sample_count

    def transform(self, data):
        # return data
        if self.mean is None or self.var is None:
            return data

        # Torch path
        if torch.is_tensor(data):
            mean = self.mean
            var  = self.var

            # mean/var might be numpy/list OR torch tensor (cpu/cuda)
            if torch.is_tensor(mean):
                mean = mean.to(device=data.device, dtype=data.dtype)
            else:
                mean = torch.as_tensor(mean, device=data.device, dtype=data.dtype)

            if torch.is_tensor(var):
                var = var.to(device=data.device, dtype=data.dtype)
            else:
                var = torch.as_tensor(var, device=data.device, dtype=data.dtype)

            return ((data - mean) / torch.sqrt(var + 1e-8)).float()

        # Numpy path
        elif isinstance(data, np.ndarray):
            mean = self.mean
            var  = self.var

            # mean/var might be torch tensors; convert safely to numpy
            if torch.is_tensor(mean):
                mean = mean.detach().cpu().numpy()
            else:
                mean = np.asarray(mean)

            if torch.is_tensor(var):
                var = var.detach().cpu().numpy()
            else:
                var = np.asarray(var)

            return ((data - mean) / np.sqrt(var + 1e-8)).astype(np.float32)

        else:
            raise TypeError(f"Unsupported data type: {type(data)}")


    def inverse_transform(self, data):
        # return data

        if self.mean is None or self.var is None:
            print("Warning: Inverse transform called before fitting normalizer. Returning data unchanged.")
            return data
        if isinstance(data, torch.Tensor):
            return data * torch.sqrt(torch.tensor(self.var).to(data.device)) + torch.tensor(self.mean).to(data.device)
        elif isinstance(data, np.ndarray):
            return data * np.sqrt(np.array(self.var)) + np.array(self.mean)