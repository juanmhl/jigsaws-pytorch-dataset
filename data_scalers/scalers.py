import torch

class BaseScaler:
    """
    Base class for scalers.
    """
    def fit(self, data):
        """
        Fit the scaler to the data.
        
        Args:
            data (torch.Tensor): The data to fit the scaler to.
        """
        raise NotImplementedError

    def transform(self, data):
        """
        Transform the data using the fitted scaler.
        
        Args:
            data (torch.Tensor): The data to transform.
        
        Returns:
            torch.Tensor: The transformed data.
        """
        raise NotImplementedError

    def fit_transform(self, data):
        """
        Fit the scaler to the data and then transform it.
        
        Args:
            data (torch.Tensor): The data to fit and transform.
        
        Returns:
            torch.Tensor: The transformed data.
        """
        self.fit(data)
        return self.transform(data)

class StandardScaler(BaseScaler):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """
        Compute the mean and std to be used for later scaling.
        
        Args:
            data (torch.Tensor): The data to fit the scaler to. 
                                 Shape: (n_samples, n_features)
        """
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)
        # Add a small epsilon to std to avoid division by zero
        self.std[self.std == 0] = 1e-7

    def transform(self, data):
        """
        Perform standardization by centering and scaling.
        
        Args:
            data (torch.Tensor): The data to transform.
        
        Returns:
            torch.Tensor: The transformed data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() before transform().")
        return (data - self.mean) / self.std

class MinMaxScaler(BaseScaler):
    """
    Transforms features by scaling each feature to a given range.
    
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g., between zero and one.
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.scale = None

    def fit(self, data):
        """
        Compute the minimum and scale to be used for later scaling.
        
        Args:
            data (torch.Tensor): The data to fit the scaler to.
                                 Shape: (n_samples, n_features)
        """
        data_min = torch.min(data, dim=0)[0]
        data_max = torch.max(data, dim=0)[0]
        
        feature_min, feature_max = self.feature_range
        
        self.min = data_min
        data_range = data_max - data_min
        data_range[data_range == 0] = 1e-7 # Avoid division by zero
        
        scale = (feature_max - feature_min) / data_range
        self.scale = scale
        self.min_val_transform = feature_min - data_min * scale

    def transform(self, data):
        """
        Scale features of data according to feature_range.
        
        Args:
            data (torch.Tensor): The data to transform.
        
        Returns:
            torch.Tensor: The transformed data.
        """
        if self.min is None or self.scale is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() before transform().")
        
        return data * self.scale + self.min_val_transform
