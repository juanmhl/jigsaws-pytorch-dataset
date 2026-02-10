from typing import Any

import torch


class BaseScaler:
    """
    Base class for scalers.
    """
    def fit(self, data: torch.Tensor) -> "BaseScaler":
        """
        Fit the scaler to the data.

        Args:
            data: The data to fit the scaler to.

        Returns:
            self: The fitted scaler for chaining.
        """
        raise NotImplementedError

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform the data using the fitted scaler.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """
        raise NotImplementedError

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the scaling transformation.

        Args:
            data: The scaled data to inverse transform.

        Returns:
            The original unscaled data.
        """
        raise NotImplementedError

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Fit the scaler to the data and then transform it.

        Args:
            data: The data to fit and transform.

        Returns:
            The transformed data.
        """
        self.fit(data)
        return self.transform(data)

    def get_params(self) -> dict[str, Any]:
        """
        Get scaler parameters for saving.

        Returns:
            Dictionary of scaler parameters.
        """
        raise NotImplementedError

    def set_params(self, params: dict[str, Any]) -> None:
        """
        Set scaler parameters from loaded values.

        Args:
            params: Dictionary of scaler parameters.
        """
        raise NotImplementedError

    def save(self, filepath: str) -> None:
        """
        Save scaler parameters to .pt file.

        Args:
            filepath: Path to save the scaler parameters.
        """
        params = self.get_params()
        params["_scaler_class"] = self.__class__.__name__
        torch.save(params, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BaseScaler":
        """
        Load scaler from .pt file.

        Args:
            filepath: Path to the saved scaler file.

        Returns:
            Loaded scaler instance.
        """
        params = torch.load(filepath, weights_only=False)
        class_name = params.pop("_scaler_class", None)

        # Determine which class to instantiate
        if class_name == "StandardScaler":
            scaler = StandardScaler()
        elif class_name == "MinMaxScaler":
            feature_range = params.get("feature_range", (0, 1))
            scaler = MinMaxScaler(feature_range=feature_range)
        else:
            # Default to cls if loading directly from subclass
            scaler = cls() if cls != BaseScaler else StandardScaler()

        scaler.set_params(params)
        return scaler

class StandardScaler(BaseScaler):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor) -> "StandardScaler":
        """
        Compute the mean and std to be used for later scaling.

        Args:
            data: The data to fit the scaler to. Shape: (n_samples, n_features)

        Returns:
            self: The fitted scaler for chaining.
        """
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)
        # Add a small epsilon to std to avoid division by zero
        self.std[self.std == 0] = 1e-7
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Perform standardization by centering and scaling.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() before transform().")
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the standardization transformation.

        Args:
            data: The scaled data to inverse transform.

        Returns:
            The original unscaled data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() before inverse_transform().")
        return data * self.std + self.mean

    def get_params(self) -> dict[str, Any]:
        """Get scaler parameters for saving."""
        return {
            "mean": self.mean,
            "std": self.std,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        """Set scaler parameters from loaded values."""
        self.mean = params.get("mean")
        self.std = params.get("std")

class MinMaxScaler(BaseScaler):
    """
    Transforms features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g., between zero and one.
    """
    def __init__(self, feature_range: tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.scale = None
        self.min_val_transform = None

    def fit(self, data: torch.Tensor) -> "MinMaxScaler":
        """
        Compute the minimum and scale to be used for later scaling.

        Args:
            data: The data to fit the scaler to. Shape: (n_samples, n_features)

        Returns:
            self: The fitted scaler for chaining.
        """
        data_min = torch.min(data, dim=0)[0]
        data_max = torch.max(data, dim=0)[0]

        feature_min, feature_max = self.feature_range

        self.min = data_min
        data_range = data_max - data_min
        data_range[data_range == 0] = 1e-7  # Avoid division by zero

        scale = (feature_max - feature_min) / data_range
        self.scale = scale
        self.min_val_transform = feature_min - data_min * scale
        return self

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Scale features of data according to feature_range.

        Args:
            data: The data to transform.

        Returns:
            The transformed data.
        """
        if self.min is None or self.scale is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() before transform().")

        return data * self.scale + self.min_val_transform

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Reverse the min-max scaling transformation.

        Args:
            data: The scaled data to inverse transform.

        Returns:
            The original unscaled data.
        """
        if self.min is None or self.scale is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() before inverse_transform().")

        return (data - self.min_val_transform) / self.scale

    def get_params(self) -> dict[str, Any]:
        """Get scaler parameters for saving."""
        return {
            "feature_range": self.feature_range,
            "min": self.min,
            "scale": self.scale,
            "min_val_transform": self.min_val_transform,
        }

    def set_params(self, params: dict[str, Any]) -> None:
        """Set scaler parameters from loaded values."""
        self.feature_range = params.get("feature_range", (0, 1))
        self.min = params.get("min")
        self.scale = params.get("scale")
        self.min_val_transform = params.get("min_val_transform")
