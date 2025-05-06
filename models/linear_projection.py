import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjection(nn.Module):
    """
    Linear Projection module for dimensionality reduction with L1 regularization.
    Reduces visual input dimensionality by 94% while preserving essential articulatory information.
    """

    def __init__(self, input_dim=24576, output_dim=256, l1_lambda=0.01):
        """
        Initialize the Linear Projection module.

        Args:
            input_dim: Input dimension (default: 128x64x3=24576)
            output_dim: Output dimension (default: 256)
            l1_lambda: L1 regularization coefficient (default: 0.01)
        """
        super(LinearProjection, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l1_lambda = l1_lambda

        # Learnable weight matrix and bias
        self.weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))

        # Initialize parameters using He initialization
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.zeros_(self.bias)

    def forward(self, x):
        """
        Forward pass of the Linear Projection module.

        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
                or [batch_size, sequence_length, channels, height, width]

        Returns:
            Projected tensor of shape [batch_size, output_dim] or
            [batch_size, sequence_length, output_dim]
        """
        # Handle both 4D and 5D inputs (single frame or sequence)
        original_shape = x.shape

        if len(original_shape) == 5:  # [batch, seq_len, channels, height, width]
            batch_size, seq_len = original_shape[0], original_shape[1]
            # Reshape to [batch*seq_len, -1]
            x = x.reshape(batch_size * seq_len, -1)
            # Apply linear projection
            output = F.linear(x, self.weight, self.bias)
            # Apply SELU activation (as mentioned in the paper)
            output = F.selu(output)
            # Reshape back to [batch, seq_len, output_dim]
            output = output.reshape(batch_size, seq_len, self.output_dim)
        else:  # [batch, channels, height, width]
            # Flatten the input
            x = x.reshape(original_shape[0], -1)
            # Apply linear projection
            output = F.linear(x, self.weight, self.bias)
            # Apply SELU activation
            output = F.selu(output)

        return output

    def get_l1_regularization(self):
        """Calculate L1 regularization term for the weights."""
        return self.l1_lambda * torch.sum(torch.abs(self.weight))