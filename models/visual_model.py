import torch
import torch.nn as nn
import torch.nn.functional as F

# Import des modules LinearProjection et CapsuleNetwork
from models.linear_projection import LinearProjection  # Assure-toi que ce fichier existe dans models/
from models.capsnet import CapsuleNetwork  # Assure-toi que ce fichier existe dans models/

class VisualModel(nn.Module):
    """
    Visual front-end for AVSR that combines Linear Projection and CapsNet.
    Processes lip region images to extract articulatory features.
    """

    def __init__(self, input_dim=24576, hidden_dim=256, output_dim=512,
                 l1_lambda=0.01, capsnet_routing_iters=3):
        """
        Initialize the Visual Front-End module.

        Args:
            input_dim: Input dimension (default: 128x64x3=24576) (Image size 128x64 with 3 color channels)
            hidden_dim: Hidden dimension after linear projection (default: 256)
            output_dim: Output dimension (default: 512)
            l1_lambda: L1 regularization coefficient (default: 0.01)
            capsnet_routing_iters: Number of routing iterations for CapsNet (default: 3)
        """
        super(VisualFrontEnd, self).__init__()

        # Linear Projection for dimensionality reduction (input_dim -> hidden_dim)
        self.linear_projection = LinearProjection(
            input_dim=input_dim,
            output_dim=hidden_dim,
            l1_lambda=l1_lambda
        )

        # Capsule Network (CapsNet) for capturing hierarchical spatial relationships in the image
        self.capsnet = CapsuleNetwork(
            input_dim=hidden_dim,  # Output of linear projection
            primary_caps=32,       # Number of primary capsules
            primary_dims=8,        # Dimensions of primary capsules
            digit_caps=16,         # Number of digit capsules
            digit_dims=16,         # Dimensions of digit capsules
            routing_iters=capsnet_routing_iters  # Number of routing iterations
        )

        # Final projection to match the desired output dimension (output_dim)
        self.output_projection = nn.Linear(16 * 16, output_dim)  # CapsNet output size is 16*16

        # Dropout for regularization (dropout rate is 20%)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """
        Forward pass of the Visual Front-End.

        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]
                or [batch_size, channels, height, width] for a single frame

        Returns:
            Visual features of shape [batch_size, seq_len, output_dim]
            or [batch_size, output_dim] for single frame
        """
        # Check if the input is a single frame or a sequence
        is_single_frame = len(x.shape) == 4

        if is_single_frame:
            # Add sequence dimension if the input is a single frame (for consistency in batch processing)
            x = x.unsqueeze(1)

        # Linear projection to reduce dimensionality from input_dim to hidden_dim
        x = self.linear_projection(x)  # Output shape: [batch, seq_len, hidden_dim]

        # Apply Capsule Network for learning spatial features
        x = self.capsnet(x)  # Output shape: [batch, seq_len, digit_caps * digit_dims]

        # Apply dropout for regularization
        x = self.dropout(x)

        # Final linear projection to match output dimension
        x = self.output_projection(x)  # Output shape: [batch, seq_len, output_dim]

        # If input was a single frame, remove sequence dimension and return output of shape [batch, output_dim]
        if is_single_frame:
            x = x.squeeze(1)

        return x

    def get_l1_regularization(self):
        """Return the L1 regularization term from the linear projection."""
        return self.linear_projection.get_l1_regularization()

    def get_capsule_norms(self, x):
        """
        Get capsule norms for visualization and interpretation.

        Args:
            x: Input tensor of shape [batch_size, seq_len, channels, height, width]

        Returns:
            Norms of the capsules [batch_size, seq_len, num_capsules]
        """
        # Check if single frame or sequence
        is_single_frame = len(x.shape) == 4

        if is_single_frame:
            # Add sequence dimension if input is a single frame
            x = x.unsqueeze(1)

        # Apply linear projection to reduce dimensionality
        x = self.linear_projection(x)

        # Get capsule norms from CapsNet
        norms = self.capsnet.get_capsule_norms(x)

        # If input was a single frame, remove the sequence dimension
        if is_single_frame:
            norms = norms.squeeze(1)

        return norms
