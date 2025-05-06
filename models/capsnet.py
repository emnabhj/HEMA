import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimaryCaps(nn.Module):
    """
    Primary Capsule Layer: Convert traditional convolutional features to capsule format.
    """

    def __init__(self, in_channels, out_caps, out_dim, kernel_size=3, stride=1):
        super(PrimaryCaps, self).__init__()
        self.out_caps = out_caps
        self.out_dim = out_dim

        # Convolutional layer to generate capsule outputs
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_caps * out_dim,
            kernel_size=kernel_size,
            stride=stride
        )

    def forward(self, x):
        """
        Forward pass of the Primary Capsule layer.

        Args:
            x: Input tensor of shape [batch_size, in_channels, sequence_length]

        Returns:
            Capsule output of shape [batch_size, out_caps, out_dim, sequence_length']
        """
        batch_size = x.size(0)

        # Apply convolution
        out = self.conv(x)  # [batch, out_caps*out_dim, seq_len']
        seq_len = out.size(2)

        # Reshape to capsule format
        out = out.view(batch_size, self.out_caps, self.out_dim, seq_len)

        # Squash function to ensure output vectors have norm between 0 and 1
        out = self.squash(out)

        return out

    def squash(self, x, dim=-2):
        """
        Squash function to normalize capsule vectors.

        Args:
            x: Input tensor
            dim: Dimension along which to compute the norm

        Returns:
            Normalized tensor
        """
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)


class DynamicRouting(nn.Module):
    """
    Dynamic Routing layer between capsules as described in the CapsNet paper.
    Implements iterative routing-by-agreement algorithm.
    """

    def __init__(self, in_caps, in_dim, out_caps, out_dim, routing_iters=3):
        super(DynamicRouting, self).__init__()
        self.in_caps = in_caps
        self.in_dim = in_dim
        self.out_caps = out_caps
        self.out_dim = out_dim
        self.routing_iters = routing_iters

        # Transformation matrices
        self.W = nn.Parameter(torch.randn(out_caps, in_caps, out_dim, in_dim))

    def forward(self, x):
        """
        Forward pass with dynamic routing algorithm.

        Args:
            x: Input tensor of shape [batch_size, in_caps, in_dim, seq_len]

        Returns:
            Capsule output after routing of shape [batch_size, out_caps, out_dim, seq_len]
        """
        batch_size = x.size(0)
        seq_len = x.size(3)

        # Reshape x for matrix multiplication
        x = x.unsqueeze(1)  # [batch, 1, in_caps, in_dim, seq_len]

        # Expand W to match batch size
        W = self.W.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # [batch, out_caps, in_caps, out_dim, in_dim]

        # Transform input capsules
        # Transpose x to [batch, 1, in_caps, seq_len, in_dim]
        x_transposed = x.transpose(-1, -2)

        # Matrix multiplication [batch, out_caps, in_caps, out_dim, in_dim] x [batch, 1, in_caps, seq_len, in_dim]
        # -> [batch, out_caps, in_caps, out_dim, seq_len]
        u_hat = torch.matmul(W, x_transposed.unsqueeze(1).expand(-1, self.out_caps, -1, -1, -1))
        u_hat = u_hat.permute(0, 1, 2, 4, 3)  # [batch, out_caps, in_caps, seq_len, out_dim]

        # Detach u_hat during routing iterations to prevent backpropagation through routing
        u_hat_detached = u_hat.detach()

        # Initialize routing logits
        b = torch.zeros(batch_size, self.out_caps, self.in_caps, seq_len).to(x.device)

        # Iterative routing
        for i in range(self.routing_iters - 1):
            # Calculate coupling coefficients
            c = F.softmax(b, dim=1)  # [batch, out_caps, in_caps, seq_len]

            # Weight prediction by routing coefficients
            # [batch, out_caps, in_caps, seq_len, 1] * [batch, out_caps, in_caps, seq_len, out_dim]
            s = (c.unsqueeze(-1) * u_hat_detached).sum(dim=2)  # [batch, out_caps, seq_len, out_dim]

            # Apply squash
            v = self.squash(s, dim=-1)  # [batch, out_caps, seq_len, out_dim]

            # Update routing logits
            # Agreement: [batch, out_caps, in_caps, seq_len, out_dim] * [batch, out_caps, 1, seq_len, out_dim]
            agreement = torch.matmul(u_hat_detached.transpose(-1, -2), v.unsqueeze(2).transpose(-1, -2))
            b = b + agreement.squeeze(-1)

        # Final iteration with non-detached u_hat
        c = F.softmax(b, dim=1)
        s = (c.unsqueeze(-1) * u_hat).sum(dim=2)  # [batch, out_caps, seq_len, out_dim]
        v = self.squash(s, dim=-1)  # [batch, out_caps, seq_len, out_dim]

        # Transpose back to match expected output format
        v = v.transpose(-1, -2)  # [batch, out_caps, out_dim, seq_len]

        return v

    def squash(self, x, dim=-1):
        """
        Squash function to normalize capsule vectors.

        Args:
            x: Input tensor
            dim: Dimension along which to compute the norm

        Returns:
            Normalized tensor
        """
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)


class CapsuleNetwork(nn.Module):
    """
    Complete Capsule Network for visual speech recognition.
    """

    def __init__(self, input_dim=256, primary_caps=32, primary_dims=8,
                 digit_caps=16, digit_dims=16, routing_iters=3):
        super(CapsuleNetwork, self).__init__()

        # Initial convolution to prepare input for primary capsules
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(256)

        # Primary capsule layer
        self.primary_caps = PrimaryCaps(
            in_channels=256,
            out_caps=primary_caps,
            out_dim=primary_dims,
            kernel_size=5,
            stride=1
        )

        # Dynamic routing capsule layer
        self.digit_caps = DynamicRouting(
            in_caps=primary_caps,
            in_dim=primary_dims,
            out_caps=digit_caps,
            out_dim=digit_dims,
            routing_iters=routing_iters
        )

    def forward(self, x):
        """
        Forward pass of the Capsule Network.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]

        Returns:
            Capsule output of shape [batch_size, seq_len, digit_caps * digit_dims]
        """
        batch_size, seq_len, input_dim = x.size()

        # Reshape for 1D convolution: [batch*seq_len, input_dim, 1]
        x = x.reshape(-1, input_dim, 1)

        # Initial convolution
        x = self.conv1(x)  # [batch*seq_len, 256, 1]
        x = self.bn1(x)
        x = F.relu(x)

        # Primary capsules
        x = self.primary_caps(x)  # [batch*seq_len, primary_caps, primary_dims, 1]

        # Digit capsules with dynamic routing
        x = self.digit_caps(x)  # [batch*seq_len, digit_caps, digit_dims, 1]

        # Reshape to remove the last dimension which is 1
        x = x.squeeze(-1)  # [batch*seq_len, digit_caps, digit_dims]

        # Reshape back to [batch, seq_len, digit_caps * digit_dims]
        x = x.transpose(1, 2)  # [batch*seq_len, digit_dims, digit_caps]
        x = x.reshape(batch_size, seq_len, -1)  # [batch, seq_len, digit_dims * digit_caps]

        return x

    def get_capsule_norms(self, x):
        """
        Computes the norms of capsules for visualization and interpretation.

        Args:
            x: Input tensor

        Returns:
            Norms of the digit capsules
        """
        batch_size, seq_len, input_dim = x.size()

        # Process through the network
        x = x.reshape(-1, input_dim, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.primary_caps(x)
        x = self.digit_caps(x)  # [batch*seq_len, digit_caps, digit_dims, 1]

        # Compute the norm of each capsule
        norms = torch.norm(x, dim=2, keepdim=False)  # [batch*seq_len, digit_caps, 1]
        norms = norms.squeeze(-1).reshape(batch_size, seq_len, -1)  # [batch, seq_len, digit_caps]

        return norms