import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.reliability import estimate_audio_snr, compute_visual_sharpness


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention fusion module that dynamically weights modalities
    based on their relative importance.
    """

    def __init__(self, audio_dim, visual_dim, hidden_dim=16):
        super(CrossModalAttention, self).__init__()

        # Projection matrices for query, key, value
        self.W_q = nn.Linear(visual_dim, hidden_dim)
        self.W_k = nn.Linear(audio_dim, hidden_dim)
        self.W_v_audio = nn.Linear(audio_dim, hidden_dim)
        self.W_v_visual = nn.Linear(visual_dim, hidden_dim)

        # Scaling factor for attention
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).cuda() if torch.cuda.is_available() else torch.sqrt(torch.FloatTensor([hidden_dim]))

        # Final projection layer
        self.fc_out = nn.Linear(hidden_dim * 2, hidden_dim)

        # Reliability estimators
        self.audio_reliability = estimate_audio_snr()
        self.visual_reliability = compute_visual_sharpness()

    def forward(self, visual_features, audio_features, raw_audio=None, raw_visual=None):
        """
        Args:
            visual_features: Features from visual model [batch_size, seq_len, visual_dim]
            audio_features: Features from audio model [batch_size, seq_len, audio_dim]
            raw_audio: Raw audio input for reliability estimation
            raw_visual: Raw visual input for reliability estimation
        """
        batch_size = visual_features.shape[0]

        # Project inputs to the same dimension space
        Q_v = self.W_q(visual_features)  # [batch_size, seq_len, hidden_dim]
        K_a = self.W_k(audio_features)  # [batch_size, seq_len, hidden_dim]
        V_v = self.W_v_visual(visual_features)  # [batch_size, seq_len, hidden_dim]
        V_a = self.W_v_audio(audio_features)  # [batch_size, seq_len, hidden_dim]

        # Compute attention scores
        # [batch_size, seq_len, seq_len]
        energy = torch.matmul(Q_v, K_a.permute(0, 2, 1)) / self.scale

        # Apply softmax for attention weights
        attention = F.softmax(energy, dim=-1)

        # Estimate reliability factors if raw inputs are provided
        if raw_audio is not None and raw_visual is not None:
            audio_reliability = self.audio_reliability(raw_audio)  # [batch_size, 1]
            visual_reliability = self.visual_reliability(raw_visual)  # [batch_size, 1]

            # Normalize reliability scores
            reliability_sum = audio_reliability + visual_reliability
            audio_weight = audio_reliability / reliability_sum
            visual_weight = visual_reliability / reliability_sum

            # Apply reliability weights
            attention = attention * audio_weight.view(batch_size, 1, 1)

        # Apply attention to combine features
        context = torch.matmul(attention, V_a)  # [batch_size, seq_len, hidden_dim]

        # Concatenate context with visual value features
        output = torch.cat((context, V_v), dim=2)  # [batch_size, seq_len, hidden_dim*2]

        # Final projection
        output = self.fc_out(output)  # [batch_size, seq_len, hidden_dim]

        return output


class DynamicRoutingFusion(nn.Module):
    """
    Fusion module based on Dynamic Routing principle from CapsNet for more robust
    multimodal integration.
    """

    def __init__(self, audio_dim, visual_dim, output_dim, routing_iterations=3):
        super(DynamicRoutingFusion, self).__init__()

        self.audio_projection = nn.Linear(audio_dim, output_dim)
        self.visual_projection = nn.Linear(visual_dim, output_dim)
        self.routing_iterations = routing_iterations

    def squash(self, x, dim=-1):
        """Apply squashing function to capsule outputs."""
        squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * x / torch.sqrt(squared_norm + 1e-8)

    def forward(self, visual_features, audio_features):
        """
        Combine audio and visual features using dynamic routing.

        Args:
            visual_features: [batch_size, seq_len, visual_dim]
            audio_features: [batch_size, seq_len, audio_dim]
        """
        # Project to same dimension
        u_visual = self.visual_projection(visual_features)  # [batch_size, seq_len, output_dim]
        u_audio = self.audio_projection(audio_features)  # [batch_size, seq_len, output_dim]

        # Initialize routing logits
        batch_size, seq_len = visual_features.shape[0], visual_features.shape[1]
        b = torch.zeros(batch_size, seq_len, 2, 1).to(u_visual.device)  # 2 modalities

        # Dynamic routing iterations
        for i in range(self.routing_iterations):
            # Compute softmax of routing logits
            c = F.softmax(b, dim=2)  # [batch_size, seq_len, 2, 1]

            # Weight features by routing coefficients
            c_expanded_visual = c[:, :, 0].unsqueeze(-1)  # [batch_size, seq_len, 1, 1]
            c_expanded_audio = c[:, :, 1].unsqueeze(-1)  # [batch_size, seq_len, 1, 1]

            # Weighted sum of features
            s = (c_expanded_visual * u_visual.unsqueeze(2) +
                 c_expanded_audio * u_audio.unsqueeze(2))  # [batch_size, seq_len, 1, output_dim]

            # Apply squashing
            v = self.squash(s, dim=-1)  # [batch_size, seq_len, 1, output_dim]

            # Update routing logits
            if i < self.routing_iterations - 1:
                # Compute agreement
                agreement_visual = torch.matmul(u_visual.unsqueeze(2), v.transpose(-1, -2))
                agreement_audio = torch.matmul(u_audio.unsqueeze(2), v.transpose(-1, -2))

                # Update routing logits
                b[:, :, 0] = b[:, :, 0] + agreement_visual
                b[:, :, 1] = b[:, :, 1] + agreement_audio

        # Output the fused representation
        fused = v.squeeze(2)  # [batch_size, seq_len, output_dim]

        return fused


class AttentionFusionModule(nn.Module):
    """
    Main fusion module combining cross-modal attention and reliability
    estimation for adaptive weighting of modalities.
    """

    def __init__(self, audio_dim, visual_dim, hidden_dim=16, output_dim=512):
        super(AttentionFusionModule, self).__init__()

        # Cross-modal attention
        self.cross_attention = CrossModalAttention(audio_dim, visual_dim, hidden_dim)

        # Dynamic routing for robust fusion
        self.dynamic_routing = DynamicRoutingFusion(audio_dim, visual_dim, hidden_dim, routing_iterations=3)

        # Final projection
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

    def forward(self, visual_features, audio_features, raw_audio=None, raw_visual=None):
        """
        Fuse audio and visual features using both attention and routing mechanisms.

        Args:
            visual_features: Visual model features [batch_size, seq_len, visual_dim]
            audio_features: Audio model features [batch_size, seq_len, audio_dim]
            raw_audio: Raw audio input for reliability estimation
            raw_visual: Raw visual input for reliability estimation
        """
        # Cross-modal attention fusion
        attention_fusion = self.cross_attention(
            visual_features, audio_features, raw_audio, raw_visual
        )

        # Dynamic routing fusion
        routing_fusion = self.dynamic_routing(visual_features, audio_features)

        # Combine both fusion strategies
        combined = torch.cat((attention_fusion, routing_fusion), dim=-1)

        # Final projection
        output = self.final_projection(combined)

        return output
