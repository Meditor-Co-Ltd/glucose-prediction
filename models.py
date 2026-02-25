import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DCNN(nn.Module):
    def __init__(self, num_inputs, encoding_depth, num_classes, normalize_labels):
        super(DCNN, self).__init__()
        self.encoding_depth = encoding_depth
        self.drop_rate = 0.2
        self.num_classes = num_classes

        if encoding_depth == 4:
            self.max_pool_size = [1, 2, 3, 5]  # 330/2/3/5=11
            self.enc_dilation_size = [128, 64, 16, 4]
            self.enc_filter_sizes = [4, 8, 16, 32]
            if num_inputs == 6:
                self.middle_filter_size = 66
            elif num_inputs == 18:
                self.middle_filter_size = 78
        elif encoding_depth == 3:
            self.max_pool_size = [2, 3, 5]  # 330/2/3/5=11
            self.enc_dilation_size = [64, 16, 4]
            self.enc_filter_sizes = [4, 8, 16]
            self.middle_filter_size = 34
        elif encoding_depth == 2:
            self.max_pool_size = [2, 3]
            self.enc_dilation_size = [64, 16]
            self.enc_filter_sizes = [4, 8]
            self.middle_filter_size = 18

        self.total_layer_depth = len(self.enc_filter_sizes)

        self.initial_conv = nn.Sequential(
            nn.Conv1d(num_inputs, num_inputs, kernel_size=1, padding=0),
            nn.Dropout(0.1)
        )

        self.enc_blocks = nn.ModuleList()
        for i in range(self.total_layer_depth):
            d = self.enc_dilation_size[i]
            d_padding = ((3 - 1) * d) // 2
            block = nn.Sequential(
                nn.Conv1d(num_inputs + sum(self.enc_filter_sizes[:i]), self.enc_filter_sizes[i], kernel_size=3, padding=1),
                nn.LeakyReLU(0.15),
                nn.BatchNorm1d(self.enc_filter_sizes[i]),
                nn.Conv1d(self.enc_filter_sizes[i], self.enc_filter_sizes[i], kernel_size=3, padding=d_padding, dilation=d),
                nn.LeakyReLU(0.15),
                nn.BatchNorm1d(self.enc_filter_sizes[i]),
                nn.Conv1d(self.enc_filter_sizes[i], self.enc_filter_sizes[i], kernel_size=1, padding=0),
                nn.LeakyReLU(0.15),
                nn.Dropout(self.drop_rate)
            )
            self.enc_blocks.append(block)

        self.pool = nn.ModuleList([nn.MaxPool1d(k, stride=k, padding=0) for k in self.max_pool_size])
        
        self.lstm_layers = 3
        self.lstm = nn.LSTM(self.middle_filter_size, self.middle_filter_size, num_layers=self.lstm_layers, batch_first=True, dropout=self.drop_rate)

        self.layernorm = nn.LayerNorm(self.middle_filter_size)

        self.middle_layers = nn.Sequential(
            nn.Conv1d(self.middle_filter_size, self.middle_filter_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.middle_filter_size),
            nn.LeakyReLU(0.15),
            # nn.Dropout(self.drop_rate),
            # nn.Conv1d(self.middle_filter_size, self.middle_filter_size, kernel_size=1, padding=0),
            # nn.LeakyReLU(0.15),

            nn.Conv1d(self.middle_filter_size, self.middle_filter_size // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.middle_filter_size // 2),
            nn.LeakyReLU(0.15),
            # nn.Dropout(self.drop_rate),
            # nn.Conv1d(self.middle_filter_size // 2, self.middle_filter_size // 2, kernel_size=1, padding=0),
            # nn.LeakyReLU(0.15),

            nn.Conv1d(self.middle_filter_size // 2, self.middle_filter_size // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.middle_filter_size // 2),
            nn.LeakyReLU(0.15),
            # nn.Dropout(self.drop_rate),
            # nn.Conv1d(self.middle_filter_size // 2, self.middle_filter_size // 2, kernel_size=1, padding=0),
            # nn.LeakyReLU(0.15),

            nn.Conv1d(self.middle_filter_size // 2, self.middle_filter_size // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.middle_filter_size // 4),
            nn.LeakyReLU(0.15),
            # nn.Dropout(self.drop_rate),
            # nn.Conv1d(self.middle_filter_size // 4, self.middle_filter_size // 4, kernel_size=1, padding=0),
            # nn.LeakyReLU(0.15),

            nn.Conv1d(self.middle_filter_size // 4, self.middle_filter_size // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.middle_filter_size // 4),
            nn.LeakyReLU(0.15),
            # nn.Dropout(self.drop_rate),
            # nn.Conv1d(self.middle_filter_size // 4, self.middle_filter_size // 4, kernel_size=1, padding=0),
            # nn.LeakyReLU(0.15),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        if num_classes == 1:
            if normalize_labels:
                self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.middle_filter_size // 4, 1),
                nn.Sigmoid()
            )
            else:
                self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.middle_filter_size // 4, 1),
                nn.Softplus()
            )
        elif num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.middle_filter_size // 4, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.middle_filter_size // 4, num_classes),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        # x shape: (batch, seq_len, num_inputs) -> convert to (batch, channels, seq_len)
        # x = x.permute(0, 2, 1)
        x = self.initial_conv(x)

        for i in range(self.total_layer_depth):
            # print(i)
            # print(x.shape)

            pooled = self.pool[i](x)
            # identity = x
            # identity_pooled = self.pool[i](identity)
            
            # print(pooled.shape)
            # print(pooled.shape)
            x = self.enc_blocks[i](x)
            x = self.pool[i](x)
            # # pad pooled to match x shape if needed
            # if pooled.shape[-1] != x.shape[-1]:
            #     diff = x.shape[-1] - pooled.shape[-1]
            #     pooled = F.pad(pooled, (0, diff))

            x = torch.cat([x, pooled], dim=1)
            # x = torch.cat([x, identity_pooled], dim=1)

        # print(x.shape)
        x = x.permute(0, 2, 1)
        # print(x.shape)

        h0 = torch.zeros(self.lstm_layers, x.size(0), self.middle_filter_size, dtype=x.dtype, device=x.device).to(x.device)
        c0 = torch.zeros(self.lstm_layers, x.size(0), self.middle_filter_size, dtype=x.dtype, device=x.device).to(x.device)
        x, _ = self.lstm(x, (h0, c0))
        x = self.layernorm(x)

        # h1 = torch.zeros(self.lstm_layers, x.size(0), self.middle_filter_size, dtype=x.dtype, device=x.device).to(x.device)
        # c1 = torch.zeros(self.lstm_layers, x.size(0), self.middle_filter_size, dtype=x.dtype, device=x.device).to(x.device)
        # x, _ = self.lstm(x, (h1, c1))
        # x = self.layernorm(x)

        # print(x.shape)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = self.middle_layers(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        
        return x


# A standard helper module for sinusoidal positional encoding
# This is crucial for the Transformer to understand the order of wavelengths.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # The buffer is not a model parameter, but should be part of the state_dict
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SpectralDropout(nn.Module):
    """
    Drops entire wavelength positions (zeroes all channels at randomly chosen
    spectral indices) to prevent the model from relying on specific wavelengths.
    Applied only during training; no-op during eval.

    Args:
        p: Probability of dropping each wavelength position.
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        # x: (batch, channels, wavelengths)
        if not self.training or self.p == 0.0:
            return x
        # Bernoulli mask over wavelength positions, broadcast across channels
        mask = torch.bernoulli(
            torch.full((x.shape[0], 1, x.shape[2]), 1.0 - self.p,
                       device=x.device, dtype=x.dtype)
        )
        return x * mask / (1.0 - self.p)  # scale to preserve expected value


# The main CNN-Transformer Hybrid Model
class SpecFormer(nn.Module):
    """
    A Spectrometry Transformer model that uses a CNN backbone for feature extraction.
    """
    def __init__(self, *, seq_len: int, input_channels: int, num_classes: int,
                 d_model: int = 128, nhead: int = 8, num_encoder_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1, normalize_labels: bool = False,
                 spectral_dropout: float = 0.0):
        """
        Args:
            seq_len (int): The length of the input sequence (e.g., 901 for 100-1000nm).
            input_channels (int): The number of input channels for the spectrum data (usually 1).
            num_classes (int): The number of output classes or 1 for regression.
            d_model (int): The embedding dimension for the Transformer. Must be divisible by nhead.
            nhead (int): The number of attention heads in the Transformer.
            num_encoder_layers (int): The number of layers in the Transformer encoder.
            dim_feedforward (int): The dimension of the feedforward network in the Transformer.
            dropout (float): The dropout rate.
            normalize_labels (bool): If True and num_classes=1, output is scaled to [0,1] with Sigmoid.
        """
        super(SpecFormer, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        print(self.num_classes)

        # 0. Spectral Dropout (drops entire wavelength positions before CNN)
        self.spectral_dropout = SpectralDropout(p=spectral_dropout)

        # 1. CNN Feature Extractor
        # This part processes the raw spectrum and extracts local features like peaks and slopes.
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(input_channels, d_model // 4, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv1d(d_model // 4, d_model // 2, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        
        # Calculate the sequence length after the CNN extractor
        # This is important for the positional encoding
        cnn_output_len = self._get_cnn_output_len(seq_len)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=cnn_output_len + 1)

        # 3. Transformer Encoder
        # This part models the global relationships between the extracted features.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False # We will permute the input to (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Classifier Head
        # This final part aggregates the Transformer's output and makes the prediction.
        # It replicates the output logic from your original code.
        self.aggregator = nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        
        # Classifier with logic based on the number of classes and normalization
        if num_classes == 1:
            if normalize_labels:
                # self.classifier = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
                # self.classifier = nn.Sequential(nn.Linear(d_model, 1), nn.Hardsigmoid())
                self.classifier = nn.Sequential(nn.Linear(d_model, 1))  # for use with torch.clamp(logits, min=0, max=1)
            else:
                self.classifier = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        elif num_classes == 2:
            self.classifier = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        else:
            # Multi-class classification: NO Softmax! CrossEntropyLoss expects raw logits
            self.classifier = nn.Linear(d_model, num_classes)
        
        if normalize_labels:
            self.mu_head = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
        else:
            self.mu_head = nn.Linear(d_model, 1)
        self.sigma_head = nn.Linear(d_model, 1)


    def _get_cnn_output_len(self, seq_len):
        # Helper to calculate the sequence length after convolutions
        def out_len(in_len, kernel, stride, padding):
            return (in_len + 2*padding - kernel) // stride + 1
        
        l = out_len(seq_len, 7, 2, 3)
        l = out_len(l, 5, 2, 2)
        l = out_len(l, 3, 2, 1)
        return l

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape [batch_size, input_channels, seq_len]
        """
        # 0. Spectral dropout (training only): randomly zero entire wavelength positions
        x = self.spectral_dropout(x)

        # 1. Pass through CNN extractor
        # Input: (batch, channels, seq_len) -> Output: (batch, d_model, new_seq_len)
        x = self.cnn_extractor(x)

        # 2. Prepare for Transformer
        # Transformer expects (seq_len, batch, features).
        # Input: (batch, d_model, new_seq_len) -> Output: (new_seq_len, batch, d_model)
        x = x.permute(2, 0, 1)

        # 3. Add positional encoding
        x = self.pos_encoder(x)

        # 4. Pass through Transformer encoder
        x = self.transformer_encoder(x)

        # 5. Aggregate and classify
        # Revert permutation for pooling: (new_seq_len, batch, d_model) -> (batch, d_model, new_seq_len)
        x = x.permute(1, 2, 0)

        # Global average pooling: (batch, d_model, new_seq_len) -> (batch, d_model, 1)
        x = self.aggregator(x)

        # Flatten for the linear layer: (batch, d_model, 1) -> (batch, d_model)
        x = torch.flatten(x, 1)
        x = torch.nn.Dropout(p=0.25)(x)

        # Check if this is multi-class classification (num_classes > 2)
        if self.num_classes > 2:
            # Multi-class classification: return raw logits (NO Softmax, CrossEntropyLoss handles it)
            logits = self.classifier(x)
            return logits
        else:
            # Probabilistic regression (num_classes == 1) or binary classification (num_classes == 2)
            # Return mu and log_var for regression
            mu = self.mu_head(x)
            log_var = self.sigma_head(x)
            return mu, log_var


# =============================================================================
# Conformer Components for SpecFormer2
# =============================================================================

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConformerFeedForward(nn.Module):
    """
    Feed-forward module for Conformer.
    Uses expansion factor of 4 and Swish activation.
    """
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        expanded_dim = d_model * expansion_factor
        self.sequential = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expanded_dim),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.sequential(x)


class ConformerConvModule(nn.Module):
    """
    Convolution module for Conformer.
    Uses depthwise separable convolution with gating mechanism.
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        # Ensure kernel_size is odd for 'same' padding
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise conv (expand to 2*d_model for GLU gating)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)

        # Depthwise conv
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()

        # Pointwise conv (project back)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        x = self.layer_norm(x)

        # Transpose for conv: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        # Transpose back: (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2)
        return x


class ConformerBlock(nn.Module):
    """
    A single Conformer block combining:
    1. Feed-forward (half-step)
    2. Multi-head self-attention
    3. Convolution module
    4. Feed-forward (half-step)
    5. Final layer norm
    """
    def __init__(self, d_model: int, nhead: int, conv_kernel_size: int = 31,
                 ff_expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()

        # First feed-forward (half-step)
        self.ff1 = ConformerFeedForward(d_model, ff_expansion_factor, dropout)

        # Multi-head self-attention
        self.attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        # Convolution module
        self.conv_module = ConformerConvModule(d_model, conv_kernel_size, dropout)

        # Second feed-forward (half-step)
        self.ff2 = ConformerFeedForward(d_model, ff_expansion_factor, dropout)

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # Feed-forward 1 (half-step residual)
        x = x + 0.5 * self.ff1(x)

        # Self-attention with residual
        attn_input = self.attn_layer_norm(x)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input)
        x = x + self.attn_dropout(attn_output)

        # Convolution module with residual
        x = x + self.conv_module(x)

        # Feed-forward 2 (half-step residual)
        x = x + 0.5 * self.ff2(x)

        # Final layer norm
        x = self.final_layer_norm(x)

        return x


class SpecFormer2(nn.Module):
    """
    A Spectrometry Transformer model that uses Conformer blocks instead of
    separate CNN extractor + Transformer encoder.

    Conformer combines convolution and self-attention in each block,
    which can be more effective for capturing both local and global patterns.
    """
    def __init__(self, *, seq_len: int, input_channels: int, num_classes: int,
                 d_model: int = 128, nhead: int = 8, num_encoder_layers: int = 4,
                 dim_feedforward: int = 512, conv_kernel_size: int = 15,
                 dropout: float = 0.1, normalize_labels: bool = False):
        """
        Args:
            seq_len (int): The length of the input sequence (e.g., 150 for 500-650nm).
            input_channels (int): The number of input channels (e.g., 7 with absorption).
            num_classes (int): The number of output classes or 1 for regression.
            d_model (int): The embedding dimension. Must be divisible by nhead.
            nhead (int): The number of attention heads.
            num_encoder_layers (int): The number of Conformer blocks.
            dim_feedforward (int): The dimension of the feedforward network.
            conv_kernel_size (int): Kernel size for depthwise conv in Conformer.
                                    For seq_len=150, try 7, 15, or 31.
            dropout (float): The dropout rate.
            normalize_labels (bool): If True and num_classes=1, output scaled to [0,1].
        """
        super(SpecFormer2, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes

        # Calculate expansion factor from dim_feedforward
        ff_expansion_factor = max(1, dim_feedforward // d_model)

        print(f"SpecFormer2: {num_classes} classes, {num_encoder_layers} layers, kernel={conv_kernel_size}, ff={dim_feedforward}")

        # 1. Input projection: project input_channels to d_model
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_channels, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 1)

        # 3. Conformer Blocks (replaces CNN extractor + Transformer encoder)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                nhead=nhead,
                conv_kernel_size=conv_kernel_size,
                ff_expansion_factor=ff_expansion_factor,
                dropout=dropout
            )
            for _ in range(num_encoder_layers)
        ])

        # 4. Classifier Head
        self.aggregator = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling

        if num_classes == 1:
            if normalize_labels:
                self.classifier = nn.Sequential(nn.Linear(d_model, 1))
            else:
                self.classifier = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        elif num_classes == 2:
            self.classifier = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        else:
            self.classifier = nn.Linear(d_model, num_classes)

        if normalize_labels:
            self.mu_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        else:
            self.mu_head = nn.Linear(d_model, 1)
        self.sigma_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape [batch_size, input_channels, seq_len]
        """
        # 1. Project input channels to d_model
        # Input: (batch, channels, seq_len) -> Output: (batch, d_model, seq_len)
        x = self.input_projection(x)

        # 2. Prepare for Conformer (batch_first=True)
        # (batch, d_model, seq_len) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2)

        # 3. Add positional encoding
        # PositionalEncoding expects (seq_len, batch, d_model), so we transpose
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # 4. Pass through Conformer blocks
        for conformer_block in self.conformer_blocks:
            x = conformer_block(x)

        # 5. Aggregate and classify
        # (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        x = x.transpose(1, 2)

        # Global average pooling: (batch, d_model, seq_len) -> (batch, d_model, 1)
        x = self.aggregator(x)

        # Flatten: (batch, d_model, 1) -> (batch, d_model)
        x = torch.flatten(x, 1)
        x = nn.Dropout(p=0.25)(x)

        if self.num_classes > 2:
            logits = self.classifier(x)
            return logits
        else:
            mu = self.mu_head(x)
            log_var = self.sigma_head(x)
            return mu, log_var


def gaussian_nll_loss(mu, sigma, y_true):
    """
    Calculates the Gaussian Negative Log-Likelihood loss.
    """
    # Calculate the two terms of the loss function
    term1 = 0.5 * torch.log(2 * torch.pi * sigma**2)
    term2 = ((y_true - mu)**2) / (2 * sigma**2)
    
    # Sum the terms and take the mean over the batch
    loss = torch.mean(term1 + term2)
    return loss


def stable_gaussian_nll_loss(mu, log_var, y_true):
    """
    Calculates a numerically stable Gaussian Negative Log-Likelihood loss.

    Args:
        mu (Tensor): The predicted mean.
        log_var (Tensor): The predicted log of the variance.
        y_true (Tensor): The ground truth labels.
    """
    # # The two terms of the loss function, rearranged for stability
    # # The formula is: 0.5 * (log(2π) + log_var + (y - mu)² / var)
    # term1 = 0.5 * log_var

    # # Calculate variance = exp(log_var)
    # variance = torch.exp(log_var)
    # term2 = 0.5 * (((y_true - mu)**2) / variance)

    # # Add a constant for the 2π term
    # log2pi = math.log(2 * math.pi)

    # loss = torch.mean(term1 + term2 + 0.5 * log2pi)
    # return loss

    # log_var = log_var.clamp(min=-20.0)
    term1 = 0.5 * log_var
    variance = torch.exp(log_var)
    term2 = 0.5 * (((y_true - mu)**2) / variance)
    log2pi = math.log(2 * math.pi)

    loss = torch.mean(term1 + term2 + 0.5 * log2pi)
    return loss


def combined_probabilistic_loss(mu, log_var, y_true, mse_weight=0.5, var_penalty=0.01):
    """
    Combined loss function to prevent regression to mean problem.
    Combines NLL loss with MSE and variance penalty.

    Args:
        mu (Tensor): The predicted mean.
        log_var (Tensor): The predicted log of the variance.
        y_true (Tensor): The ground truth labels.
        mse_weight (float): Weight for MSE loss (0 to 1). Default 0.5 balances NLL and MSE.
        var_penalty (float): Penalty coefficient for high variance. Default 0.01.

    Returns:
        Combined loss value

    How this prevents regression to mean:
        - NLL: Learns uncertainty (sigma)
        - MSE: Forces accurate predictions (can't hide behind high uncertainty)
        - Variance penalty: Prevents lazy high-uncertainty predictions
    """
    import torch.nn.functional as F

    # 1. NLL loss with variance penalty
    term1 = 0.5 * log_var
    variance = torch.exp(log_var)
    term2 = 0.5 * (((y_true - mu)**2) / variance)
    log2pi = math.log(2 * math.pi)
    nll_loss = torch.mean(term1 + term2 + 0.5 * log2pi)

    # 2. Variance penalty - discourages high uncertainty
    variance_penalty = var_penalty * torch.mean(variance)

    # 3. MSE loss - directly penalizes prediction errors
    mse_loss = F.mse_loss(mu, y_true)

    # 4. Combine all three components
    total_loss = (1 - mse_weight) * (nll_loss + variance_penalty) + mse_weight * mse_loss

    return total_loss


# ============================================================================
# ORDINAL LOSS FUNCTIONS
# ============================================================================

def emd_focal_loss(logits, targets, num_classes, gamma=2.0, alpha=0.25, emd_weight=1.0):
    """
    Combined Earth Mover's Distance and Focal Loss for ordinal classification
    with class imbalance.

    Args:
        logits: Model predictions [batch_size, num_classes]
        targets: True class indices [batch_size]
        num_classes: Number of classes
        gamma: Focal loss focusing parameter (higher = focus more on hard examples)
               Typical values: 0.5 to 5.0, default 2.0
        alpha: Focal loss balancing parameter for rare classes
               Typical values: 0.25 to 0.75, default 0.25
        emd_weight: Weight for EMD term vs Focal term
                    Higher = prioritize ordinal correctness
                    Lower = prioritize getting exact class right

    Returns:
        total_loss: Combined loss
        focal_loss: Focal loss component (for logging)
        emd_loss: EMD component (for logging)
    """
    # Get predicted probabilities
    probs = F.softmax(logits, dim=1)

    # === FOCAL LOSS COMPONENT ===
    # Get probability of true class
    true_class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Focal weight: (1 - p_t)^gamma
    # Hard examples (low p_t) get higher weight
    focal_weight = (1 - true_class_probs) ** gamma

    # Cross entropy loss
    ce_loss = F.cross_entropy(logits, targets, reduction='none')

    # Apply focal weighting and alpha balancing
    focal_loss = alpha * focal_weight * ce_loss
    focal_loss_mean = focal_loss.mean()

    # === EMD COMPONENT ===
    # Convert targets to one-hot
    targets_one_hot = F.one_hot(targets, num_classes).float()

    # Compute cumulative distributions
    pred_cdf = torch.cumsum(probs, dim=1)
    target_cdf = torch.cumsum(targets_one_hot, dim=1)

    # EMD is the L1 distance between CDFs
    emd_loss = torch.mean(torch.sum(torch.abs(pred_cdf - target_cdf), dim=1))

    # === COMBINED LOSS ===
    total_loss = focal_loss_mean + emd_weight * emd_loss

    return total_loss, focal_loss_mean, emd_loss


def weighted_emd_focal_loss(logits, targets, num_classes, class_weights=None,
                           gamma=2.0, emd_weight=1.0):
    """
    EMD + Focal Loss with explicit class weights for severe imbalance.

    Args:
        class_weights: Tensor of shape [num_classes] with weight for each class
                      If None, will be computed from batch statistics
        gamma: Focal loss focusing parameter
        emd_weight: Weight for EMD term

    Returns:
        total_loss: Combined loss
        weighted_focal_loss: Weighted focal loss component
        weighted_emd: Weighted EMD component
    """
    batch_size = targets.size(0)
    device = logits.device

    # Auto-compute class weights if not provided
    if class_weights is None:
        # Count samples per class in this batch
        class_counts = torch.bincount(targets, minlength=num_classes).float()
        # Inverse frequency weighting
        class_counts = torch.clamp(class_counts, min=1.0)  # Avoid division by zero
        class_weights = batch_size / (num_classes * class_counts)
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize

    class_weights = class_weights.to(device)

    # Get predicted probabilities
    probs = F.softmax(logits, dim=1)

    # Ensure targets are Long type for gather and one_hot operations
    targets = targets.long()

    # === WEIGHTED FOCAL LOSS ===
    true_class_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_weight = (1 - true_class_probs) ** gamma

    # Get weight for each sample based on its class
    sample_weights = class_weights[targets]

    # Cross entropy with class weights
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    weighted_focal_loss = sample_weights * focal_weight * ce_loss
    weighted_focal_loss_mean = weighted_focal_loss.mean()

    # === WEIGHTED EMD ===
    targets_one_hot = F.one_hot(targets, num_classes).float()
    pred_cdf = torch.cumsum(probs, dim=1)
    target_cdf = torch.cumsum(targets_one_hot, dim=1)

    # Weight EMD by class weights too
    emd_per_sample = torch.sum(torch.abs(pred_cdf - target_cdf), dim=1)
    weighted_emd = (sample_weights * emd_per_sample).mean()

    # === COMBINED ===
    total_loss = weighted_focal_loss_mean + emd_weight * weighted_emd

    return total_loss, weighted_focal_loss_mean, weighted_emd


def compute_gradient_penalty(model, x_batch, logits, lambda_grad=0.01):
    """
    Compute gradient penalty to encourage robustness to input perturbations.

    This penalizes the model if small input changes cause large output changes,
    making predictions more stable across measurement variations.

    Args:
        model: The neural network model
        x_batch: Input tensor [batch_size, num_channels, seq_length]
        logits: Model output (already computed)
        lambda_grad: Weight for gradient penalty (typical: 0.001 to 0.1)

    Returns:
        gradient_penalty: Scalar tensor
    """
    if not x_batch.requires_grad:
        x_batch.requires_grad = True

    # Sum of all output logits (to get total sensitivity)
    output_sum = logits.sum()

    # Compute gradient of outputs w.r.t. inputs
    gradients = torch.autograd.grad(
        outputs=output_sum,
        inputs=x_batch,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute L2 norm of gradients (penalize large sensitivities)
    gradient_penalty = lambda_grad * torch.mean(gradients ** 2)

    return gradient_penalty


def emd_focal_loss_with_regularization(model, x_batch, logits, targets, num_classes,
                                       gamma=2.0, alpha=0.25, emd_weight=1.0,
                                       lambda_grad=0.0):
    """
    EMD + Focal Loss with optional gradient penalty for robustness.

    Args:
        model: The neural network model (needed for gradient computation)
        x_batch: Input tensor [batch_size, num_channels, seq_length]
        logits: Model predictions [batch_size, num_classes]
        targets: True class indices [batch_size]
        num_classes: Number of classes
        gamma: Focal loss focusing parameter
        alpha: Focal loss balancing parameter
        emd_weight: Weight for EMD component
        lambda_grad: Weight for gradient penalty (0 = disabled)

    Returns:
        total_loss: Combined loss
        focal_loss: Focal loss component
        emd_loss: EMD component
        grad_penalty: Gradient penalty component (0 if disabled)
    """
    # Compute standard EMD + Focal loss
    base_loss, focal_loss, emd_loss = emd_focal_loss(
        logits, targets, num_classes, gamma, alpha, emd_weight
    )

    # Add gradient penalty if requested
    if lambda_grad > 0:
        grad_penalty = compute_gradient_penalty(model, x_batch, logits, lambda_grad)
        total_loss = base_loss + grad_penalty
    else:
        grad_penalty = torch.tensor(0.0, device=logits.device)
        total_loss = base_loss

    return total_loss, focal_loss, emd_loss, grad_penalty


def weighted_emd_focal_loss_with_regularization(model, x_batch, logits, targets, num_classes,
                                                class_weights=None, gamma=2.0, emd_weight=1.0,
                                                lambda_grad=0.0):
    """
    Weighted EMD + Focal Loss with optional gradient penalty for robustness.

    Args:
        model: The neural network model (needed for gradient computation)
        x_batch: Input tensor [batch_size, num_channels, seq_length]
        logits: Model predictions [batch_size, num_classes]
        targets: True class indices [batch_size]
        num_classes: Number of classes
        class_weights: Tensor of shape [num_classes] with weight for each class
        gamma: Focal loss focusing parameter
        emd_weight: Weight for EMD component
        lambda_grad: Weight for gradient penalty (0 = disabled)

    Returns:
        total_loss: Combined loss
        weighted_focal_loss: Weighted focal loss component
        weighted_emd: Weighted EMD component
        grad_penalty: Gradient penalty component (0 if disabled)
    """
    # Compute standard weighted EMD + Focal loss
    base_loss, weighted_focal_loss, weighted_emd = weighted_emd_focal_loss(
        logits, targets, num_classes, class_weights, gamma, emd_weight
    )

    # Add gradient penalty if requested
    if lambda_grad > 0:
        grad_penalty = compute_gradient_penalty(model, x_batch, logits, lambda_grad)
        total_loss = base_loss + grad_penalty
    else:
        grad_penalty = torch.tensor(0.0, device=logits.device)
        total_loss = base_loss

    return total_loss, weighted_focal_loss, weighted_emd, grad_penalty

# ============================================================================
