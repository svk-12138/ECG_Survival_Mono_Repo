import math
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .base import BaseVAE
from .types_ import Tensor


class MedianBeatVAE(BaseVAE):
    """
    One-dimensional convolutional VAE for median ECG beats as described in the manuscript.
    """

    def __init__(
        self,
        in_channels: int = 12,
        seq_len: int = 512,
        latent_dim: int = 24,
        hidden_dims: Optional[List[int]] = None,
        kernel_sizes: Optional[List[int]] = None,
        beta: float = 0.25,
        eps: float = 1e-5,
        out_channels: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if latent_dim > 30:
            raise ValueError("latent_dim 应不大于 30 以匹配论文设置。")

        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.beta = beta
        self.eps = eps
        self.output_channels = out_channels or in_channels

        self.hidden_dims = hidden_dims or [32, 64, 128, 256]
        self.kernel_sizes = kernel_sizes or [9, 7, 5, 3]
        if len(self.kernel_sizes) != len(self.hidden_dims):
            raise ValueError("kernel_sizes 与 hidden_dims 长度需一致。")

        modules = []
        current_channels = in_channels
        length = seq_len
        for h_dim, k in zip(self.hidden_dims, self.kernel_sizes):
            padding = k // 2
            block = nn.Sequential(
                nn.Conv1d(current_channels, h_dim, kernel_size=k, stride=2, padding=padding),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(),
            )
            modules.append(block)
            current_channels = h_dim
            length = math.floor((length + 2 * padding - (k - 1) - 1) / 2 + 1)

        self.encoder = nn.Sequential(*modules)
        self.final_length = length
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * length, latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * length, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, self.hidden_dims[-1] * length)
        decoder_modules = []
        rev_dims = self.hidden_dims[::-1]
        rev_kernels = self.kernel_sizes[::-1]
        for idx in range(len(rev_dims) - 1):
            k = rev_kernels[idx]
            decoder_modules.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
                    nn.Conv1d(rev_dims[idx], rev_dims[idx + 1], kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(rev_dims[idx + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.Sequential(*decoder_modules)
        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(rev_dims[-1], self.output_channels, kernel_size=self.kernel_sizes[0], padding=self.kernel_sizes[0] // 2),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.final_length)
        if len(self.decoder) > 0:
            result = self.decoder(result)
        result = self.final_layer(result)
        if result.shape[-1] != self.seq_len:
            result = F.interpolate(result, size=self.seq_len, mode="linear", align_corners=False)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def smape_loss(self, recons: Tensor, input: Tensor) -> Tensor:
        numerator = torch.abs(recons - input)
        denominator = torch.abs(recons) + torch.abs(input) + self.eps
        smape = 2.0 * numerator / denominator
        return smape.mean()

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs.get("M_N", 1.0)

        recons_loss = self.smape_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))
        loss = recons_loss + self.beta * kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss,
            "KLD": kld_loss,
        }

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
