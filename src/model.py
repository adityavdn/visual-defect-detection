"""
A convolutional autoencoder for one-class anomaly detection.

The idea in one sentence: train the network to rebuild NORMAL products only,
then show it a defective one — it rebuilds the defect as though it were normal,
and the difference between input and reconstruction is where the fault is.

The architecture is deliberately small and symmetric. Depth here is not the
interesting variable; the bottleneck is. Too wide a bottleneck and the network
learns to copy pixels, reconstructing defects perfectly and detecting nothing.
Too narrow and normal products reconstruct badly too, so everything looks
anomalous. Finding that balance is part of the project.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    def __init__(self, base_channels: int = 32, latent_channels: int = 128, depth: int = 4):
        super().__init__()
        channels = [3] + [base_channels * (2 ** i) for i in range(depth - 1)] + [latent_channels]

        self.encoder = nn.Sequential(
            *[_encoder_block(channels[i], channels[i + 1]) for i in range(depth)]
        )
        reverse = channels[::-1]
        self.decoder = nn.Sequential(
            *[_decoder_block(reverse[i], reverse[i + 1]) for i in range(depth - 1)],
            nn.ConvTranspose2d(reverse[depth - 1], 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),      # outputs in [0, 1], matching ToTensor() inputs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_model(config: dict) -> nn.Module:
    if config["architecture"] != "conv_autoencoder":
        raise ValueError("Unsupported architecture '{}'.".format(config["architecture"]))
    return ConvAutoencoder(
        base_channels=int(config.get("base_channels", 32)),
        latent_channels=int(config.get("latent_channels", 128)),
        depth=int(config.get("depth", 4)),
    )
