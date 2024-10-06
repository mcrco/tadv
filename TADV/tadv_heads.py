import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DiffusionFlatten(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # interpolate maps up to 64 x 64
        self.interpolate = lambda x: F.interpolate(
            x, size=(8, 8), mode="bilinear", align_corners=False
        )

        # reduce channels from 320 to 64
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        # flatten into (n_batch * n_frames) x 4096
        self.flatten = nn.Flatten()

    def forward(self, x):
        n_batch, n_frames, channels, height, width = x.shape
        # (n_batch, n_frames, channels, height, width) -> (nbatch * n_frames, channels, height, width)
        x = x.reshape(n_batch * n_frames, channels, height, width)
        # (n_batch * n_frames, channels, height, width) -> (nbatch * n_frames, channels, 8, 8)
        x = self.interpolate(x)
        # (n_batch * n_frames, channels, 8, 8) -> (nbatch * n_frames, 64, 8, 8)
        x = self.conv_down(x)
        # (n_batch * n_frames, 64, 8, 8) -> (nbatch * n_frames, 4096)
        x = self.flatten(x)
        return x


class LinearHead(nn.Module):
    def __init__(self, num_classes, in_channels):
        nn.Module.__init__(self)
        self.mean = lambda x: torch.mean(
            x, 1
        )  # first dim is video, second dim is frames
        self.flatten = nn.Flatten()
        self.cls_head = nn.Linear(in_channels, num_classes)  # 6 classes in tsh-mpii

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        x = self.mean(x)
        x = self.flatten(x)
        x = self.cls_head(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)].detach().to(x.get_device())


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        num_layers=6,
        num_heads=8,
        embed_dim=512,
        hidden_dim=2048,
        dropout=0.1,
        num_classes=51,
        num_frames=8,
    ):
        super().__init__()

        self.pos_encoding = PositionalEncoding(embed_dim, num_frames)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.pooling = lambda x: x.mean(dim=1)
        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = self.pooling(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class NeeharHead(nn.Module):
    def __init__(
        self,
        num_classes=51,
        in_channels=717,
        hidden_dim=2048,
        dropout=0.1,
        num_frames=8,
    ):

        nn.Module.__init__(self)

        # global average pooling
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        # 3 layer mlp
        self.classifier = nn.Sequential(
            nn.Linear(num_frames * in_channels, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        b, f, c, h, w = x.shape
        x = rearrange(x, "b f c h w -> (b f) c h w", b=b, f=f)
        x = self.gap(x)
        x = rearrange(x, "(b f) c -> b (f c)", b=b, f=f)
        x = self.classifier(x)
        return x


class RogerioHead(nn.Module):
    def __init__(
        self,
        num_classes=6,
        in_channels=320,
        embed_dim=4096,
        num_heads=8,
        num_layers=6,
        hidden_dim=2048,
        dropout=0.1,
        num_frames=8,
        init_super=True,
    ):

        if init_super:
            super().__init__()

        # reduce to 4096 features per frame
        self.dim_reduce = DiffusionFlatten(in_channels)
        self.embed_dim = embed_dim

        # transformer-based classifier
        self.classifier = TransformerClassifier(
            num_heads=num_heads,
            num_layers=num_layers,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
            num_frames=num_frames,
        )

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        n_batch, n_frames, channels, height, width = x.shape
        x = self.dim_reduce(x)
        # (n_batch * n_frames, 4096) -> (nbatch, n_frames, 4096)
        x = x.reshape(n_batch, n_frames, self.embed_dim)
        # (n_batch, n_frames, 4096) -> (nbatch, num_classes)
        x = self.classifier(x)
        return x


class MLPClassifier(nn.Module):
    def __init__(
        self,
        num_classes=51,
        embed_dim=2048,
        hidden_dim=2048,
        num_frames=8,
        dropout=0.5,
        init_super=True,
    ):

        if init_super:
            super().__init__()

        self.embed_dim = embed_dim
        self.num_frames = num_frames

        self.layers = nn.Sequential(
            nn.Linear(embed_dim * num_frames, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class MLPHead(nn.Module):
    def __init__(
        self,
        num_classes=6,
        in_channels=717,
        # embed_dim=4096,
        hidden_dim=2048,
        dropout=0.5,
        num_frames=8,
        init_super=True,
    ):

        if init_super:
            super().__init__()

        self.num_frames = num_frames

        # reduce diffusion features to num_channels
        # self.dim_reduce = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten()
        # )
        self.dim_reduce = DiffusionFlatten(in_channels)

        # mlp classifier
        self.mlp = MLPClassifier(
            embed_dim=4096,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_frames=num_frames,
            num_classes=num_classes,
        )

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        n_batch, n_frames, channels, height, width = x.shape
        x = self.dim_reduce(x)

        # (n_batch * n_frames, embed_dim) -> (n_batch, n_frames * embed_dim)
        x = rearrange(x, "(b f) c -> b (f c)", b=n_batch, f=n_frames)

        x = self.mlp(x)

        return x


class ConvHead(nn.Module):
    def __init__(
        self,
        num_classes=51,
        in_channels=320,
        in_dim=32,
        num_frames=8,
        dropout=0.1,
        init_super=True,
    ):

        if init_super:
            super().__init__()

        # (n, 717, 8, 32, 32) -> (n, 64, 8, 32, 32)
        self.conv1 = nn.Conv3d(
            in_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.LeakyReLU()
        # (n, 717, 8, 32, 32) -> (n, 64, 4, 16, 16)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dropout1 = nn.Dropout(dropout)

        # (n, 64, 4, 16, 16) -> (n, 128, 4, 16, 16)
        self.conv2 = nn.Conv3d(
            64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.LeakyReLU()
        # (n, 128, 4, 16, 16) -> (n, 128, 2, 8, 8)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dropout2 = nn.Dropout(dropout)

        # (n, 128, 2, 8, 8) -> (n, 256, 2, 8, 8)
        self.conv3 = nn.Conv3d(
            128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.LeakyReLU()
        # (n, 256, 2, 8, 8) -> (n, 256, 1, 4, 4)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dropout3 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(num_frames * in_dim * in_dim // 2, 1024)
        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        # x.shape: batch, frames, channels, height, width
        x = x.permute(0, 2, 1, 3, 4)
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))

        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x


class ConvMLP(nn.Module):
    def __init__(
        self,
        num_classes=6,
        in_channels=717,
        embed_dim=2048,
        hidden_dim=2048,
        dropout=0.5,
        num_frames=8,
        init_super=True,
    ):

        if init_super:
            super().__init__()

        self.num_frames = num_frames

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),
        )

        # reduce diffusion features to num_channels
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        # mlp classifier
        self.mlp = MLPClassifier(
            embed_dim=embed_dim,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_frames=num_frames,
            num_classes=num_classes,
        )

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        n_batch, n_frames, channels, height, width = x.shape

        x = rearrange(x, "b f c h w -> (b f) c h w")
        x = self.conv(x)

        x = rearrange(x, "(b f) c h w -> b f c h w", b=n_batch, f=n_frames)
        x = self.gap(x)

        x = self.mlp(x)

        return x
