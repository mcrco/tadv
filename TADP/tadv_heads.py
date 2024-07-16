import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearHead(nn.Module):
    """Classification head for tadv testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
    """

    def __init__(self,
                 num_classes,
                 in_channels):
        nn.Module.__init__(self)
        self.mean = lambda x : torch.mean(x, 1) # first dim is video, second dim is frames
        self.flatten = nn.Flatten()
        self.cls_head = nn.Linear(in_channels, num_classes) # 6 classes in tsh-mpii

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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)
        
    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach().to(x.get_device())

class TransformerClassifier(nn.Module):
    def __init__(self, num_layers, num_heads, embed_dim, hidden_dim, dropout, num_classes, num_frames=8):
        super().__init__()
        
        self.pos_encoding = PositionalEncoding(embed_dim, num_frames)
        
        encoder_layers = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=hidden_dim, 
                dropout=dropout, 
                batch_first=True
                )
        self.transformer_encoder = nn.TransformerEncoder(
                encoder_layers, 
                num_layers
                )

        self.fc1 = nn.Linear(num_frames * embed_dim, embed_dim)
        self.fc2 = nn.Linear(embed_dim, num_classes)

        # param_count = 0
        # for p in self.parameters():
        #     param_count += p.numel()
        # print('transformer params', param_count)

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class NeeharHead(nn.Module):
    """Classification head for tadv testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
    """

    def __init__(self,
                 num_classes=6,
                 in_channels=320,
                 num_heads=4,
                 num_layers=4,
                 hidden_dim=1024,
                 dropout=0.1,
                 num_frames=8):

        nn.Module.__init__(self)

        # reshape from batch x frames x 320 x 64 x 64 -> batch x frames x 320
        self.mean = lambda x : torch.mean(x, dim=(-1, -2)) 

        # transformer-based classifier
        self.classifier = TransformerClassifier(
                num_heads=num_heads,
                num_layers=num_layers,
                embed_dim=in_channels,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_classes=num_classes,
                num_frames=num_frames
                )

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, x):
        x = self.mean(x)
        x = self.classifier(x)
        return x

class RogerioHead(nn.Module):
    """Classification head for tadv testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
    """

    def __init__(self,
                 num_classes=6,
                 in_channels=320,
                 embed_dim=512,
                 num_heads=8,
                 num_layers=6,
                 hidden_dim=2048,
                 dropout=0.1,
                 num_frames=8,
                 init_super=True):

        if init_super:
            super().__init__()

        # interpolate maps down from 64 x 64 to 8 x 8 
        self.interpolate = lambda x: F.interpolate(x, size=(8, 8), mode='bilinear', align_corners=False)

        # reduce channels from 320 to 64
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU()
        )

        # flatten into (n_batch * n_frames) x 4096
        self.flatten = nn.Flatten(-3, -1)

        # linear embedding to feed into transformer
        self.embed = nn.Linear(4096, 512)

        # transformer-based classifier
        self.classifier = TransformerClassifier(
                num_heads=num_heads,
                num_layers=num_layers,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                num_classes=num_classes,
                num_frames=num_frames
                )

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

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
        # (n_batch * n_frames, 4096) -> (nbatch * n_frames, 512)
        x = self.embed(x)
        # (n_batch * n_frames, 4096) -> (nbatch, n_frames, 512)
        x = x.reshape(n_batch, n_frames, 512)
        # (n_batch, n_frames, 512) -> (nbatch, num_classes)
        x = self.classifier(x)
        return x
