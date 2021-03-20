# Homework: build a resnet-like network and apply to the EMNIST dataset
from typing import Any, Dict
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


CONV_DIM = 64
FC_DIM = 128
IMAGE_SIZE = 28
NUM_CONV_BLOCKS = 2


class ResBlock(nn.Module):
    """
    3x3 conv with padding size 1 (to leave the input size unchanged), followed by batch norm, and a ReLu
    Identity pass is added to be ResNet-like
    """
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            of dimensions (B, C, H, W)

        Returns
        -------
        torch.Tensor
            of dimensions (B, C, H, W)
        """
        # Identity
        identity = x

        # Conv layer
        out = self.conv1(x)
        out = self.bn(out)

        # Add identity pass
        out += identity
        out = self.relu(out)
        return out


class CNN2(nn.Module):
    """Simple CNN for recognizing characters in a square image."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])

        conv_dim = self.args.get("conv_dim", CONV_DIM)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        self.num_conv_blocks = self.args.get("num_conv_blocks", NUM_CONV_BLOCKS)

        self.conv1 = ResBlock(input_dims[0], conv_dim)

        # Define a list of ResBlocks as parameterized by the # of Conv Blocks,
        # minus 1 is to offset the first conv block, i.e., self.conv1
        self.conv2_list = nn.ModuleList([ResBlock(conv_dim, conv_dim) for _ in range(self.num_conv_blocks - 1)])
        
        # Instead of using 2-by-2 max pooling, we use a 2x2 conv layer with stride=2
        # to reduce the size of the input by half
        self.conv3 = nn.Conv2d(conv_dim, conv_dim, kernel_size=2, stride=2)

        # Because our 3x3 convs have padding size 1, they leave the input size unchanged.
        # The stride-2 conv layer divides the input size by 2. Flattening squares it.
        conv_output_size = IMAGE_SIZE // 2
        fc_input_dim = int(conv_output_size * conv_output_size * conv_dim)
        self.fc1 = nn.Linear(fc_input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        B_, C_, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.conv1(x)

        for conv_block in self.conv2_list:
            x = conv_block(x)
        
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--num_conv_blocks", type=int, default=NUM_CONV_BLOCKS)
        return parser