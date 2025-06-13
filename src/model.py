# src/model.py

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.modules import Activation
from typing import Optional, Union

class RegressionHead(nn.Sequential):
    """
    A simple regression head that applies a convolution, optional upsampling, and an activation.
    This head is used to predict dense 3D coordinates from the decoder features.
    """
    def __init__(self, in_channels, out_channels=3, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class UnetReg(smp.Unet):
    """
    The main Assembly-Depth model, based on the U-Net architecture.

    This model extends the standard segmentation U-Net to perform three tasks simultaneously,
    as described in the paper:
    1.  **Equipment Segmentation**: Predicts a binary mask for the assembly. (Primary head)
    2.  **Point Correspondence**: Predicts dense, normalized 3D coordinates for each pixel of the assembly. (Custom regression head)
    3.  **Part Classification**: Predicts the presence/absence of each part in the assembly. (Optional auxiliary head)

    Args:
        encoder_name (str): Name of the classification model to use as an encoder (e.g., 'efficientnet-b0').
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB or Depth).
        classes (int): Number of output classes for the segmentation mask (usually 1 for binary segmentation).
        aux_params (dict, optional): Parameters for the auxiliary classification head. If None, the head is not created.
    """
    def __init__(
        self,
        encoder_name: str,
        in_channels: int,
        classes: int,
        decoder_channels: list = (256, 128, 64, 32, 16),
        aux_params: Optional[dict] = None,
        activation: Optional[Union[str, callable]] = None
    ):
        super().__init__(
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=classes,
            decoder_attention_type='scse', # As used in original implementation
            aux_params=aux_params,
            decoder_channels=decoder_channels,
        )

        # The regression head predicts 5 channels:
        # - 3 channels for the (x, y, z) model point coordinates
        # - 1 channel for the metric depth (not used in the final paper's approach but kept for compatibility)
        # - 1 channel for a predicted error (not used in the final paper's approach but kept for compatibility)
        self.regression_head = RegressionHead(
            in_channels=decoder_channels[-1],
            out_channels=5,
            activation=activation,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the model.

        Returns:
            A tuple containing:
            - masks (Tensor): The output from the segmentation head.
            - positions (Tensor): The output from the custom regression head.
            - classifications (Tensor, optional): The output from the classification head, if it exists.
        """
        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)
        positions = self.regression_head(decoder_output)

        if self.classification_head is not None:
            classifications = self.classification_head(features[-1])
            return masks, positions, classifications
        else:
            return masks, positions, None

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        """
        Inference method. Calls forward in a no_grad context.
        """
        return self.forward(x)
