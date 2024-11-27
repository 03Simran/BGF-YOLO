import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DeformableConv2d']

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Offset convolution: Predicts the sampling offsets
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size,  # 2 for (x, y) offsets
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Standard convolution for feature extraction
        self.regular_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding
        )

    def forward(self, x):
        # Compute offsets
        offsets = self.offset_conv(x)
        
        # Apply deformable convolution
        return self.deform_conv(x, offsets)

    def deform_conv(self, x, offsets):
        B, C, H, W = x.shape  # Input tensor shape
        kernel_H, kernel_W = self.kernel_size, self.kernel_size

    # Generate a base grid
        y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=x.device),
        torch.arange(W, device=x.device),
        indexing="ij"
        )
        base_grid = torch.stack((x_coords, y_coords), dim=-1).float()  # Shape: (H, W, 2)

    # Add offsets to the base grid
    # Ensure offsets shape: (B, H, W, 2 * kernel_H * kernel_W)
        assert offsets.shape[1] == 2 * kernel_H * kernel_W, \
            f"Expected offsets to have {2 * kernel_H * kernel_W} channels, got {offsets.shape[1]}"

    # Reshape offsets to match (B, H, W, 2, kernel_H * kernel_W)
        offsets = offsets.view(B, kernel_H * kernel_W, 2, H, W).permute(0, 3, 4, 1, 2)  # (B, H, W, kernel_H * kernel_W, 2)

    # Add offsets to the base grid
        offset_grid = base_grid.unsqueeze(0).unsqueeze(-2).repeat(B, 1, 1, kernel_H * kernel_W, 1)  # (B, H, W, kernel_H * kernel_W, 2)
        offset_grid = offset_grid + offsets  # Add offsets to base grid

    # Flatten grid to prepare for sampling
        offset_grid = offset_grid.view(B, H, W, kernel_H * kernel_W, 2)  # Flatten (B, H, W, kernel_H * kernel_W, 2)

    # Normalize grid for F.grid_sample
        norm_grid = torch.zeros_like(offset_grid)
        norm_grid[..., 0] = (2 * offset_grid[..., 0] / max(W - 1, 1)) - 1  # Normalize x to [-1, 1]
        norm_grid[..., 1] = (2 * offset_grid[..., 1] / max(H - 1, 1)) - 1  # Normalize y to [-1, 1]

    # Sample features with grid_sample
        sampled_features = torch.zeros(
        (B, C, H, W, kernel_H * kernel_W), device=x.device
      )  # Prepare to accumulate sampled features
        for i in range(kernel_H * kernel_W):
            sampled_features[..., i] = F.grid_sample(
            x, norm_grid[..., i, :], mode='bilinear', padding_mode='zeros', align_corners=True
        ).squeeze(1)

    # Combine sampled features by summing or other aggregation
        sampled_features = sampled_features.sum(dim=-1)  # Aggregating across kernel positions (H * W)

    # Apply regular convolution to sampled features
        out = self.regular_conv(sampled_features)
        return out