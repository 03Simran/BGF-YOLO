import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
       
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):

        offsets = self.offset_conv(x) 
        return self.deform_conv(x, offsets)

    def deform_conv(self, x, offsets):
        B, C, H, W = x.size()
        k = self.kernel_size
        
        # y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        # grid = torch.stack((x, y), dim=0).float().to(x.device)  
        # grid = grid.unsqueeze(0).repeat(B, 1, 1, 1) 
   
        offsets = offsets.view(B, k * k, 2, H, W).permute(0, 2, 1, 3, 4)  
        grid = grid.unsqueeze(2) + offsets  
        
  
        grid = grid.permute(0, 3, 4, 2, 1).reshape(B, H * W * k * k, 2)  
        
        x = x.unsqueeze(1).expand(-1, k * k, -1, -1, -1).reshape(B, C, -1) 
        sampled_features = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
       
        out = self.conv(sampled_features)
        return out
