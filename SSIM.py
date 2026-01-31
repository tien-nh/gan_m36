import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def create_window_3d(window_size, channel):
    def gaussian(window_size, sigma):
        # Tạo Gaussian kernel 1D
        gauss = torch.tensor([np.exp(-(x - window_size // 2)**2 / float(2 * sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    # Tạo 3D kernel bằng cách nhân outer product
    _3D_window = _2D_window.unsqueeze(2) @ _1D_window.t().unsqueeze(0) 
    
    # Reshape cho phù hợp Conv3d: (Out_channels, In_channels/Groups, D, H, W)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

def ssim_3d_func(img1, img2, window_size=11, size_average=True):
    """
    Hàm tính SSIM 3D thuần túy
    """
    channel = img1.size(1)
    # Tự động tạo window và đẩy sang device của ảnh (CPU/GPU)
    window = create_window_3d(window_size, channel).to(img1.device).type(img1.dtype)
    
    mu1 = F.conv3d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# --- CLASS WRAPPER (Phần quan trọng cần thêm) ---
class SSIM3D(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return ssim_3d_func(img1, img2, self.window_size, self.size_average)