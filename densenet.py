import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], 1)

class DenseBlock3D(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer3D(in_channels + i * growth_rate, growth_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class TransitionLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x))))

class ExpertDenseNet18(nn.Module):
    # --- [SỬA ĐỔI QUAN TRỌNG]: Thêm input_channels và feature_mode ---
    def __init__(self, input_channels=1, num_classes=2, feature_mode=False):
        super().__init__()
        self.feature_mode = feature_mode  # Cờ để kiểm soát output (Feature hay Logits)
        self.growth_rate = 32
        num_planes = 64
        
        # Initial Conv
        # --- [SỬA ĐỔI]: Dùng input_channels thay vì số 1 cố định ---
        self.pre_layers = nn.Sequential(
            nn.Conv3d(input_channels, num_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(3, stride=2, padding=1)
        )
        
        # Dense Blocks (Config for DenseNet-18)
        self.block1 = DenseBlock3D(2, num_planes, self.growth_rate)
        num_planes += 2 * self.growth_rate
        self.trans1 = TransitionLayer3D(num_planes, num_planes // 2)
        num_planes = num_planes // 2
        
        self.block2 = DenseBlock3D(2, num_planes, self.growth_rate)
        num_planes += 2 * self.growth_rate
        self.trans2 = TransitionLayer3D(num_planes, num_planes // 2)
        num_planes = num_planes // 2
        
        self.block3 = DenseBlock3D(2, num_planes, self.growth_rate)
        num_planes += 2 * self.growth_rate
        self.trans3 = TransitionLayer3D(num_planes, num_planes // 2)
        num_planes = num_planes // 2
        
        self.block4 = DenseBlock3D(2, num_planes, self.growth_rate)
        num_planes += 2 * self.growth_rate
        
        self.bn_final = nn.BatchNorm3d(num_planes)
        
        # --- [SỬA ĐỔI]: Chỉ tạo classifier khi KHÔNG ở chế độ feature_mode ---
        if not self.feature_mode:
            self.classifier = nn.Linear(num_planes, num_classes)

    def forward(self, x):
        out = self.pre_layers(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.trans3(self.block3(out))
        out = self.block4(out)
        out = F.relu(self.bn_final(out))
        
        # --- [SỬA ĐỔI]: Trả về feature map 3D nếu đang cần dùng cho ViT ---
        if self.feature_mode:
            return out 
        
        # Logic cũ cho việc train Expert (Classification)
        out = F.adaptive_avg_pool3d(out, 1)
        out = torch.flatten(out, 1)
        return self.classifier(out)