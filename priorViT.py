import torch
import torch.nn as nn
# Import class backbone từ file densenet.py của bạn
# (Nếu tên file là densenet18.py thì sửa lại import cho đúng)
from densenet import ExpertDenseNet18

class PriorViT(nn.Module):
    def __init__(self, atlas):
        """
        atlas: Tensor hoặc Numpy array kích thước (1, D, H, W)
        """
        super(PriorViT, self).__init__()
        
        # --- 1. Xử lý Atlas ---
        # Chuyển đổi sang Tensor nếu là numpy
        if not torch.is_tensor(atlas):
            atlas = torch.tensor(atlas).float()
        
        # Đảm bảo atlas có 4 chiều (1, D, H, W)
        if atlas.dim() == 3:
            atlas = atlas.unsqueeze(0)
            
        # Đăng ký buffer để atlas tự động theo model sang GPU/CPU
        self.register_buffer('atlas', atlas)

        # --- 2. Backbone (3D CNN) ---
        # Input channel = 2 (1 ảnh gốc + 1 ảnh trọng số Atlas)
        # feature_mode=True: Để lấy feature map thay vì kết quả phân loại
        self.backbone = ExpertDenseNet18(input_channels=2, num_classes=2, feature_mode=True)
        
        # --- 3. Transformer (ViT) ---
        # [QUAN TRỌNG]: Sửa d_model = 128 để khớp với output của DenseNet18
        self.d_model = 128 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=4,            # 128 chia hết cho 4 (mỗi head 32 chiều)
            dim_feedforward=512, 
            dropout=0.3,
            batch_first=True    # Quan trọng: Input sẽ là (Batch, Seq_Len, Feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- 4. Classifier Head ---
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 2) # 2 classes: Normal (0) vs Disease (1)
        )

    def forward(self, x):
        # x: (Batch, 1, D, H, W)
        
        # --- Bước A: Tích hợp tri thức tiên nghiệm (Prior Knowledge) ---
        # Nhân ảnh đầu vào với Atlas
        x_weighted = x * self.atlas 
        
        # Nối ảnh gốc và ảnh weighted -> (Batch, 2, D, H, W)
        x_combined = torch.cat([x, x_weighted], dim=1)

        # --- Bước B: Trích xuất đặc trưng (CNN) ---
        # features shape: (Batch, 128, D', H', W') 
        # (Với ảnh 160x160x96 thì output khoảng 5x5x3)
        features = self.backbone(x_combined)
        
        # --- Bước C: Transformer ---
        # 1. Flatten không gian 3D thành chuỗi (Sequence)
        # (B, 128, D', H', W') -> (B, 128, N) với N = D'*H'*W'
        x_seq = features.flatten(2) 
        
        # 2. Đảo chiều để đưa Channel (128) xuống cuối làm Feature Dim
        # (B, 128, N) -> (B, N, 128)
        x_seq = x_seq.transpose(1, 2) 
        
        # 3. Qua Transformer
        # Output: (B, N, 128)
        x_trans = self.transformer(x_seq)
        
        # --- Bước D: Phân loại ---
        # Global Average Pooling trên toàn bộ chuỗi
        # (B, N, 128) -> (B, 128)
        x_cls = x_trans.mean(dim=1) 
        
        # Fully Connected
        logits = self.mlp_head(x_cls)
        
        return logits