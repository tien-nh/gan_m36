# create_atlas.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import Dataset của bạn
from load_data import LongitudinalCSVDataset

def generate_disease_atlas():
    # --- Cấu hình ---
    ROOT_DIR = '../filter_ds'
    CSV_FILE = '../datacsv/fold_3_train.csv' # Dùng tập train để tính Atlas
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("--- Đang tính toán Disease Atlas từ dữ liệu thật (Real BL vs Real M36) ---")
    
    # Load dữ liệu thật
    dataset = LongitudinalCSVDataset(
        root_dir=ROOT_DIR, 
        csv_file=CSV_FILE, 
        target_shape=(160, 160, 96)
    )
    # Batch size lớn chút cho nhanh, không cần gradient nên không lo OOM
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    
    accumulation = None
    count = 0
    
    for bl_imgs, m36_imgs, _ in tqdm(dataloader):
        # bl_imgs: Ảnh chụp lần đầu (Real Baseline)
        # m36_imgs: Ảnh chụp sau 36 tháng (Real M36)
        
        # 1. Tính Residual (Hiệu số tuyệt đối) |BL - M36|
        # Đây chính là vùng não đã thay đổi sau 3 năm
        residual = torch.abs(bl_imgs - m36_imgs)
        
        # 2. Cộng dồn (Pixel-level merging operation)
        if accumulation is None:
            accumulation = torch.zeros_like(residual[0]) # (1, D, H, W)
        
        # Sum theo batch rồi cộng vào tổng
        accumulation += torch.sum(residual, dim=0)
        count += bl_imgs.size(0) # Đếm tổng số bệnh nhân
        
    # 3. Tính trung bình
    mean_atlas = accumulation / count
    
    # 4. Normalization (Chuẩn hóa về 0-1)
    # Để làm map trọng số thì giá trị phải từ 0 đến 1
    max_val = torch.max(mean_atlas)
    min_val = torch.min(mean_atlas)
    atlas_normalized = (mean_atlas - min_val) / (max_val - min_val)
    
    # 5. Lưu lại
    save_path = 'npy/disease_atlas.npy'
    np.save(save_path, atlas_normalized.cpu().numpy())
    print(f"Đã lưu Atlas chuẩn bài báo tại: {save_path}")
    print(f"Kích thước: {atlas_normalized.shape}")

if __name__ == "__main__":
    generate_disease_atlas()