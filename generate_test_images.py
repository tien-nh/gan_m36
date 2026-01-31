# generate_test_images.py
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import nibabel as nib
import numpy as np
import pandas as pd

# Import modules cũ
from load_data import LongitudinalCSVDataset
from unet import UNetGenerator3D

def generate_test_data():
    # --- CẤU HÌNH ---
    CONFIG = {
        "csv_file": "../datacsv/fold_3_test.csv",      # File test của bạn
        "root_dir": "../filter_ds",          # Folder chứa dữ liệu gốc
        "generator_path": "models/generator_latest.pth", # Model GAN đã train
        "output_dir": "Generated_Test_M36",     # Nơi lưu ảnh sinh ra
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    DEVICE = torch.device(CONFIG["device"])

    # 1. Load Dataset Test
    # Lưu ý: Ta vẫn dùng class LongitudinalCSVDataset để load ảnh BL
    test_dataset = LongitudinalCSVDataset(
        root_dir=CONFIG["root_dir"],
        csv_file=CONFIG["csv_file"],
        target_shape=(160, 160, 96)
    )
    # Batch size = 1 để an toàn khi sinh ảnh và lưu file
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. Load Generator
    print(f"Loading Generator from {CONFIG['generator_path']}...")
    generator = UNetGenerator3D().to(DEVICE)
    generator.load_state_dict(torch.load(CONFIG["generator_path"], map_location=DEVICE))
    generator.eval()

    print(f"Start generating images for {len(test_dataset)} subjects...")
    
    # 3. Inference Loop
    # Cần đọc lại file CSV để lấy Subject ID tương ứng với từng index
    df = pd.read_csv(CONFIG["csv_file"])
    # Lọc giống logic trong Dataset để đảm bảo index khớp nhau
    valid_classes = ["Class 1 (CN to CN)", "Class 5 (MCI to AD)"]
    df_filtered = df[df['class_label'].isin(valid_classes)].reset_index(drop=True)

    with torch.no_grad():
        for i, (bl_img, _, _) in enumerate(tqdm(test_loader)):
            # bl_img: Ảnh Baseline thật
            bl_img = bl_img.to(DEVICE)
            
            # Sinh ảnh M36 giả
            fake_m36 = generator(bl_img)
            
            # Lấy Subject ID từ DataFrame để đặt tên file
            subj_id = df_filtered.iloc[i]['subject ID']
            
            # Convert về Numpy để lưu
            img_numpy = fake_m36.squeeze().cpu().numpy() # (160, 160, 96)
            
            # Lưu file .nii.gz
            save_path = os.path.join(CONFIG["output_dir"], f"{subj_id}.nii.gz")
            
            # Tạo Nifti Image (affine mặc định identity vì ta đã chuẩn hóa)
            nifti_img = nib.Nifti1Image(img_numpy, affine=np.eye(4))
            nib.save(nifti_img, save_path)

    print(f"Hoàn tất! Ảnh sinh ra đã lưu tại: {CONFIG['output_dir']}")

if __name__ == "__main__":
    generate_test_data()