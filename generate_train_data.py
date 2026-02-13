# generate_train_data.py
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import nibabel as nib
import numpy as np
import pandas as pd

from load_data import LongitudinalCSVDataset
from unet import UNetGenerator3D

def generate_train_data():
    # --- CẤU HÌNH ---
    CONFIG = {
        "csv_file": "./fold_1_train+val.csv",       # <--- Dùng tập TRAIN
        "root_dir": "../filter_ds",
        "generator_path": "models/generator_latest.pth",
        "output_dir": "Generated_M36",      # <--- Lưu vào folder TRAIN
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    DEVICE = torch.device(CONFIG["device"])

    # Load Data
    train_dataset = LongitudinalCSVDataset(
        root_dir=CONFIG["root_dir"],
        csv_file=CONFIG["csv_file"],
        target_shape=(160, 160, 96)
    )
    # Batch size = 1 để an toàn
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Load Generator
    print(f"Loading Generator from {CONFIG['generator_path']}...")
    generator = UNetGenerator3D().to(DEVICE)
    if os.path.exists(CONFIG["generator_path"]):
        generator.load_state_dict(torch.load(CONFIG["generator_path"], map_location=DEVICE))
    else:
        raise FileNotFoundError("Chưa train GAN xong! Không thấy file generator_latest.pth")
        
    generator.eval()

    print(f"Generating synthetic training images...")
    
    # Load CSV để lấy ID
    df = pd.read_csv(CONFIG["csv_file"])
    valid_classes = ["Class 1 (CN to CN)", "Class 5 (MCI to AD)"]
    df_filtered = df[df['class_label'].isin(valid_classes)].reset_index(drop=True)

    with torch.no_grad():
        for i, (bl_img, _, _) in enumerate(tqdm(train_loader)):
            bl_img = bl_img.to(DEVICE)
            
            # Sinh ảnh
            fake_m36 = generator(bl_img)
            
            # Lấy ID
            try:
                subj_id = df_filtered.iloc[i]['subject ID']
            except IndexError:
                continue # Skip nếu lỗi index
            
            # Lưu
            img_numpy = fake_m36.squeeze().cpu().numpy()
            save_path = os.path.join(CONFIG["output_dir"], f"{subj_id}.nii.gz")
            nifti_img = nib.Nifti1Image(img_numpy, affine=np.eye(4))
            nib.save(nifti_img, save_path)

    print("Done! Training data generated.")

if __name__ == "__main__":
    generate_train_data()