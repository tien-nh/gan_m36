import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import glob
import pandas as pd
import nibabel as nib
import wandb
from tqdm import tqdm

# Import model
from priorViT import PriorViT
# Import dataset gốc để load dữ liệu thật cho tập Val
from load_data import LongitudinalCSVDataset 

# --- 1. Dataset cho ảnh sinh ra (Dùng cho TRAINING) ---
class GeneratedDataset(Dataset):
    def __init__(self, gen_data_dir, csv_file, target_shape=(160, 160, 96)):
        self.file_paths = glob.glob(os.path.join(gen_data_dir, '*.nii.gz'))
        self.target_shape = target_shape
        self.labels_map = self._load_label_dict(csv_file)
        
        self.valid_files = []
        for f in self.file_paths:
            filename = os.path.basename(f)
            subj_id = filename.replace('.nii.gz', '')
            if subj_id in self.labels_map:
                self.valid_files.append(f)
        
        print(f"-> Train Dataset (Generated): Loaded {len(self.valid_files)} images.")

    def _load_label_dict(self, csv_file):
        df = pd.read_csv(csv_file)
        label_dict = {}
        valid_classes = ["Class 1 (CN to CN)", "Class 5 (MCI to AD)"]
        for _, row in df.iterrows():
            if row['class_label'] in valid_classes:
                subj_id = row['subject ID']
                label = 0 if "Class 1" in row['class_label'] else 1
                label_dict[subj_id] = label
        return label_dict
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        path = self.valid_files[idx]
        filename = os.path.basename(path)
        subj_id = filename.replace('.nii.gz', '')
        
        img = nib.load(path).get_fdata()
        # Min-Max Normalization (0-1)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        label = self.labels_map[subj_id]
        
        return img_tensor, torch.tensor(label, dtype=torch.long)

# --- 2. Main Training Loop ---
def train_vit_classifier():
    # ================= CẤU HÌNH =================
    CONFIG = {
        "gen_data_dir": "Generated_M36",  # Folder ảnh GAN (Train)
        "train_csv": "../datacsv/fold_3_train.csv",    # CSV Train
        "val_csv": "../datacsv/fold_3_val.csv",        # CSV Validation
        "root_dir_real": "../filter_ds",     # Folder chứa ảnh thật (cho Val)
        "atlas_path": "../npy/disease_atlas.npy",
        "batch_size": 4,
        "lr": 1e-4,
        "epochs": 60,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "save_path": "model_vit/prior_vit_best.pth"
    }
    # ============================================

    wandb.init(project="AD_PriorViT_Classifier", config=CONFIG, name="Run_Scheduler_Tuning")
    DEVICE = torch.device(CONFIG["device"])
    
    # 1. Load Atlas
    if os.path.exists(CONFIG["atlas_path"]):
        atlas_numpy = np.load(CONFIG["atlas_path"])
        print(f"Loaded Atlas shape: {atlas_numpy.shape}")
    else:
        raise FileNotFoundError("Thiếu file disease_atlas.npy! Hãy chạy create_atlas.py trước.")

    # 2. Tạo Datasets
    # A. TRAIN SET: Dùng ảnh sinh ra (Generated)
    train_dataset = GeneratedDataset(CONFIG["gen_data_dir"], CONFIG["train_csv"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)

    # B. VAL SET: Dùng ảnh THẬT (Real M36) từ file CSV Val
    val_dataset = LongitudinalCSVDataset(
        root_dir=CONFIG["root_dir_real"],
        csv_file=CONFIG["val_csv"],
        target_shape=(160, 160, 96)
    )
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
    print(f"-> Val Dataset (Real): Loaded {len(val_dataset)} pairs.")

    # 3. Model & Optimizer
    model = PriorViT(atlas=atlas_numpy).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- [MỚI] LEARNING RATE SCHEDULER ---
    # Tự động giảm LR đi 10 lần (factor=0.1) nếu Val Loss không giảm sau 5 epoch (patience=5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    best_val_acc = 0.0

    print("--- START TRAINING WITH SCHEDULER ---")
    
    for epoch in range(CONFIG["epochs"]):
        # ================= TRAIN PHASE =================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{CONFIG['epochs']}] [Train]")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # ================= VALIDATION PHASE =================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for _, m36_imgs, labels in val_loader:
                m36_imgs, labels = m36_imgs.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(m36_imgs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        # --- [MỚI] STEP SCHEDULER ---
        # Báo cáo Val Loss cho Scheduler để nó quyết định có giảm LR không
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # ================= LOGGING & SAVING =================
        print(f"-> Summary Epoch {epoch+1}:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"   LR: {current_lr}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "learning_rate": current_lr # <-- Log thêm cái này để vẽ đồ thị LR
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"   [>>>] New Best Model Saved! (Val Acc: {val_acc:.2f}%)")
            
        torch.save(model.state_dict(), "prior_vit_latest.pth")

    print(f"Training Complete. Best Val Acc: {best_val_acc:.2f}%")
    wandb.finish()

if __name__ == "__main__":
    train_vit_classifier()