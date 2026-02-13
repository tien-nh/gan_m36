# evaluate_model.py
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report

# Import model và dataset
from priorViT import PriorViT
from train_vit_classifier import GeneratedDataset # Tận dụng lại class này

def evaluate():
    # --- CẤU HÌNH ---
    CONFIG = {
        "gen_test_dir": "Generated_Test_M36", # Folder ảnh vừa sinh ở Bước 1
        "csv_file": "../5_folds_split_3D/fold_1_test.csv",          # File nhãn thật
        "atlas_path": "npy/disease_atlas.npy",      # File Atlas
        "model_path": "model_vit/prior_vit_latest.pth",    # Model ViT đã train
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    DEVICE = torch.device(CONFIG["device"])

    # 1. Load Data
    print("Loading Test Data...")
    dataset = GeneratedDataset(CONFIG["gen_test_dir"], CONFIG["csv_file"])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # 2. Load Model & Atlas
    print(f"Loading Model from {CONFIG['model_path']}...")
    if not os.path.exists(CONFIG['atlas_path']):
        raise FileNotFoundError("Thiếu file Atlas!")
        
    atlas = np.load(CONFIG['atlas_path'])
    model = PriorViT(atlas=atlas).to(DEVICE)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=DEVICE))
    model.eval()

    # 3. Evaluation Loop
    all_preds = []
    all_labels = []
    all_probs = [] # Xác suất để tính AUC

    print("Running Inference...")
    with torch.no_grad():
        for imgs, labels in tqdm(dataloader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward
            outputs = model(imgs) # Logits
            probs = torch.softmax(outputs, dim=1) # Probability
            
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1]) # Lấy xác suất lớp 1 (Disease)

    # 4. Tính Metrics
    acc = accuracy_score(all_labels, all_preds)
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0 # Trường hợp chỉ có 1 class trong test set

    

    print("\n" + "="*30)
    print("KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST")
    print("="*30)
    print(f"Accuracy (Độ chính xác): {acc:.4f} ({acc*100:.2f}%)")
    print(f"Sensitivity (Độ nhạy - Tìm ra bệnh): {sensitivity:.4f}")
    print(f"Specificity (Độ đặc hiệu - Loại trừ lành tính): {specificity:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(f"TP: {tp} | FN: {fn}")
    print(f"FP: {fp} | TN: {tn}")
    print("="*30)

    import sklearn.metrics as metrics
    f1 = metrics.f1_score(all_labels, all_preds)

    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    evaluate()