import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
# --- 1. Import module ---
# Đảm bảo file load_data.py và densenet18.py nằm cùng thư mục
from load_data import LongitudinalCSVDataset
# Lưu ý: Nếu tên file của bạn là densenet18.py thì import thế này, nếu là densenet.py thì sửa lại
from densenet import ExpertDenseNet18 

import os 

def train_expert():
    # --- 2. Cấu hình ---
    BATCH_SIZE = 16
    LR = 1e-3
    EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = 'expert'
    os.makedirs(SAVE_DIR, exist_ok=True)
    SAVE_PATH = os.path.join(SAVE_DIR, 'expert_checkpoint.pth')
    
    # Cấu hình đường dẫn (Dựa trên ảnh bạn gửi)
    ROOT_DIR = '../filter_ds'             # Thư mục cha chứa NIfTI
    CSV_FILE = '../5_folds_split_3D/fold_1_train.csv' # <-- Thay 'ten_file.csv' bằng tên thật file CSV của bạn

    print(f"Training Expert Model on {DEVICE}...")

    # --- 3. Load Dữ liệu (SỬA LẠI ĐOẠN NÀY) ---
    dataset = LongitudinalCSVDataset(
        root_dir=ROOT_DIR, 
        csv_file=CSV_FILE, 
        target_shape=(160, 160, 96) # Đảm bảo đúng kích thước ảnh mới
    )
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # --- 4. Khởi tạo Model (SỬA LẠI ĐOẠN NÀY) ---
    # feature_mode=False: Để model trả về vector phân loại (cho việc training expert)
    # input_channels=1: Vì expert soi ảnh xám (1 kênh)
    model = ExpertDenseNet18(input_channels=1, num_classes=2, feature_mode=False).to(DEVICE)
    
    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # --- 5. Training Loop ---
    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for bl_img, nt_real, label in loop:
            # bl_img: Ảnh gốc (không dùng train expert)
            # nt_real: Ảnh thật M36 (Dùng để train expert)
            # label: Nhãn bệnh
            
            imgs = nt_real.to(DEVICE) 
            labels = label.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(imgs) 
            
            # Tính loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Thống kê
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())
        # In kết quả mỗi epoch
        if len(dataloader) > 0:
            avg_loss = running_loss / len(dataloader)
            epoch_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {epoch_acc:.2f}%")

    # --- 6. Lưu trọng số ---
    
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Đã lưu model expert tại: {SAVE_PATH}")

if __name__ == "__main__":
    train_expert()