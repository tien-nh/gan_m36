import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # Hiển thị tiến độ
import os

# --- Import modules ---
from load_data import LongitudinalCSVDataset
from unet import UNetGenerator3D
from discriminator import Discriminator3D
from SSIM import SSIM3D
# Import Expert để tính Loss
from densenet import ExpertDenseNet18 

def train_gan():
    # --- 1. Cấu hình Hyperparameters ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Batch size = 1 vì ảnh 160x160x96 rất nặng
    BATCH_SIZE = 8 
    LR = 1e-4
    EPOCHS = 200    # GAN cần train lâu
    
    # Trọng số Loss
    LAMBDA_ADV = 0.1
    LAMBDA_CLS = 1.0
    
    # Đường dẫn (Sửa lại cho đúng file CSV của bạn)
    ROOT_DIR = '../filter_ds'
    CSV_FILE = '../5_folds_split_3D/fold_1_train.csv'
    EXPERT_PATH = 'expert/expert_checkpoint.pth' # File bạn vừa train xong

    print(f"--- START TRAINING GAN on {DEVICE} ---")

    # --- 2. Load Dữ liệu ---
    dataset = LongitudinalCSVDataset(
        root_dir=ROOT_DIR, 
        csv_file=CSV_FILE, 
        target_shape=(160, 160, 96)
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # --- 3. Khởi tạo Models ---
    
    # A. Generator (U-Net)
    generator = UNetGenerator3D().to(DEVICE)
    
    # B. Discriminator
    discriminator = Discriminator3D().to(DEVICE)
    
    # C. Expert (Load pre-trained weights)
    print(f"Loading Expert model from {EXPERT_PATH}...")
    # Lưu ý: feature_mode=False vì ta cần output là logits để tính CrossEntropyLoss
    expert = ExpertDenseNet18(input_channels=1, num_classes=2, feature_mode=False).to(DEVICE)
    
    if os.path.exists(EXPERT_PATH):
        expert.load_state_dict(torch.load(EXPERT_PATH))
        print("Expert loaded successfully!")
    else:
        raise FileNotFoundError(f"Chưa tìm thấy file {EXPERT_PATH}. Hãy chạy train_expert.py trước!")
        
    expert.eval() # Đóng băng Expert (Chỉ dùng để chấm điểm)
    for param in expert.parameters():
        param.requires_grad = False

    # --- 4. Optimizers & Loss Functions ---
    opt_g = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    
    l1_crit = nn.L1Loss()
    adv_crit = nn.MSELoss() # LSGAN Loss
    cls_crit = nn.CrossEntropyLoss()
    ssim_3d = SSIM3D().to(DEVICE)

    # --- 5. Training Loop ---
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        g_loss_val = 0.0
        d_loss_val = 0.0
        
        for bl_img, nt_real, label in loop:
            bl_img = bl_img.to(DEVICE)
            nt_real = nt_real.to(DEVICE)
            label = label.to(DEVICE)
            
            # ==================================================================
            #  BƯỚC 1: GENERATE ẢNH GIẢ (1 LẦN DUY NHẤT)
            # ==================================================================
            fake_nt = generator(bl_img)

            # ==================================================================
            #  BƯỚC 2: TRAIN DISCRIMINATOR (Active từ Epoch 100)
            # ==================================================================
            loss_d = torch.tensor(0.0).to(DEVICE)
            
            if epoch >= 100:
                opt_d.zero_grad()
                
                # 2.1 Train với ảnh thật
                pred_real = discriminator(nt_real)
                loss_d_real = adv_crit(pred_real, torch.ones_like(pred_real))
                
                # 2.2 Train với ảnh giả (Dùng .detach() để ngắt gradient về G)
                pred_fake_detach = discriminator(fake_nt.detach())
                loss_d_fake = adv_crit(pred_fake_detach, torch.zeros_like(pred_fake_detach))
                
                # 2.3 Update D
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                opt_d.step()
            
            # ==================================================================
            #  BƯỚC 3: TRAIN GENERATOR (Luôn active)
            # ==================================================================
            opt_g.zero_grad()
            
            # 3.1 Pixel Losses
            loss_l1 = l1_crit(fake_nt, nt_real)
            loss_ssim = 1 - ssim_3d(fake_nt, nt_real)
            
            # 3.2 Adversarial Loss (G muốn lừa D)
            # Tái sử dụng fake_nt (vẫn giữ graph) đưa vào D
            pred_fake = discriminator(fake_nt)
            loss_adv = adv_crit(pred_fake, torch.ones_like(pred_fake))
            
            # Tổng hợp loss cơ bản
            loss_g = loss_l1 + loss_ssim + LAMBDA_ADV * loss_adv
            
            # 3.3 Expert Loss (Chỉ active từ Epoch 200)
            if epoch >= 200:
                pred_cls = expert(fake_nt) # Expert chấm điểm ảnh giả
                loss_cls = cls_crit(pred_cls, label) # So sánh với nhãn thật của bệnh nhân
                loss_g += LAMBDA_CLS * loss_cls
                
            # 3.4 Update G
            loss_g.backward()
            opt_g.step()
            
            # Update thanh progress bar
            g_loss_val = loss_g.item()
            d_loss_val = loss_d.item()
            loop.set_postfix(G_Loss=g_loss_val, D_Loss=d_loss_val)

        # --- 6. Lưu Checkpoint ---
        # Lưu định kỳ mỗi 50 epoch
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
            print(f"-> Saved checkpoint at epoch {epoch+1}")
            
        # Luôn lưu bản mới nhất
        torch.save(generator.state_dict(), 'generator_latest.pth')

    print("TRAINING COMPLETE!")

if __name__ == "__main__":
    train_gan()